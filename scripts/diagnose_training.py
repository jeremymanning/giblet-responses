#!/usr/bin/env python3
"""
Diagnostic script for analyzing training issues.

This script loads a trained model and analyzes:
1. Gradient flow through the network
2. Data statistics (mean, std, range) for each modality
3. Loss component magnitudes
4. Learning rate schedule behavior
5. Weight initialization statistics

Usage:
    python scripts/diagnose_training.py --config configs/training/production_500epoch_config.yaml \
                                        --checkpoint checkpoints/checkpoint_epoch_15.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.dataset import MultimodalDataset
from giblet.models.autoencoder import create_autoencoder
from giblet.training.losses import CombinedAutoEncoderLoss
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def analyze_data_statistics(dataset, n_samples=100):
    """
    Analyze data statistics for all modalities.

    Returns:
        dict: Statistics for each modality (mean, std, min, max, range)
    """
    print("\n" + "=" * 80)
    print("DATA STATISTICS ANALYSIS")
    print("=" * 80)

    stats = {
        'video': {'values': []},
        'audio': {'values': []},
        'text': {'values': []},
        'fmri': {'values': []},
    }

    # Sample from dataset
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    print(f"\nAnalyzing {n_samples} samples from dataset...")

    for idx in indices:
        sample = dataset[idx]
        for key in ['video', 'audio', 'text', 'fmri']:
            if key in sample:
                values = sample[key].numpy() if isinstance(sample[key], torch.Tensor) else sample[key]
                stats[key]['values'].append(values.flatten())

    # Compute statistics
    results = {}
    for key in ['video', 'audio', 'text', 'fmri']:
        if stats[key]['values']:
            all_values = np.concatenate(stats[key]['values'])
            results[key] = {
                'mean': float(np.mean(all_values)),
                'std': float(np.std(all_values)),
                'min': float(np.min(all_values)),
                'max': float(np.max(all_values)),
                'range': float(np.max(all_values) - np.min(all_values)),
                'median': float(np.median(all_values)),
            }

            print(f"\n{key.upper()}:")
            print(f"  Mean:   {results[key]['mean']:12.4f}")
            print(f"  Std:    {results[key]['std']:12.4f}")
            print(f"  Min:    {results[key]['min']:12.4f}")
            print(f"  Max:    {results[key]['max']:12.4f}")
            print(f"  Range:  {results[key]['range']:12.4f}")
            print(f"  Median: {results[key]['median']:12.4f}")

    return results


def analyze_model_initialization(model):
    """
    Analyze model weight initialization statistics.

    Returns:
        dict: Statistics for each layer's weights
    """
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION ANALYSIS")
    print("=" * 80)

    results = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.cpu().numpy()
            results[name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'shape': list(param.shape),
            }

            # Only print summary for key layers
            if any(x in name for x in ['encoder', 'decoder', 'bottleneck']):
                print(f"\n{name}:")
                print(f"  Shape:  {param.shape}")
                print(f"  Mean:   {results[name]['mean']:12.6f}")
                print(f"  Std:    {results[name]['std']:12.6f}")
                print(f"  Range:  [{results[name]['min']:.6f}, {results[name]['max']:.6f}]")

    return results


def analyze_gradient_flow(model, dataloader, criterion, device):
    """
    Analyze gradient magnitudes through the network.

    Returns:
        dict: Gradient statistics for each layer
    """
    print("\n" + "=" * 80)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 80)

    model.train()

    # Get one batch
    batch = next(iter(dataloader))
    video = batch['video'].to(device)
    audio = batch['audio'].to(device)
    text = batch['text'].to(device)
    fmri = batch['fmri'].to(device)

    print(f"\nBatch shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Text:  {text.shape}")
    print(f"  fMRI:  {fmri.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    outputs = model(video, audio, text, fmri)

    # Compute loss
    print("Computing loss...")
    loss_output = criterion(
        outputs['video_recon'],
        video,
        outputs['audio_recon'],
        audio,
        outputs['text_recon'],
        text,
        outputs['fmri_pred'],
        fmri,
    )

    if isinstance(loss_output, tuple):
        loss, loss_dict = loss_output
    else:
        loss = loss_output
        loss_dict = {}

    print(f"\nLoss components:")
    print(f"  Total loss: {loss.item():.4f}")
    for key, val in loss_dict.items():
        print(f"  {key}: {val.item():.4f}")

    # Backward pass
    print("\nRunning backward pass...")
    model.zero_grad()
    loss.backward()

    # Analyze gradients
    print("\nGradient statistics:")
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad = param.grad.cpu().numpy()
            grad_stats[name] = {
                'mean': float(np.mean(grad)),
                'std': float(np.std(grad)),
                'min': float(np.min(grad)),
                'max': float(np.max(grad)),
                'norm': float(np.linalg.norm(grad)),
                'mean_abs': float(np.mean(np.abs(grad))),
            }

            # Print summary for key layers
            if any(x in name for x in ['encoder', 'decoder', 'bottleneck']):
                print(f"\n{name}:")
                print(f"  Mean:     {grad_stats[name]['mean']:12.6e}")
                print(f"  Std:      {grad_stats[name]['std']:12.6e}")
                print(f"  Mean |g|: {grad_stats[name]['mean_abs']:12.6e}")
                print(f"  Norm:     {grad_stats[name]['norm']:12.6e}")
                print(f"  Range:    [{grad_stats[name]['min']:.6e}, {grad_stats[name]['max']:.6e}]")

    return grad_stats, loss_dict


def analyze_learning_rate_schedule(config):
    """
    Analyze learning rate schedule over training.

    Returns:
        dict: Learning rate at key epochs
    """
    print("\n" + "=" * 80)
    print("LEARNING RATE SCHEDULE ANALYSIS")
    print("=" * 80)

    lr = config.get('learning_rate', 1e-4)
    warmup_epochs = config.get('warmup_epochs', 10)
    num_epochs = config.get('num_epochs', 500)
    min_lr = config.get('min_lr', 1e-6)
    scheduler_type = config.get('scheduler_type', 'cosine')

    print(f"\nConfiguration:")
    print(f"  Initial LR:     {lr:.6e}")
    print(f"  Warmup epochs:  {warmup_epochs}")
    print(f"  Total epochs:   {num_epochs}")
    print(f"  Min LR:         {min_lr:.6e}")
    print(f"  Scheduler type: {scheduler_type}")

    # Simulate learning rate schedule
    print(f"\nLearning rate at key epochs:")

    # During warmup (linear from 0.1x to 1.0x)
    for epoch in [0, 5, warmup_epochs-1]:
        if epoch < warmup_epochs:
            factor = 0.1 + 0.9 * (epoch / warmup_epochs)
            epoch_lr = lr * factor
            print(f"  Epoch {epoch:3d}: {epoch_lr:.6e} (warmup)")

    # After warmup (cosine annealing)
    if scheduler_type == 'cosine':
        for epoch in [warmup_epochs, 50, 100, 250, num_epochs-1]:
            if epoch >= warmup_epochs:
                # Cosine annealing formula
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                epoch_lr = min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                print(f"  Epoch {epoch:3d}: {epoch_lr:.6e}")

    return {
        'initial_lr': lr,
        'warmup_epochs': warmup_epochs,
        'min_lr': min_lr,
        'scheduler_type': scheduler_type,
    }


def analyze_loss_components_scale(model, dataloader, criterion, device, n_batches=10):
    """
    Analyze the relative scale of different loss components.

    Returns:
        dict: Statistics for each loss component
    """
    print("\n" + "=" * 80)
    print("LOSS COMPONENT SCALE ANALYSIS")
    print("=" * 80)

    model.eval()

    loss_history = {
        'total': [],
        'reconstruction': [],
        'fmri': [],
        'video': [],
        'audio': [],
        'text': [],
    }

    print(f"\nAnalyzing {n_batches} batches...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            fmri = batch['fmri'].to(device)

            # Forward pass
            outputs = model(video, audio, text, fmri)

            # Compute loss
            loss_output = criterion(
                outputs['video_recon'],
                video,
                outputs['audio_recon'],
                audio,
                outputs['text_recon'],
                text,
                outputs['fmri_pred'],
                fmri,
            )

            if isinstance(loss_output, tuple):
                loss, loss_dict = loss_output
            else:
                loss = loss_output
                loss_dict = {}

            loss_history['total'].append(loss.item())
            for key in ['reconstruction_loss', 'fmri_loss', 'video_loss', 'audio_loss', 'text_loss']:
                if key in loss_dict:
                    short_key = key.replace('_loss', '')
                    loss_history[short_key].append(loss_dict[key].item())

    # Compute statistics
    print("\nLoss component statistics:")
    results = {}
    for key, values in loss_history.items():
        if values:
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            print(f"\n{key.upper()}:")
            print(f"  Mean: {results[key]['mean']:12.4f}")
            print(f"  Std:  {results[key]['std']:12.4f}")
            print(f"  Min:  {results[key]['min']:12.4f}")
            print(f"  Max:  {results[key]['max']:12.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Diagnose training issues')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (optional)')
    parser.add_argument('--data-samples', type=int, default=100,
                        help='Number of data samples to analyze')
    parser.add_argument('--loss-batches', type=int, default=10,
                        help='Number of batches for loss analysis')
    parser.add_argument('--output', type=str, default='diagnostic_report.txt',
                        help='Output file for report')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_autoencoder(
        video_height=config['model']['video_height'],
        video_width=config['model']['video_width'],
        audio_mels=config['model']['audio_mels'],
        text_dim=config['model']['text_dim'],
        n_voxels=config['model']['n_voxels'],
        bottleneck_dim=config['model']['bottleneck_dim'],
        video_features=config['model']['video_features'],
        audio_features=config['model']['audio_features'],
        text_features=config['model']['text_features'],
        decoder_hidden_dim=config['model']['decoder_hidden_dim'],
        decoder_dropout=config['model']['decoder_dropout'],
        use_encodec=config['model']['use_encodec'],
        audio_frames_per_tr=config['model']['audio_frames_per_tr'],
        gradient_checkpointing=False,  # Disable for diagnostics
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = MultimodalDataset(
        data_dir=config['data']['data_dir'],
        subjects=config['data']['subjects'],
        split='train',
        apply_hrf=config['data']['apply_hrf'],
        mode=config['data']['mode'],
        tr=config['data']['tr'],
        frame_skip=config['data']['frame_skip'],
        use_encodec=config['model']['use_encodec'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Single process for diagnostics
    )

    # Create loss function
    criterion = CombinedAutoEncoderLoss(
        reconstruction_weight=config.get('reconstruction_weight', 1.0),
        fmri_weight=config.get('fmri_weight', 1.0),
        video_weight=config.get('video_weight', 1.0),
        audio_weight=config.get('audio_weight', 1.0),
        text_weight=config.get('text_weight', 1.0),
        fmri_loss_type=config.get('fmri_loss_type', 'mse'),
    ).to(device)

    # Run diagnostics
    print("\n" + "=" * 80)
    print("RUNNING DIAGNOSTICS")
    print("=" * 80)

    # 1. Data statistics
    data_stats = analyze_data_statistics(dataset, n_samples=args.data_samples)

    # 2. Model initialization
    init_stats = analyze_model_initialization(model)

    # 3. Learning rate schedule
    lr_schedule = analyze_learning_rate_schedule(config)

    # 4. Gradient flow
    grad_stats, loss_dict = analyze_gradient_flow(model, dataloader, criterion, device)

    # 5. Loss component scales
    loss_stats = analyze_loss_components_scale(model, dataloader, criterion, device,
                                                n_batches=args.loss_batches)

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

    # Save full report
    print(f"\nSaving full report to {args.output}...")
    with open(args.output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING DIAGNOSTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {device}\n\n")

        import json
        f.write("\n\nDATA STATISTICS:\n")
        f.write(json.dumps(data_stats, indent=2))

        f.write("\n\nLEARNING RATE SCHEDULE:\n")
        f.write(json.dumps(lr_schedule, indent=2))

        f.write("\n\nLOSS COMPONENT SCALES:\n")
        f.write(json.dumps(loss_stats, indent=2))

        f.write("\n\nGRADIENT STATISTICS (key layers):\n")
        for name, stats in grad_stats.items():
            if any(x in name for x in ['encoder', 'decoder', 'bottleneck']):
                f.write(f"\n{name}:\n")
                f.write(json.dumps(stats, indent=2))

    print(f"Report saved to {args.output}")


if __name__ == '__main__':
    main()

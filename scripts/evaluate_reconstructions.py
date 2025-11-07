"""
Evaluate model reconstructions on test data.

This script loads a trained checkpoint and generates reconstructions for:
- Video frames
- Audio (EnCodec codes)
- Text embeddings (CLIP)
- fMRI voxel activations

Outputs visualizations showing ground truth vs reconstructions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from giblet.data.dataset import MultimodalDataset
from giblet.models.autoencoder import MultimodalAutoencoder
import yaml


def load_checkpoint(checkpoint_path, config):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    # Create model
    print("\nCreating model...")
    model = MultimodalAutoencoder(
        video_height=config.model.video_height,
        video_width=config.model.video_width,
        audio_mels=config.model.audio_mels,
        text_dim=config.model.text_dim,
        n_voxels=config.model.n_voxels,
        bottleneck_dim=config.model.bottleneck_dim,
        video_features=config.model.video_features,
        audio_features=config.model.audio_features,
        text_features=config.model.text_features,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
        decoder_dropout=config.model.decoder_dropout,
        use_encodec=config.model.use_encodec,
        audio_frames_per_tr=config.model.audio_frames_per_tr,
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("No model state dict found in checkpoint")

    model.eval()
    print("Model loaded successfully!")

    return model, checkpoint


def create_test_dataloader(config):
    """Create test dataloader."""
    print("\nCreating test dataset...")

    # Use validation split as test for now
    dataset = MultimodalDataset(
        data_dir=config.data.data_dir,
        subjects=config.data.subjects,
        split='val',  # Use validation as test
        apply_hrf=config.data.apply_hrf,
        mode=config.data.mode,
        frame_skip=config.data.frame_skip,
        fps=config.data.fps,
        tr=config.data.tr,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Test samples: {len(dataset)}")
    return dataloader


def visualize_video_reconstruction(video_gt, video_recon, save_path, num_frames=8):
    """Visualize video frame reconstructions."""
    # video_gt, video_recon: (batch, channels, height, width) or flattened

    # Reshape if flattened
    if video_gt.dim() == 2:
        # Reshape to (batch, H, W, C)
        batch_size = video_gt.size(0)
        video_dim = video_gt.size(1)
        height = int(np.sqrt(video_dim / 3))
        width = height
        channels = 3
        video_gt = video_gt.view(batch_size, height, width, channels)
        video_recon = video_recon.view(batch_size, height, width, channels)

    # Take first sample from batch
    video_gt = video_gt[0].cpu().numpy()
    video_recon = video_recon[0].cpu().numpy()

    # Normalize to [0, 1] for display
    video_gt = np.clip(video_gt / 255.0 if video_gt.max() > 1 else video_gt, 0, 1)
    video_recon = np.clip(video_recon / 255.0 if video_recon.max() > 1 else video_recon, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Ground truth
    axes[0].imshow(video_gt)
    axes[0].set_title('Ground Truth Video Frame', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Reconstruction
    axes[1].imshow(video_recon)
    axes[1].set_title('Reconstructed Video Frame', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved video reconstruction to {save_path}")


def visualize_audio_reconstruction(audio_gt, audio_recon, save_path):
    """Visualize audio code reconstructions."""
    # audio_gt, audio_recon: (batch, n_codes) - EnCodec codes

    audio_gt = audio_gt[0].cpu().numpy()
    audio_recon = audio_recon[0].cpu().numpy()

    # If codes, show first 100 codes
    if len(audio_gt.shape) == 1:
        audio_gt = audio_gt[:100]
        audio_recon = audio_recon[:100]

    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    # Ground truth
    axes[0].plot(audio_gt, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Ground Truth Audio Codes (first 100)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Code Value')
    axes[0].grid(True, alpha=0.3)

    # Reconstruction
    axes[1].plot(audio_recon, 'r-', linewidth=0.5, alpha=0.7)
    axes[1].set_title('Reconstructed Audio Codes (first 100)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Code Index')
    axes[1].set_ylabel('Code Value')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved audio reconstruction to {save_path}")


def visualize_text_reconstruction(text_gt, text_recon, save_path):
    """Visualize text embedding reconstructions."""
    # text_gt, text_recon: (batch, text_dim) - CLIP embeddings

    text_gt = text_gt[0].cpu().numpy()
    text_recon = text_recon[0].cpu().numpy()

    # Show first 100 dimensions
    text_gt = text_gt[:100]
    text_recon = text_recon[:100]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Ground truth
    axes[0].plot(text_gt, 'b-', linewidth=1, alpha=0.7, label='Ground Truth')
    axes[0].set_title('Ground Truth Text Embeddings (CLIP, first 100 dims)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Embedding Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction
    axes[1].plot(text_recon, 'r-', linewidth=1, alpha=0.7, label='Reconstruction')
    axes[1].set_title('Reconstructed Text Embeddings (first 100 dims)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Embedding Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Overlay comparison
    axes[2].plot(text_gt, 'b-', linewidth=1, alpha=0.5, label='Ground Truth')
    axes[2].plot(text_recon, 'r--', linewidth=1, alpha=0.5, label='Reconstruction')
    axes[2].set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Embedding Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved text embedding reconstruction to {save_path}")


def visualize_fmri_reconstruction(fmri_gt, fmri_recon, save_path):
    """Visualize fMRI voxel reconstructions."""
    # fmri_gt, fmri_recon: (batch, n_voxels)

    fmri_gt = fmri_gt[0].cpu().numpy()
    fmri_recon = fmri_recon[0].cpu().numpy()

    # Sample 1000 random voxels for visualization
    n_sample = min(1000, len(fmri_gt))
    indices = np.random.choice(len(fmri_gt), n_sample, replace=False)
    indices = np.sort(indices)

    fmri_gt_sample = fmri_gt[indices]
    fmri_recon_sample = fmri_recon[indices]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Ground truth
    axes[0].plot(fmri_gt_sample, 'b-', linewidth=0.5, alpha=0.7, label='Ground Truth')
    axes[0].set_title(f'Ground Truth fMRI Voxel Activations (sampled {n_sample} voxels)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Activation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction
    axes[1].plot(fmri_recon_sample, 'r-', linewidth=0.5, alpha=0.7, label='Reconstruction')
    axes[1].set_title(f'Reconstructed fMRI Voxel Activations (sampled {n_sample} voxels)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Activation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Scatter plot: GT vs Reconstruction
    axes[2].scatter(fmri_gt_sample, fmri_recon_sample, alpha=0.3, s=5)
    axes[2].plot([fmri_gt_sample.min(), fmri_gt_sample.max()],
                 [fmri_gt_sample.min(), fmri_gt_sample.max()],
                 'k--', linewidth=1, alpha=0.5, label='Perfect Reconstruction')
    axes[2].set_xlabel('Ground Truth Activation')
    axes[2].set_ylabel('Reconstructed Activation')
    axes[2].set_title('Ground Truth vs Reconstruction Correlation', fontsize=12, fontweight='bold')

    # Calculate correlation
    corr = np.corrcoef(fmri_gt, fmri_recon)[0, 1]
    axes[2].text(0.05, 0.95, f'Correlation: {corr:.4f}',
                transform=axes[2].transAxes,
                fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved fMRI reconstruction to {save_path}")
    print(f"    fMRI correlation: {corr:.4f}")


def evaluate_model(model, dataloader, output_dir, num_samples=5, device='cpu'):
    """Evaluate model and save visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating model on {num_samples} test samples...")
    print(f"Output directory: {output_dir}")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            print(f"\nSample {i+1}/{num_samples}:")

            # Move batch to device
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            fmri = batch['fmri'].to(device)

            # Forward pass
            outputs = model(video, audio, text)

            # Extract reconstructions
            video_recon = outputs['video_recon']
            audio_recon = outputs['audio_recon']
            text_recon = outputs['text_recon']
            fmri_pred = outputs['predicted_fmri']

            # Create visualizations
            sample_dir = output_dir / f"sample_{i+1}"
            sample_dir.mkdir(exist_ok=True)

            visualize_video_reconstruction(
                video, video_recon,
                sample_dir / "video_reconstruction.png"
            )

            visualize_audio_reconstruction(
                audio, audio_recon,
                sample_dir / "audio_reconstruction.png"
            )

            visualize_text_reconstruction(
                text, text_recon,
                sample_dir / "text_reconstruction.png"
            )

            visualize_fmri_reconstruction(
                fmri, fmri_pred,
                sample_dir / "fmri_reconstruction.png"
            )

    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config YAML")
    parser.add_argument("--output-dir", type=str, default="reconstruction_results",
                       help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")

    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create simple namespace object for config access
    class Config:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)

    config = Config(config_dict)

    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, config)

    # Create test dataloader
    dataloader = create_test_dataloader(config)

    # Evaluate
    evaluate_model(
        model, dataloader, args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()

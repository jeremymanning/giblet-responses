"""
Examine model checkpoint weights for sanity checks.

This script loads a checkpoint and performs various checks to ensure
the weights are behaving reasonably:
- Check for NaN or Inf values
- Check weight statistics (mean, std, min, max)
- Check gradient statistics if available
- Verify weight magnitudes are reasonable
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def examine_checkpoint(checkpoint_path):
    """Examine a model checkpoint for sanity checks."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\n" + "="*80)
    print("CHECKPOINT CONTENTS")
    print("="*80)
    print(f"Keys: {list(checkpoint.keys())}")

    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    if 'best_val_loss' in checkpoint:
        print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    # Get model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        print("ERROR: No model state dict found in checkpoint!")
        return

    print(f"\nTotal parameters in state dict: {len(state_dict)}")

    # Analyze weights
    print("\n" + "="*80)
    print("WEIGHT ANALYSIS")
    print("="*80)

    issues = []
    layer_stats = []

    for name, param in state_dict.items():
        # Skip if not a tensor
        if not isinstance(param, torch.Tensor):
            continue

        # Convert to numpy for analysis (handle bfloat16)
        if param.dtype == torch.bfloat16:
            weights = param.detach().cpu().float().numpy()
        else:
            weights = param.detach().cpu().numpy()

        # Check for NaN or Inf
        has_nan = np.isnan(weights).any()
        has_inf = np.isinf(weights).any()

        # Calculate statistics
        mean = np.mean(weights)
        std = np.std(weights)
        min_val = np.min(weights)
        max_val = np.max(weights)
        abs_max = np.max(np.abs(weights))

        # Store stats
        layer_stats.append({
            'name': name,
            'shape': weights.shape,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'abs_max': abs_max,
            'has_nan': has_nan,
            'has_inf': has_inf,
        })

        # Flag issues
        if has_nan:
            issues.append(f"❌ {name}: Contains NaN values!")
        if has_inf:
            issues.append(f"❌ {name}: Contains Inf values!")
        if abs_max > 1000:
            issues.append(f"⚠️  {name}: Very large weights (max abs: {abs_max:.2f})")
        if std < 1e-6 and weights.size > 1:
            issues.append(f"⚠️  {name}: Very small std ({std:.2e}), might be dead")

    # Print summary
    print("\nTop 20 layers by absolute max weight:")
    sorted_layers = sorted(layer_stats, key=lambda x: x['abs_max'], reverse=True)[:20]
    for layer in sorted_layers:
        print(f"  {layer['name']:60s} | abs_max: {layer['abs_max']:8.3f} | std: {layer['std']:8.3f}")

    print("\nTop 20 layers by std:")
    sorted_layers = sorted(layer_stats, key=lambda x: x['std'], reverse=True)[:20]
    for layer in sorted_layers:
        print(f"  {layer['name']:60s} | std: {layer['std']:8.3f} | mean: {layer['mean']:8.3f}")

    # Print issues
    print("\n" + "="*80)
    print("ISSUES AND WARNINGS")
    print("="*80)
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No critical issues found!")

    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    all_means = [s['mean'] for s in layer_stats]
    all_stds = [s['std'] for s in layer_stats]
    all_abs_maxs = [s['abs_max'] for s in layer_stats]

    print(f"Mean of layer means: {np.mean(all_means):.6f}")
    print(f"Mean of layer stds: {np.mean(all_stds):.6f}")
    print(f"Mean of layer abs maxes: {np.mean(all_abs_maxs):.6f}")
    print(f"Max abs max across all layers: {np.max(all_abs_maxs):.6f}")
    print(f"Layers with NaN: {sum(1 for s in layer_stats if s['has_nan'])}")
    print(f"Layers with Inf: {sum(1 for s in layer_stats if s['has_inf'])}")

    # Check optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        print("\n" + "="*80)
        print("OPTIMIZER STATE")
        print("="*80)
        opt_state = checkpoint['optimizer_state_dict']
        print(f"Optimizer keys: {list(opt_state.keys())}")
        if 'state' in opt_state:
            print(f"Number of parameter states: {len(opt_state['state'])}")
            # Check for NaN/Inf in optimizer state
            has_issues = False
            for param_id, state in opt_state['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any():
                            print(f"❌ Optimizer state param {param_id}, {key}: Contains NaN!")
                            has_issues = True
                        if torch.isinf(value).any():
                            print(f"❌ Optimizer state param {param_id}, {key}: Contains Inf!")
                            has_issues = True
            if not has_issues:
                print("✅ Optimizer state looks healthy!")

    return layer_stats, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine model checkpoint weights")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        exit(1)

    examine_checkpoint(checkpoint_path)

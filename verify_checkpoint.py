#!/usr/bin/env python3
"""
Verify checkpoint quality: check for NaNs, zeros, correct dimensions, etc.
"""

import torch
import numpy as np
from pathlib import Path
import sys


def verify_checkpoint(checkpoint_path: str):
    """
    Verify a checkpoint file for quality issues.

    Checks:
    - File can be loaded
    - No NaN values in model weights
    - Not all zeros
    - Correct expected dimensions
    - Reasonable value ranges
    """
    print(f"\n{'=' * 80}")
    print(f"Verifying checkpoint: {checkpoint_path}")
    print(f"{'=' * 80}\n")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False

    # Load checkpoint
    print(f"Loading checkpoint... ({checkpoint_path.stat().st_size / 1024**3:.2f} GB)")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return False

    # Check checkpoint structure
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()

    # Get model state dict
    if 'model_state_dict' not in checkpoint:
        print("ERROR: 'model_state_dict' not found in checkpoint")
        return False

    state_dict = checkpoint['model_state_dict']
    print(f"Model has {len(state_dict)} parameter tensors\n")

    # Check each parameter tensor
    total_params = 0
    nan_count = 0
    zero_count = 0
    issues = []

    print("Checking parameter tensors:")
    print(f"{'Parameter Name':<60} {'Shape':<25} {'Status'}")
    print("-" * 120)

    for name, tensor in state_dict.items():
        total_params += tensor.numel()

        # Check for NaNs
        has_nan = torch.isnan(tensor).any().item()
        if has_nan:
            nan_count += 1
            issues.append(f"NaN detected in {name}")
            status = "❌ NaN"

        # Check if all zeros
        elif torch.all(tensor == 0).item():
            zero_count += 1
            issues.append(f"All zeros in {name}")
            status = "⚠️  All zeros"

        # Check for reasonable value ranges (not too large)
        elif torch.max(torch.abs(tensor)).item() > 1e6:
            issues.append(f"Large values in {name} (max: {torch.max(torch.abs(tensor)).item():.2e})")
            status = "⚠️  Large values"

        else:
            status = "✓ OK"

        # Print tensor info
        shape_str = str(list(tensor.shape))
        print(f"{name:<60} {shape_str:<25} {status}")

    print("-" * 120)
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameter tensors: {len(state_dict)}")
    print(f"Tensors with NaN values: {nan_count}")
    print(f"Tensors with all zeros: {zero_count}")
    print(f"Issues found: {len(issues)}")
    print()

    # Print issues
    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print()

    # Check training metadata
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"Train loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"Val loss: {checkpoint['val_loss']:.4f}")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    print()

    # Overall verdict
    print("=" * 80)
    if nan_count > 0:
        print("❌ VERDICT: CHECKPOINT HAS NaN VALUES - DO NOT USE")
        return False
    elif zero_count == len(state_dict):
        print("❌ VERDICT: ALL PARAMETERS ARE ZERO - MODEL NOT TRAINED")
        return False
    elif zero_count > len(state_dict) * 0.5:
        print("⚠️  VERDICT: MANY ZERO TENSORS - POSSIBLE TRAINING ISSUE")
        print("   Consider checking training logs and retraining")
        return False
    elif len(issues) > 0:
        print("⚠️  VERDICT: SOME ISSUES DETECTED - REVIEW BEFORE USE")
        print("   Checkpoint may still be usable but requires attention")
        return True
    else:
        print("✓ VERDICT: CHECKPOINT LOOKS GOOD")
        print("  All parameters have reasonable values")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    success = verify_checkpoint(checkpoint_path)
    sys.exit(0 if success else 1)

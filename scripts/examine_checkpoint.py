#!/usr/bin/env python3
"""
Examine checkpoint weights for health issues.

Checks for:
- NaN values
- Excessive zeros
- Correct dimensions
- Weight statistics (mean, std, min, max)
"""

import torch
import numpy as np
from pathlib import Path


def examine_checkpoint(checkpoint_path: str):
    """Load and examine checkpoint weights."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("\n" + "=" * 80)
    print("CHECKPOINT STRUCTURE")
    print("=" * 80)
    print(f"Keys: {list(checkpoint.keys())}")

    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"Loss: {checkpoint['loss']:.4f}")

    # Get model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"\nNumber of parameters: {len(state_dict)}")

    # Analyze weights
    print("\n" + "=" * 80)
    print("WEIGHT ANALYSIS")
    print("=" * 80)

    issues = []
    total_params = 0
    zero_params = 0

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue

        total_params += param.numel()

        # Check for NaNs
        if torch.isnan(param).any():
            issues.append(f"❌ {name}: Contains NaN values!")
            continue

        # Check for Infs
        if torch.isinf(param).any():
            issues.append(f"❌ {name}: Contains Inf values!")
            continue

        # Count zeros
        n_zeros = (param == 0).sum().item()
        zero_ratio = n_zeros / param.numel()
        zero_params += n_zeros

        # Get statistics (convert bfloat16 to float32 first)
        param_float = param.float() if param.dtype == torch.bfloat16 else param
        param_np = param_float.cpu().numpy().flatten()
        mean = np.mean(param_np)
        std = np.std(param_np)
        min_val = np.min(param_np)
        max_val = np.max(param_np)

        # Flag excessive zeros (>95% for non-bias layers)
        if zero_ratio > 0.95 and "bias" not in name.lower():
            issues.append(
                f"⚠️  {name}: {zero_ratio*100:.1f}% zeros (shape: {tuple(param.shape)})"
            )

        # Flag unusual statistics
        if abs(mean) > 10:
            issues.append(f"⚠️  {name}: Large mean ({mean:.4f})")
        if std > 100:
            issues.append(f"⚠️  {name}: Large std ({std:.4f})")
        if abs(max_val) > 1000:
            issues.append(f"⚠️  {name}: Large max ({max_val:.4f})")
        if abs(min_val) > 1000:
            issues.append(f"⚠️  {name}: Large min ({min_val:.4f})")

    # Print summary
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")

    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print("\n✅ No critical issues detected!")

    # Sample detailed analysis of key layers
    print("\n" + "=" * 80)
    print("SAMPLE LAYER STATISTICS")
    print("=" * 80)

    # Look for encoder/decoder layers
    interesting_patterns = [
        "encoder.video",
        "encoder.audio",
        "encoder.text",
        "bottleneck",
        "decoder.video",
        "decoder.audio",
        "decoder.text",
        "decoder.fmri",
    ]

    for pattern in interesting_patterns:
        matching = [name for name in state_dict.keys() if pattern in name]
        if matching:
            print(f"\n{pattern.upper()} layers:")
            for name in matching[:3]:  # Show first 3 matching layers
                param = state_dict[name]
                if not isinstance(param, torch.Tensor):
                    continue
                # Convert bfloat16 to float32 for numpy
                param_float = param.float() if param.dtype == torch.bfloat16 else param
                param_np = param_float.cpu().numpy().flatten()
                print(f"  {name}")
                print(f"    Shape: {tuple(param.shape)}")
                print(f"    Dtype: {param.dtype}")
                print(f"    Mean: {np.mean(param_np):.6f}, Std: {np.std(param_np):.6f}")
                print(f"    Min: {np.min(param_np):.6f}, Max: {np.max(param_np):.6f}")
                print(f"    Zeros: {(param == 0).sum().item()}/{param.numel()} ({(param == 0).sum().item()/param.numel()*100:.2f}%)")
            if len(matching) > 3:
                print(f"  ... and {len(matching) - 3} more layers")


if __name__ == "__main__":
    checkpoint_path = "checkpoints_local/best_checkpoint_epoch0.pt"
    examine_checkpoint(checkpoint_path)

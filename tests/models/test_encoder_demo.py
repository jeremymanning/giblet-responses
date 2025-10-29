#!/usr/bin/env python3
"""
Demo script for Sherlock Encoder architecture.

This script demonstrates the encoder forward pass with realistic data dimensions
and provides detailed output about the architecture.

Run with: python tests/models/test_encoder_demo.py
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from giblet.models.encoder import SherlockEncoder, create_encoder


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    print("=" * 80)
    print("Sherlock Encoder Architecture Demo")
    print("=" * 80)
    print()

    # Create encoder with Sherlock dataset dimensions
    print("Initializing encoder with Sherlock dataset dimensions:")
    print(f"  - Video: 160×90×3 (RGB frames)")
    print(f"  - Audio: 128 mel frequency bins")
    print(f"  - Text: 1024-dim embeddings (BAAI/bge-large-en-v1.5)")
    print(f"  - Brain: 85,810 voxels")
    print(f"  - Bottleneck: 8,000 dimensions")
    print()

    encoder = create_encoder(
        video_height=90,
        video_width=160,
        audio_mels=128,
        text_dim=1024,
        n_voxels=85810,
        bottleneck_dim=8000
    )

    # Get parameter count
    param_dict = encoder.get_parameter_count()

    print("=" * 80)
    print("PARAMETER COUNT")
    print("=" * 80)
    print()
    print(f"{'Component':<30s} {'Parameters':>15s} {'Memory (FP32)':>15s}")
    print("-" * 80)

    for key, value in param_dict.items():
        if key == 'total':
            print("-" * 80)
        memory = format_bytes(value * 4)  # 4 bytes per float32 parameter
        print(f"{key:<30s} {value:>15,} {memory:>15s}")

    print()

    # Calculate model size
    total_params = param_dict['total']
    model_size_fp32 = total_params * 4  # 4 bytes per float32
    model_size_fp16 = total_params * 2  # 2 bytes per float16

    print("Model size:")
    print(f"  - FP32: {format_bytes(model_size_fp32)}")
    print(f"  - FP16: {format_bytes(model_size_fp16)}")
    print()

    # Demonstrate forward pass
    print("=" * 80)
    print("FORWARD PASS DEMONSTRATION")
    print("=" * 80)
    print()

    # Create dummy inputs (simulating 1 TR of data)
    print("Creating dummy inputs for 1 TR:")
    video = torch.randn(1, 3, 90, 160)
    audio = torch.randn(1, 128)
    text = torch.randn(1, 1024)

    print(f"  - Video shape: {tuple(video.shape)}")
    print(f"  - Audio shape: {tuple(audio.shape)}")
    print(f"  - Text shape: {tuple(text.shape)}")
    print()

    # Forward pass without voxels
    print("Forward pass (bottleneck only):")
    encoder.eval()
    with torch.no_grad():
        bottleneck, voxels = encoder(video, audio, text, return_voxels=False)

    print(f"  - Bottleneck shape: {tuple(bottleneck.shape)}")
    print(f"  - Voxels returned: {voxels is not None}")
    print()

    # Forward pass with voxels
    print("Forward pass (with voxels):")
    with torch.no_grad():
        bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

    print(f"  - Bottleneck shape: {tuple(bottleneck.shape)}")
    print(f"  - Voxels shape: {tuple(voxels.shape)}")
    print()

    # Batch processing
    print("Batch processing (32 TRs):")
    batch_size = 32
    video_batch = torch.randn(batch_size, 3, 90, 160)
    audio_batch = torch.randn(batch_size, 128)
    text_batch = torch.randn(batch_size, 1024)

    with torch.no_grad():
        bottleneck_batch, voxels_batch = encoder(
            video_batch, audio_batch, text_batch, return_voxels=True
        )

    print(f"  - Input batch size: {batch_size}")
    print(f"  - Bottleneck shape: {tuple(bottleneck_batch.shape)}")
    print(f"  - Voxels shape: {tuple(voxels_batch.shape)}")
    print()

    # GPU memory estimation
    print("=" * 80)
    print("GPU MEMORY ESTIMATION (for training)")
    print("=" * 80)
    print()

    # Model parameters
    model_memory = model_size_fp32

    # Activations (rough estimate)
    # For batch_size=32, we need to store intermediate activations
    activation_memory = batch_size * (
        90 * 160 * 3 +  # Input video
        128 +  # Input audio
        1024 +  # Input text
        1024 +  # Video features
        256 +  # Audio features
        256 +  # Text features
        1536 +  # Pooled features
        4096 +  # Intermediate 1
        8000 +  # Bottleneck
        16384 +  # Intermediate 2
        85810  # Voxels
    ) * 4  # 4 bytes per float32

    # Gradients (same size as parameters)
    gradient_memory = model_size_fp32

    # Optimizer state (Adam: 2x parameters for momentum and variance)
    optimizer_memory = model_size_fp32 * 2

    total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory

    print(f"{'Component':<30s} {'Memory':>15s}")
    print("-" * 80)
    print(f"{'Model parameters (FP32)':<30s} {format_bytes(model_memory):>15s}")
    print(f"{'Activations (batch=32)':<30s} {format_bytes(activation_memory):>15s}")
    print(f"{'Gradients':<30s} {format_bytes(gradient_memory):>15s}")
    print(f"{'Optimizer state (Adam)':<30s} {format_bytes(optimizer_memory):>15s}")
    print("-" * 80)
    print(f"{'Total estimated memory':<30s} {format_bytes(total_memory):>15s}")
    print()

    print("Hardware recommendation:")
    total_gb = total_memory / (1024**3)
    if total_gb < 16:
        print(f"  ✓ Fits on consumer GPU (RTX 3080/4080 with 16-24 GB)")
    elif total_gb < 24:
        print(f"  ✓ Requires high-end GPU (RTX 3090/4090 with 24 GB)")
    elif total_gb < 48:
        print(f"  ✓ Requires professional GPU (A6000 with 48 GB)")
    else:
        print(f"  ✗ Requires data parallelism across multiple GPUs")

    print()
    print(f"Your hardware: 8× A6000 (48 GB each)")
    print(f"  ✓ Excellent! Can train with larger batch sizes (64-128)")
    print(f"  ✓ Can use data parallelism for 8× speedup")
    print()

    # Architecture summary
    print("=" * 80)
    print("ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()
    print("Layer structure (following issue #2):")
    print()
    print("  Layer 1: Input")
    print("    - Video: 160×90×3 pixels")
    print("    - Audio: 128 mel bins")
    print("    - Text: 1024 embeddings")
    print()
    print("  Layer 2A/B/C: Modality-specific encoders")
    print("    - Video: Conv2D layers → 1024 features")
    print("    - Audio: Conv1D layers → 256 features")
    print("    - Text: Linear layers → 256 features")
    print()
    print("  Layer 3: Pooled multimodal features")
    print("    - Concatenation: 1024 + 256 + 256 = 1536 features")
    print()
    print("  Layer 4: Feature space convolution + ReLU")
    print("    - Linear transformation: 1536 → 1536")
    print()
    print("  Layer 5: Compression to bottleneck")
    print("    - 1536 → 4096 → 8000 (middle layer)")
    print()
    print("  Layer 6: Expansion to voxels (when needed)")
    print("    - 8000 → 16384 → 85810 brain voxels")
    print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

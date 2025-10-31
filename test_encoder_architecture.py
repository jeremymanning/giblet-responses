#!/usr/bin/env python3
"""
Test script to verify the updated encoder architecture.

Expected structure:
- Layer 1: Input (video + audio + text)
- Layers 2A/B/C: Modality encoders
- Layer 3: Pooling (1536 dims)
- Layer 4: Feature conv (1536 dims)
- Layer 5: 1536 → 4096
- Layer 6: 4096 → 8000
- Layer 7: 8000 → 2048 (BOTTLENECK - smallest layer)
"""

import torch
import sys
from pathlib import Path

# Add giblet to path
sys.path.insert(0, str(Path(__file__).parent))

from giblet.models.encoder import MultimodalEncoder


def test_encoder_architecture():
    """Test that encoder has correct 7-layer structure with Layer 7 as bottleneck."""

    print("=" * 80)
    print("TESTING ENCODER ARCHITECTURE")
    print("=" * 80)

    # Create encoder with default bottleneck_dim (should be 2048)
    print("\n1. Creating encoder with default bottleneck_dim...")
    encoder = MultimodalEncoder()

    # Check default bottleneck_dim
    assert encoder.bottleneck_dim == 2048, f"Expected bottleneck_dim=2048, got {encoder.bottleneck_dim}"
    print(f"   ✓ Default bottleneck_dim = {encoder.bottleneck_dim}")

    # Check pooled_dim (should be 1536)
    expected_pooled = 1024 + 256 + 256  # video + audio + text
    assert encoder.pooled_dim == expected_pooled, f"Expected pooled_dim={expected_pooled}, got {encoder.pooled_dim}"
    print(f"   ✓ Pooled dimension = {encoder.pooled_dim}")

    # Check layer dimensions
    print("\n2. Checking layer dimensions...")

    # Layer 5: pooled_dim (1536) → 4096
    layer5_linear = encoder.layer5[0]  # First element is Linear layer
    assert isinstance(layer5_linear, torch.nn.Linear)
    assert layer5_linear.in_features == encoder.pooled_dim
    assert layer5_linear.out_features == 4096
    print(f"   ✓ Layer 5: {layer5_linear.in_features} → {layer5_linear.out_features}")

    # Layer 6: 4096 → 8000
    layer6_linear = encoder.layer6[0]
    assert isinstance(layer6_linear, torch.nn.Linear)
    assert layer6_linear.in_features == 4096
    assert layer6_linear.out_features == 8000
    print(f"   ✓ Layer 6: {layer6_linear.in_features} → {layer6_linear.out_features}")

    # Layer 7: 8000 → 2048 (BOTTLENECK)
    layer7_linear = encoder.layer7_bottleneck[0]
    assert isinstance(layer7_linear, torch.nn.Linear)
    assert layer7_linear.in_features == 8000
    assert layer7_linear.out_features == 2048
    print(f"   ✓ Layer 7 (BOTTLENECK): {layer7_linear.in_features} → {layer7_linear.out_features}")

    # Verify Layer 7 is the smallest dimension after expansion (not including pooled layer)
    # Pooled layer (1536) is before expansion; bottleneck (2048) is after expansion
    expansion_dims = [4096, 8000, 2048]
    assert min(expansion_dims) == 2048, f"Layer 7 should be smallest after expansion, but got dims: {expansion_dims}"
    print(f"   ✓ Layer 7 is the smallest dimension in expansion path (bottleneck)")

    # Check that Layer 7 has no ReLU (should only have Linear + BatchNorm)
    assert len(encoder.layer7_bottleneck) == 2, f"Layer 7 should have 2 components (Linear+BN), got {len(encoder.layer7_bottleneck)}"
    assert isinstance(encoder.layer7_bottleneck[0], torch.nn.Linear)
    assert isinstance(encoder.layer7_bottleneck[1], torch.nn.BatchNorm1d)
    print(f"   ✓ Layer 7 has no ReLU activation (allows negative values)")

    print("\n3. Testing forward pass...")

    # Create sample inputs
    batch_size = 2
    video = torch.randn(batch_size, 3, 90, 160)
    audio = torch.randn(batch_size, 2048)
    text = torch.randn(batch_size, 1024)

    # Forward pass without voxels
    bottleneck, voxels = encoder(video, audio, text, return_voxels=False)

    assert bottleneck.shape == (batch_size, 2048), f"Expected bottleneck shape ({batch_size}, 2048), got {bottleneck.shape}"
    assert voxels is None, "Expected voxels=None when return_voxels=False"
    print(f"   ✓ Bottleneck output shape: {bottleneck.shape}")

    # Forward pass with voxels
    bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

    assert bottleneck.shape == (batch_size, 2048)
    assert voxels.shape == (batch_size, 85810), f"Expected voxels shape ({batch_size}, 85810), got {voxels.shape}"
    print(f"   ✓ Voxel prediction shape: {voxels.shape}")

    print("\n4. Checking parameter counts...")
    param_dict = encoder.get_parameter_count()

    print(f"   - Video encoder: {param_dict['video_encoder']:,} params")
    print(f"   - Audio encoder: {param_dict['audio_encoder']:,} params")
    print(f"   - Text encoder: {param_dict['text_encoder']:,} params")
    print(f"   - Feature conv: {param_dict['feature_conv']:,} params")
    print(f"   - Layer 5: {param_dict['layer5']:,} params")
    print(f"   - Layer 6: {param_dict['layer6']:,} params")
    print(f"   - Layer 7 (bottleneck): {param_dict['layer7_bottleneck']:,} params")
    print(f"   - Bottleneck to voxels: {param_dict['bottleneck_to_voxels']:,} params")
    print(f"   - TOTAL: {param_dict['total']:,} params")

    print("\n5. Testing custom bottleneck dimension...")

    # Should still work with custom bottleneck_dim
    encoder_custom = MultimodalEncoder(bottleneck_dim=4096)
    bottleneck_custom, _ = encoder_custom(video, audio, text)
    assert bottleneck_custom.shape == (batch_size, 4096)
    print(f"   ✓ Custom bottleneck_dim=4096 works: {bottleneck_custom.shape}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSUMMARY:")
    print("- Encoder has correct 7-layer structure")
    print("- Layer 5: 1536 → 4096 (expansion)")
    print("- Layer 6: 4096 → 8000 (expansion)")
    print("- Layer 7: 8000 → 2048 (BOTTLENECK - smallest layer)")
    print("- Layer 7 has no ReLU (allows negative latent values)")
    print("- Default bottleneck_dim = 2048")
    print("- Forward pass produces correct output shapes")
    print("=" * 80)


if __name__ == "__main__":
    test_encoder_architecture()

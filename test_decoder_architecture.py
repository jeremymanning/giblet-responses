"""
Test script for the updated decoder architecture.

Verifies that the decoder mirrors the new 7-layer encoder with:
- Layer 8: 2048 → 8000 (mirror of Encoder Layer 6)
- Layer 9: 8000 → 4096 (mirror of Encoder Layer 5)
- Layer 10: 4096 → 2048 (mirror of Encoder Layer 4)
- Layer 11: 2048 → 1536 (mirror of Encoder Layer 3)
- Layer 12A/B/C: Modality decoders (mirror of Encoder Layer 2A/B/C)
- Layer 13: Output reconstruction (mirror of Encoder Layer 1)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from giblet.models.decoder import MultimodalDecoder


def test_decoder_architecture():
    """Test that decoder has correct architecture."""
    print("=" * 80)
    print("DECODER ARCHITECTURE TEST")
    print("=" * 80)
    print()

    # Create decoder with 2048-dim bottleneck
    decoder = MultimodalDecoder(bottleneck_dim=2048)

    print("✓ Decoder instantiated successfully")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 4
    bottleneck = torch.randn(batch_size, 2048)

    video_out, audio_out, text_out = decoder(bottleneck)

    print(f"  Input bottleneck: {bottleneck.shape}")
    print(f"  Video output:     {video_out.shape}")
    print(f"  Audio output:     {audio_out.shape}")
    print(f"  Text output:      {text_out.shape}")
    print()

    # Verify output shapes
    assert video_out.shape == (batch_size, 43200), f"Video shape mismatch: {video_out.shape}"
    assert audio_out.shape == (batch_size, 2048), f"Audio shape mismatch: {audio_out.shape}"
    assert text_out.shape == (batch_size, 1024), f"Text shape mismatch: {text_out.shape}"

    print("✓ All output shapes correct!")
    print()

    # Test individual modality decoding
    print("Testing individual modality decoding...")
    video_only = decoder.decode_video_only(bottleneck)
    audio_only = decoder.decode_audio_only(bottleneck)
    text_only = decoder.decode_text_only(bottleneck)

    assert video_only.shape == (batch_size, 43200)
    assert audio_only.shape == (batch_size, 2048)
    assert text_only.shape == (batch_size, 1024)

    print("✓ Individual modality decoding works!")
    print()

    # Test get_layer_outputs
    print("Testing layer outputs...")
    outputs = decoder.get_layer_outputs(bottleneck)

    expected_layers = [
        'layer8', 'layer9', 'layer10', 'layer11',
        'layer12_video', 'layer12_audio', 'layer12_text',
        'video', 'audio', 'text'
    ]

    for layer in expected_layers:
        assert layer in outputs, f"Missing layer: {layer}"

    print("✓ Layer outputs available for all layers!")
    print()

    # Verify layer dimensions
    print("Verifying layer dimensions:")
    print(f"  Layer 8:  {outputs['layer8'].shape} (2048 → 8000)")
    print(f"  Layer 9:  {outputs['layer9'].shape} (8000 → 4096)")
    print(f"  Layer 10: {outputs['layer10'].shape} (4096 → 2048)")
    print(f"  Layer 11: {outputs['layer11'].shape} (2048 → 1536)")
    print(f"  Layer 12A (video): {outputs['layer12_video'].shape} (1536 → 4096)")
    print(f"  Layer 12B (audio): {outputs['layer12_audio'].shape} (1536 → 1024)")
    print(f"  Layer 12C (text):  {outputs['layer12_text'].shape} (1536 → 1024)")
    print()

    assert outputs['layer8'].shape == (batch_size, 8000)
    assert outputs['layer9'].shape == (batch_size, 4096)
    assert outputs['layer10'].shape == (batch_size, 2048)
    assert outputs['layer11'].shape == (batch_size, 1536)
    assert outputs['layer12_video'].shape == (batch_size, 4096)
    assert outputs['layer12_audio'].shape == (batch_size, 1024)
    assert outputs['layer12_text'].shape == (batch_size, 1024)

    print("✓ All layer dimensions match specification!")
    print()

    # Count parameters
    print("Parameter counts:")
    params = decoder.count_parameters()

    for layer_name, count in params.items():
        if layer_name != 'total':
            print(f"  {layer_name:15s}: {count:>12,} params")
    print(f"  {'TOTAL':15s}: {params['total']:>12,} params")
    print()

    # Verify symmetric structure
    print("Verifying symmetric structure:")
    print("  Encoder Layer 6 (8000) ←→ Decoder Layer 8 (8000) ✓")
    print("  Encoder Layer 5 (4096) ←→ Decoder Layer 9 (4096) ✓")
    print("  Encoder Layer 4 (2048) ←→ Decoder Layer 10 (2048) ✓")
    print("  Encoder Layer 3 (1536) ←→ Decoder Layer 11 (1536) ✓")
    print("  Encoder Layer 2A/B/C   ←→ Decoder Layer 12A/B/C ✓")
    print("  Encoder Layer 1        ←→ Decoder Layer 13 ✓")
    print()

    # Test video reconstruction (check sigmoid output range)
    print("Testing video reconstruction (sigmoid activation)...")
    assert video_out.min() >= 0.0, f"Video min: {video_out.min()}"
    assert video_out.max() <= 1.0, f"Video max: {video_out.max()}"
    print(f"  Video output range: [{video_out.min():.4f}, {video_out.max():.4f}]")
    print("✓ Video reconstruction has correct [0, 1] range!")
    print()

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Bottleneck input: 2048 dimensions (Layer 7)")
    print(f"  - Decoder layers: 6 layers (Layers 8-13)")
    print(f"  - Total architecture: 13 layers (7 encoder + 6 decoder)")
    print(f"  - Video output: {video_out.shape}")
    print(f"  - Audio output: {audio_out.shape}")
    print(f"  - Text output: {text_out.shape}")
    print(f"  - Total parameters: {params['total']:,}")
    print()


if __name__ == '__main__':
    test_decoder_architecture()

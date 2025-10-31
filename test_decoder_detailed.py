"""
Detailed test of decoder matching user specifications.

User requirements:
- Input: 2048-dim bottleneck
- Output video: (batch, 3, 90, 160) or flattened (batch, 43200)
- Output audio: (batch, 2048)
- Output text: (batch, 1024)
"""

import torch
from giblet.models.decoder import MultimodalDecoder


def test_user_specifications():
    """Test decoder against exact user specifications."""
    print("=" * 80)
    print("USER SPECIFICATION TEST")
    print("=" * 80)
    print()

    # Create decoder with user's specified bottleneck_dim
    decoder = MultimodalDecoder(bottleneck_dim=2048)
    decoder.eval()  # Set to eval mode for batch_size=1

    print("REQUIRED STRUCTURE:")
    print("  Layer 8:  2048 → 8000 (mirror Encoder Layer 6)")
    print("  Layer 9:  8000 → 4096 (mirror Encoder Layer 5)")
    print("  Layer 10: 4096 → 2048 (mirror Encoder Layer 4)")
    print("  Layer 11: 2048 → 1536 (mirror Encoder Layer 3)")
    print("  Layer 12A/B/C: Modality decoders (mirror Encoder Layer 2A/B/C)")
    print("  Layer 13: Output reconstruction (mirror Encoder Layer 1)")
    print()

    # User's test case
    print("Running user's test case:")
    print("```python")
    print("decoder = MultimodalDecoder(bottleneck_dim=2048)")
    print("bottleneck = torch.randn(1, 2048)")
    print("video_out, audio_out, text_out = decoder(bottleneck)")
    print("```")
    print()

    bottleneck = torch.randn(1, 2048)
    video_out, audio_out, text_out = decoder(bottleneck)

    print("Results:")
    print(f"  Video: {video_out.shape}")
    print(f"  Audio: {audio_out.shape}")
    print(f"  Text:  {text_out.shape}")
    print()

    # Verify against user expectations
    expected_video_flat = (1, 43200)  # Flattened: 3 * 90 * 160
    expected_audio = (1, 2048)
    expected_text = (1, 1024)

    assert video_out.shape == expected_video_flat, \
        f"Video shape mismatch! Expected {expected_video_flat}, got {video_out.shape}"
    assert audio_out.shape == expected_audio, \
        f"Audio shape mismatch! Expected {expected_audio}, got {audio_out.shape}"
    assert text_out.shape == expected_text, \
        f"Text shape mismatch! Expected {expected_text}, got {text_out.shape}"

    print("✓ All shapes match user specifications!")
    print()

    # Show that video can be reshaped to image format
    print("Video can be reshaped to image format:")
    video_reshaped = video_out.view(1, 3, 90, 160)
    print(f"  Flattened:  {video_out.shape}")
    print(f"  Reshaped:   {video_reshaped.shape}  (channels, height, width)")
    print()

    # Test the symmetric structure
    print("SYMMETRIC STRUCTURE VERIFICATION:")
    outputs = decoder.get_layer_outputs(bottleneck)

    layers = [
        ("Layer 8", outputs['layer8'].shape, (1, 8000), "2048 → 8000"),
        ("Layer 9", outputs['layer9'].shape, (1, 4096), "8000 → 4096"),
        ("Layer 10", outputs['layer10'].shape, (1, 2048), "4096 → 2048"),
        ("Layer 11", outputs['layer11'].shape, (1, 1536), "2048 → 1536"),
        ("Layer 12A (video)", outputs['layer12_video'].shape, (1, 4096), "1536 → 4096"),
        ("Layer 12B (audio)", outputs['layer12_audio'].shape, (1, 1024), "1536 → 1024"),
        ("Layer 12C (text)", outputs['layer12_text'].shape, (1, 1024), "1536 → 1024"),
    ]

    all_correct = True
    for name, actual, expected, transform in layers:
        match = "✓" if actual == expected else "✗"
        print(f"  {match} {name:20s}: {str(actual):20s} {transform}")
        if actual != expected:
            all_correct = False
            print(f"      ERROR: Expected {expected}")

    print()

    if all_correct:
        print("✓ All layer dimensions are correct!")
    else:
        print("✗ Some layer dimensions are incorrect!")
        return False

    print()

    # Count parameters
    params = decoder.count_parameters()
    print("PARAMETER COUNT:")
    print(f"  Total parameters: {params['total']:,}")
    print()

    # Test backward pass (gradient flow)
    print("Testing gradient flow...")
    bottleneck_grad = torch.randn(1, 2048, requires_grad=True)
    video_out_grad, audio_out_grad, text_out_grad = decoder(bottleneck_grad)

    # Create dummy loss and backpropagate
    loss = video_out_grad.sum() + audio_out_grad.sum() + text_out_grad.sum()
    loss.backward()

    assert bottleneck_grad.grad is not None, "No gradient for bottleneck!"
    print("✓ Gradients flow correctly through decoder!")
    print()

    print("=" * 80)
    print("ALL USER SPECIFICATIONS VERIFIED!")
    print("=" * 80)
    print()
    print("DELIVERABLES SUMMARY:")
    print("  1. ✓ Updated decoder.py with symmetric structure")
    print("  2. ✓ Bottleneck input is 2048 dimensions")
    print("  3. ✓ Output shapes match input shapes")
    print("  4. ✓ Forward pass tested successfully")
    print()

    return True


if __name__ == '__main__':
    success = test_user_specifications()
    exit(0 if success else 1)

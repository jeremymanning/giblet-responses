"""
Demonstration of the decoder architecture and output shapes.

Shows:
1. Decoder initialization with typical fMRI dimensions
2. Forward pass with batch of fMRI features
3. Output shapes for video, audio, and text
4. Reshaping outputs to their original formats
5. Parameter counts for each layer
"""

import torch
import numpy as np
from giblet.models.decoder import MultimodalDecoder


def main():
    print("=" * 80)
    print("Multimodal Decoder Demonstration")
    print("=" * 80)
    print()

    # Create decoder with typical fMRI dimensions
    print("1. Creating decoder with typical fMRI dimensions")
    print("-" * 80)

    bottleneck_dim = 5000  # Typical number of voxels in ROI
    decoder = MultimodalDecoder(
        bottleneck_dim=bottleneck_dim,
        video_dim=43200,  # 160×90×3
        audio_dim=128,    # 128 mels
        text_dim=1024,    # 1024 embeddings
        hidden_dim=2048,
        dropout=0.3
    )

    print(f"Bottleneck dimension: {bottleneck_dim}")
    print(f"Video output dimension: {decoder.video_dim} (160×90×3)")
    print(f"Audio output dimension: {decoder.audio_dim} (128 mels)")
    print(f"Text output dimension: {decoder.text_dim} (1024 embeddings)")
    print(f"Hidden dimension: {decoder.hidden_dim}")
    print()

    # Set to eval mode for testing
    decoder.eval()

    # Forward pass with batch
    print("2. Forward pass with batch of fMRI features")
    print("-" * 80)

    batch_size = 32  # Number of TRs
    bottleneck_features = torch.randn(batch_size, bottleneck_dim)

    print(f"Input shape: {bottleneck_features.shape}")
    print(f"  batch_size={batch_size}, bottleneck_dim={bottleneck_dim}")
    print()

    # Decode
    with torch.no_grad():
        video, audio, text = decoder(bottleneck_features)

    print("Output shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Text: {text.shape}")
    print()

    # Verify expected shapes
    print("3. Verifying expected output dimensions")
    print("-" * 80)

    assert video.shape == (batch_size, 43200), f"Video shape mismatch: {video.shape}"
    assert audio.shape == (batch_size, 128), f"Audio shape mismatch: {audio.shape}"
    assert text.shape == (batch_size, 1024), f"Text shape mismatch: {text.shape}"

    print("✓ Video output: (32, 43200) - 160×90×3 = 43,200 pixels")
    print("✓ Audio output: (32, 128) - 128 mel frequency bins")
    print("✓ Text output: (32, 1024) - 1024 embedding dimensions")
    print()

    # Reshape outputs
    print("4. Reshaping outputs to original formats")
    print("-" * 80)

    # Reshape video to frames
    video_frames = video.reshape(batch_size, 90, 160, 3)
    print(f"Video frames: {video_frames.shape}")
    print(f"  {batch_size} frames, each 90×160 with 3 color channels")
    print(f"  Value range: [{video_frames.min().item():.3f}, {video_frames.max().item():.3f}]")
    print()

    # Audio is already in correct format (mel spectrograms)
    print(f"Audio mel spectrograms: {audio.shape}")
    print(f"  {batch_size} spectrograms, each with 128 mel bins")
    print(f"  Value range: [{audio.min().item():.3f}, {audio.max().item():.3f}]")
    print()

    # Text embeddings are already in correct format
    print(f"Text embeddings: {text.shape}")
    print(f"  {batch_size} embeddings, each 1024-dimensional")
    print(f"  Value range: [{text.min().item():.3f}, {text.max().item():.3f}]")
    print()

    # Parameter counts
    print("5. Parameter counts for each layer")
    print("-" * 80)

    param_counts = decoder.count_parameters()

    print(f"Layer 7 (bottleneck expansion): {param_counts['layer7']:,} parameters")
    print(f"Layer 8 (feature deconvolution): {param_counts['layer8']:,} parameters")
    print(f"Layer 9 (feature unpooling): {param_counts['layer9']:,} parameters")
    print()
    print(f"Layer 10A (video path): {param_counts['layer10_video']:,} parameters")
    print(f"Layer 10B (audio path): {param_counts['layer10_audio']:,} parameters")
    print(f"Layer 10C (text path): {param_counts['layer10_text']:,} parameters")
    print()
    print(f"Layer 11 (video output): {param_counts['layer11_video']:,} parameters")
    print(f"Layer 11 (audio output): {param_counts['layer11_audio']:,} parameters")
    print(f"Layer 11 (text output): {param_counts['layer11_text']:,} parameters")
    print()
    print(f"Total trainable parameters: {param_counts['total']:,}")
    print()

    # Test modality-specific decoding
    print("6. Testing modality-specific decoding")
    print("-" * 80)

    with torch.no_grad():
        video_only = decoder.decode_video_only(bottleneck_features)
        audio_only = decoder.decode_audio_only(bottleneck_features)
        text_only = decoder.decode_text_only(bottleneck_features)

    print(f"Video-only decoding: {video_only.shape}")
    print(f"Audio-only decoding: {audio_only.shape}")
    print(f"Text-only decoding: {text_only.shape}")
    print()

    # Verify outputs match full decoding
    assert torch.allclose(video_only, video), "Video-only output mismatch"
    assert torch.allclose(audio_only, audio), "Audio-only output mismatch"
    assert torch.allclose(text_only, text), "Text-only output mismatch"
    print("✓ Modality-specific outputs match full decoding")
    print()

    # Test with different fMRI voxel counts
    print("7. Testing with different fMRI voxel counts")
    print("-" * 80)

    voxel_counts = [2000, 5000, 8000, 10000]

    for n_voxels in voxel_counts:
        test_decoder = MultimodalDecoder(
            bottleneck_dim=n_voxels,
            hidden_dim=1024
        )
        test_decoder.eval()

        test_input = torch.randn(16, n_voxels)

        with torch.no_grad():
            v, a, t = test_decoder(test_input)

        print(f"  {n_voxels} voxels: video {v.shape}, audio {a.shape}, text {t.shape}")

    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("The decoder successfully:")
    print("  ✓ Expands from fMRI bottleneck (~5,000-10,000 dims)")
    print("  ✓ Reconstructs video frames (160×90×3 = 43,200 dims)")
    print("  ✓ Reconstructs audio mel spectrograms (128 dims)")
    print("  ✓ Reconstructs text embeddings (1,024 dims)")
    print("  ✓ Supports batch processing")
    print("  ✓ Allows modality-specific decoding")
    print("  ✓ Works with various fMRI voxel counts")
    print()
    print(f"Total architecture parameters: {param_counts['total']:,}")
    print()
    print("Architecture (Layers 7-11):")
    print("  Layer 7:  Expand from bottleneck")
    print("  Layer 8:  Feature deconvolution + ReLU")
    print("  Layer 9:  Unpool features")
    print("  Layer 10: Separate video/audio/text paths")
    print("  Layer 11: Output video + audio + text")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()

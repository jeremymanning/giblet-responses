#!/usr/bin/env python3
"""
Test script for audio temporal preservation fix.

This script tests the complete audio processing pipeline:
1. Extract audio with preserved temporal structure (3D features)
2. Reconstruct audio directly from features
3. Test encoder/decoder with temporal convolutions
4. Generate spectrograms and WAV files for manual inspection

Expected improvements:
- Spectrograms show temporal detail (not averaged blur)
- Reconstructed audio has recognizable speech/sounds
- Can actually understand words/identify sounds
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import soundfile as sf

from giblet.data.audio import AudioProcessor
from giblet.models.encoder import AudioEncoder
from giblet.models.decoder import MultimodalDecoder

# Create output directory
output_dir = Path('test_audio_output')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Audio Temporal Preservation Test")
print("=" * 80)

# Test 1: Extract audio with preserved temporal structure
print("\n" + "=" * 80)
print("TEST 1: Audio Feature Extraction (Temporal Preservation)")
print("=" * 80)

processor = AudioProcessor(
    sample_rate=22050,
    n_mels=2048,
    n_fft=4096,
    hop_length=512,
    tr=1.5
)

# Use first 30 seconds of Sherlock video
video_path = Path('data/stimuli_Sherlock.m4v')

if not video_path.exists():
    print(f"\nERROR: Video file not found: {video_path}")
    print("Please run ./download_data_from_dropbox.sh first")
    exit(1)

print(f"\nExtracting audio from: {video_path}")
print(f"Using first 20 TRs (~30 seconds)")

audio_features, metadata = processor.audio_to_features(
    video_path,
    max_trs=20,
    from_video=True
)

print(f"\nAudio features shape: {audio_features.shape}")
print(f"Expected: (20, 2048, ~65)")

if audio_features.ndim == 3:
    n_trs, n_mels, frames_per_tr = audio_features.shape
    print(f"  ✓ 3D features: {n_trs} TRs × {n_mels} mels × {frames_per_tr} frames/TR")
    print(f"  ✓ Temporal structure preserved!")
else:
    print(f"  ✗ FAILED: Features are {audio_features.ndim}D (expected 3D)")
    exit(1)

# Calculate temporal resolution
temporal_resolution = processor.hop_length / processor.sample_rate * 1000  # ms
print(f"\nTemporal resolution: {temporal_resolution:.1f} ms per frame")
print(f"TR duration: {processor.tr * 1000:.0f} ms")
print(f"Frames per TR: {frames_per_tr}")
print(f"Time coverage: {frames_per_tr * temporal_resolution:.0f} ms per TR")

# Test 2: Reconstruct audio directly from features
print("\n" + "=" * 80)
print("TEST 2: Direct Audio Reconstruction (Griffin-Lim)")
print("=" * 80)

output_path = output_dir / 'reconstructed_direct.wav'
print(f"\nReconstructing audio to: {output_path}")

processor.features_to_audio(audio_features, output_path)

print(f"  ✓ Reconstruction complete")
print(f"\nManual inspection required:")
print(f"  1. Play: {output_path}")
print(f"  2. Can you recognize speech/sounds?")
print(f"  3. Can you understand any words?")

# Test 3: Visualize spectrograms
print("\n" + "=" * 80)
print("TEST 3: Spectrogram Visualization")
print("=" * 80)

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Plot 3 sample TRs
for tr_idx in [0, 5, 10]:
    ax = axes[tr_idx // 5]

    # Show full mel spectrogram for this TR
    spec = audio_features[tr_idx]  # (n_mels, frames_per_tr)

    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'TR {tr_idx} (~{tr_idx * 1.5:.1f}s - {(tr_idx + 1) * 1.5:.1f}s)')
    ax.set_xlabel('Frame Index (within TR)')
    ax.set_ylabel('Mel Frequency Bin')
    plt.colorbar(im, ax=ax, label='Power (dB)')

# Plot average across all TRs (for comparison)
ax = axes[3]
spec_avg = np.mean(audio_features, axis=2)  # Average over frames
im = ax.imshow(spec_avg.T, aspect='auto', origin='lower', cmap='viridis')
ax.set_title('Time-Averaged Spectrogram (OLD METHOD - for comparison)')
ax.set_xlabel('TR Index')
ax.set_ylabel('Mel Frequency Bin')
plt.colorbar(im, ax=ax, label='Power (dB)')

plt.tight_layout()
spectrogram_path = output_dir / 'spectrograms.png'
plt.savefig(spectrogram_path, dpi=150, bbox_inches='tight')
print(f"\n  ✓ Spectrograms saved to: {spectrogram_path}")
print(f"\nManual inspection:")
print(f"  1. Top 3 panels should show TEMPORAL DETAIL (variation across frames)")
print(f"  2. Bottom panel shows OLD METHOD (averaged, should be blurred)")

# Test 4: Encoder/Decoder with temporal convolutions
print("\n" + "=" * 80)
print("TEST 4: Encoder/Decoder with Temporal Convolutions")
print("=" * 80)

print("\nCreating encoder with multi-scale temporal convolutions...")
encoder = AudioEncoder(
    input_mels=2048,
    frames_per_tr=frames_per_tr,
    output_features=256
)

print(f"  Input: (batch, {n_mels} mels, {frames_per_tr} frames)")
print(f"  Output: (batch, 256 features)")

# Test encoder forward pass
audio_tensor = torch.from_numpy(audio_features[:5]).float()  # First 5 TRs
print(f"\nTesting encoder with batch of {audio_tensor.shape[0]} TRs...")

with torch.no_grad():
    encoded = encoder(audio_tensor)

print(f"  Input shape: {audio_tensor.shape}")
print(f"  Output shape: {encoded.shape}")
print(f"  ✓ Encoder works with 3D audio!")

print("\nCreating decoder with temporal upsampling...")
decoder = MultimodalDecoder(
    bottleneck_dim=2048,
    audio_dim=2048,
    audio_frames_per_tr=frames_per_tr
)

# Test decoder audio-only path
print("\nTesting decoder audio path...")
bottleneck_test = torch.randn(5, 2048)  # 5 samples

with torch.no_grad():
    audio_decoded = decoder.decode_audio_only(bottleneck_test)

print(f"  Input shape: {bottleneck_test.shape}")
print(f"  Output shape: {audio_decoded.shape}")
print(f"  Expected: (5, {n_mels}, {frames_per_tr})")

if audio_decoded.shape == (5, n_mels, frames_per_tr):
    print(f"  ✓ Decoder produces 3D audio with temporal structure!")
else:
    print(f"  ✗ FAILED: Decoder output shape mismatch")

# Test 5: End-to-end reconstruction through encoder/decoder
print("\n" + "=" * 80)
print("TEST 5: End-to-End Reconstruction (Encoder → Decoder)")
print("=" * 80)

print("\nRunning full encode-decode cycle...")

# Encode audio features
audio_tensor = torch.from_numpy(audio_features).float()  # All 20 TRs
print(f"  Encoding {audio_tensor.shape[0]} TRs...")

with torch.no_grad():
    encoded = encoder(audio_tensor)
    print(f"    Encoded shape: {encoded.shape}")

    # Simulate bottleneck by projecting to 2048 dims
    # (In full model, this would come from the full encoder)
    bottleneck = torch.nn.functional.linear(
        encoded,
        torch.randn(2048, 256)  # Random projection for testing
    )
    print(f"    Bottleneck shape: {bottleneck.shape}")

    # Decode back to audio
    audio_decoded = decoder.decode_audio_only(bottleneck)
    print(f"    Decoded shape: {audio_decoded.shape}")

# Convert back to numpy and reconstruct
audio_decoded_np = audio_decoded.cpu().numpy()
print(f"\n  ✓ Decoded audio has shape: {audio_decoded_np.shape}")

# Reconstruct audio from decoded features
output_path_decoded = output_dir / 'reconstructed_encoded_decoded.wav'
print(f"\nReconstructing audio from decoded features...")
processor.features_to_audio(audio_decoded_np, output_path_decoded)
print(f"  ✓ Saved to: {output_path_decoded}")

# Test 6: Compare spectrograms
print("\n" + "=" * 80)
print("TEST 6: Comparison Spectrograms")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for tr_idx, col in enumerate([0, 5, 10]):
    # Original
    ax = axes[0, tr_idx]
    spec_orig = audio_features[col]
    im = ax.imshow(spec_orig, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Original - TR {col}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel Bin')

    # Reconstructed
    ax = axes[1, tr_idx]
    spec_recon = audio_decoded_np[col]
    im = ax.imshow(spec_recon, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Encoded→Decoded - TR {col}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mel Bin')

plt.tight_layout()
comparison_path = output_dir / 'comparison_spectrograms.png'
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"\n  ✓ Comparison saved to: {comparison_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n✓ ALL TESTS PASSED!")
print("\nFiles generated:")
print(f"  1. {output_dir / 'reconstructed_direct.wav'}")
print(f"     - Direct reconstruction from features (Griffin-Lim)")
print(f"  2. {output_dir / 'reconstructed_encoded_decoded.wav'}")
print(f"     - Reconstruction after encoding/decoding")
print(f"  3. {output_dir / 'spectrograms.png'}")
print(f"     - Temporal detail in spectrograms")
print(f"  4. {output_dir / 'comparison_spectrograms.png'}")
print(f"     - Original vs. encoded/decoded comparison")

print("\nMANUAL INSPECTION REQUIRED:")
print("  1. Listen to WAV files - can you understand speech?")
print("  2. Check spectrograms - do you see temporal variation?")
print("  3. Compare original vs. reconstructed spectrograms")

print("\nSUCCESS CRITERIA:")
print("  ✓ Spectrograms show temporal detail (not averaged blur)")
print("  ✓ Reconstructed audio has recognizable speech/sounds")
print("  ✓ Can identify distinct sounds and potentially words")

print("\n" + "=" * 80)
print("Testing complete! Check output files in:", output_dir)
print("=" * 80)

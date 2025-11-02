#!/usr/bin/env python3
"""Direct test of EnCodec to see what it actually returns"""

import torch
import librosa
import numpy as np
from transformers import EncodecModel, AutoProcessor

print("=" * 80)
print("Direct EnCodec Test - What does it actually return?")
print("=" * 80)

# Load model
print("\n1. Loading EnCodec model (24kHz)...")
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
print("✓ Model loaded")

# Load just 10 seconds of audio from Sherlock
print("\n2. Loading 10 seconds of Sherlock audio...")
y, sr = librosa.load('data/stimuli_Sherlock.m4v', sr=24000, mono=True, duration=10.0)
print(f"✓ Audio loaded: {len(y)} samples at {sr} Hz")
print(f"  Duration: {len(y) / sr:.2f} seconds")

# Process with EnCodec
print("\n3. Encoding with EnCodec at different bandwidths...")

for bandwidth in [1.5, 3.0, 6.0]:
    print(f"\n--- Bandwidth: {bandwidth} kbps ---")

    inputs = processor(raw_audio=y, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        encoded = model.encode(
            inputs["input_values"],
            inputs["padding_mask"],
            bandwidth=bandwidth
        )

    codes = encoded.audio_codes[0]
    print(f"Codes shape: {codes.shape}")
    print(f"  n_codebooks: {codes.shape[0]}")
    print(f"  n_frames: {codes.shape[1]}")

    # Expected codebooks based on bandwidth
    bandwidth_to_codebooks = {
        1.5: 2,
        3.0: 8,
        6.0: 16,
        12.0: 32,
        24.0: 32
    }
    expected = bandwidth_to_codebooks.get(bandwidth, "unknown")
    print(f"  Expected codebooks: {expected}")

    if codes.shape[0] == expected:
        print(f"  ✓ Codebook count matches expectation")
    else:
        print(f"  ✗ Codebook count MISMATCH! Expected {expected}, got {codes.shape[0]}")

# Now test with TR-length windows
print("\n4. Testing TR-length slicing (TR=1.5s)...")
bandwidth = 3.0
tr_length = 1.5
encodec_frame_rate = 75.0  # EnCodec fixed frame rate
frames_per_tr = int(encodec_frame_rate * tr_length)

print(f"TR length: {tr_length}s")
print(f"EnCodec frame rate: {encodec_frame_rate} Hz")
print(f"Frames per TR: {frames_per_tr}")

inputs = processor(raw_audio=y, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    encoded = model.encode(
        inputs["input_values"],
        inputs["padding_mask"],
        bandwidth=bandwidth
    )

codes = encoded.audio_codes[0]
print(f"\nFull codes shape: {codes.shape}")

# Try to extract first TR
start_frame = 0
end_frame = frames_per_tr

tr_codes = codes[:, start_frame:end_frame]
print(f"First TR codes shape: {tr_codes.shape}")
print(f"  n_codebooks: {tr_codes.shape[0]}")
print(f"  n_frames: {tr_codes.shape[1]}")

# Expected shape
expected_codebooks = 8
expected_shape = (expected_codebooks, frames_per_tr)
print(f"Expected shape: {expected_shape}")

if tr_codes.shape == expected_shape:
    print("✓ Shape matches expectation")
else:
    print(f"✗ Shape mismatch!")
    print(f"  Expected: {expected_shape}")
    print(f"  Actual: {tr_codes.shape}")

# Try the normalization operation that was failing
print("\n5. Testing normalization operation...")
normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)
print(f"Target tensor shape: {normalized_codes.shape}")

try:
    n_available = min(tr_codes.shape[0], expected_codebooks)
    print(f"Copying {n_available} codebooks...")
    normalized_codes[:n_available, :] = tr_codes[:n_available, :]
    print(f"✓ Normalization successful")
    print(f"  Final shape: {normalized_codes.shape}")
except RuntimeError as e:
    print(f"✗ Normalization FAILED: {e}")

# Test flattening
print("\n6. Testing flattening...")
flat_codes = normalized_codes.reshape(-1)
print(f"Flattened shape: {flat_codes.shape}")
expected_flat = expected_codebooks * frames_per_tr
print(f"Expected flat size: {expected_flat}")

if flat_codes.shape[0] == expected_flat:
    print(f"✓ Flattening successful: {flat_codes.shape}")
else:
    print(f"✗ Flattening mismatch!")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)

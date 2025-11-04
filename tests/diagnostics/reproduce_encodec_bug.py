#!/usr/bin/env python3
"""Reproduce the EnCodec dimension bug locally with real Sherlock audio"""

import sys

sys.path.insert(0, ".")

from giblet.data.audio import AudioProcessor
import traceback
import torch
import numpy as np

print("=" * 80)
print("EnCodec Bug Reproduction with Real Sherlock Audio")
print("=" * 80)

processor = AudioProcessor(
    use_encodec=True, encodec_bandwidth=3.0, sample_rate=12000, tr=1.5
)

try:
    print("\n1. Testing with first 5 TRs...")
    features, metadata = processor.audio_to_features(
        "data/stimuli_Sherlock.m4v", max_trs=5, from_video=True
    )
    print(f"✓ Success! Shape: {features.shape}")
    print(f"✓ Dtype: {features.dtype}")
    print(f"\nMetadata:")
    print(metadata)

    # Check consistency
    print(f"\n2. Checking shape consistency across TRs...")
    shapes = [features[i].shape for i in range(len(features))]
    unique_shapes = set(shapes)
    print(f"Unique shapes: {unique_shapes}")

    if len(unique_shapes) == 1:
        print(f"✓ All TRs have consistent shape: {unique_shapes.pop()}")
    else:
        print(f"✗ INCONSISTENT SHAPES DETECTED!")
        for i, shape in enumerate(shapes):
            print(f"  TR {i}: {shape}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\n=== Full Traceback ===")
    traceback.print_exc()

    print("\n=== Debugging Info ===")
    print("Attempting to diagnose the exact issue...")

    # Load and process audio manually to see what EnCodec actually returns
    try:
        import librosa

        print("\nLoading audio from Sherlock video...")
        y, sr = librosa.load("data/stimuli_Sherlock.m4v", sr=24000, mono=False)
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        print(f"Audio loaded: {len(y)} samples at {sr} Hz")
        print(f"Duration: {len(y) / sr:.2f} seconds")

        # Process with EnCodec
        print("\nEncoding with EnCodec...")
        inputs = processor.encodec_processor(
            raw_audio=y, sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(processor.device) for k, v in inputs.items()}

        with torch.no_grad():
            encoded = processor.encodec_model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=processor.encodec_bandwidth,
            )

        codes = encoded.audio_codes[0].cpu()
        print(f"\nEnCodec output shape: {codes.shape}")
        print(f"  n_codebooks: {codes.shape[0]}")
        print(f"  n_frames: {codes.shape[1]}")

        # Calculate expected values
        encodec_frame_rate = 75.0
        tr_length = 1.5
        frames_per_tr = int(encodec_frame_rate * tr_length)
        expected_codebooks = 8  # for 3.0 kbps

        print(f"\nExpected values:")
        print(f"  Codebooks: {expected_codebooks}")
        print(f"  Frames per TR: {frames_per_tr}")
        print(f"  Expected flattened dim per TR: {expected_codebooks * frames_per_tr}")

        print(f"\nActual values:")
        print(f"  Codebooks: {codes.shape[0]}")
        print(f"  Total frames: {codes.shape[1]}")

        if codes.shape[0] != expected_codebooks:
            print(f"\n⚠ WARNING: Codebook count mismatch!")
            print(f"  Expected: {expected_codebooks}")
            print(f"  Actual: {codes.shape[0]}")
            print(f"  This is the source of the dimension error!")

        # Try to reproduce the exact error
        print("\n3. Attempting to reproduce the error...")
        start_frame = 0
        end_frame = frames_per_tr
        tr_codes = codes[:, start_frame:end_frame]
        print(f"TR codes shape: {tr_codes.shape}")

        # Try the problematic operation
        normalized_codes = torch.zeros(
            expected_codebooks, frames_per_tr, dtype=tr_codes.dtype
        )
        print(f"Normalized target shape: {normalized_codes.shape}")

        print(f"\nAttempting assignment: normalized_codes[:4, :] = tr_codes[:4, :]")
        print(f"  LHS shape: {normalized_codes[:4, :].shape}")
        print(f"  RHS shape: {tr_codes[:4, :].shape}")

        try:
            normalized_codes[:4, :] = tr_codes[:4, :]
            print("✓ Assignment successful")
        except RuntimeError as assign_error:
            print(f"✗ Assignment failed: {assign_error}")

    except Exception as debug_error:
        print(f"Debug attempt failed: {debug_error}")
        traceback.print_exc()

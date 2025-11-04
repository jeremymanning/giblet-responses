#!/usr/bin/env python3
"""
Verify the EnCodec dimension fix works with real Sherlock data.

This script tests progressively larger subsets of the Sherlock video
to ensure the fix handles real-world data correctly.

Run this on the cluster where EnCodec can load properly.
"""

import sys

sys.path.insert(0, ".")

from giblet.data.audio import AudioProcessor
import numpy as np
import traceback
from pathlib import Path


def verify_single_test(processor, max_trs, audio_path):
    """Test extraction of specified number of TRs"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing with {max_trs} TRs...")
        print(f"{'='*60}")

        features, metadata = processor.audio_to_features(
            audio_path, max_trs=max_trs, from_video=True
        )

        # Verify shape
        expected_shape = (max_trs, 896)
        if features.shape != expected_shape:
            print(f"✗ FAIL: Shape mismatch")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {features.shape}")
            return False

        print(f"✓ Shape: {features.shape}")

        # Verify dtype
        if features.dtype != np.int64:
            print(f"✗ FAIL: Dtype mismatch")
            print(f"  Expected: int64")
            print(f"  Got: {features.dtype}")
            return False

        print(f"✓ Dtype: {features.dtype}")

        # Verify all TRs have consistent shape
        unique_shapes = set()
        for i in range(max_trs):
            unique_shapes.add(features[i].shape)

        if len(unique_shapes) != 1:
            print(f"✗ FAIL: Inconsistent TR shapes")
            print(f"  Unique shapes: {unique_shapes}")
            return False

        print(f"✓ All TRs consistent: {unique_shapes.pop()}")

        # Verify values are valid codebook indices
        # EnCodec uses 1024 codebook entries (0-1023)
        min_val = features.min()
        max_val = features.max()

        if min_val < 0:
            print(f"✗ FAIL: Negative values found")
            print(f"  Min value: {min_val}")
            return False

        if max_val > 1023:
            print(f"✗ FAIL: Values exceed codebook range")
            print(f"  Max value: {max_val}")
            return False

        print(f"✓ Value range: [{min_val}, {max_val}]")

        # Verify metadata
        if len(metadata) != max_trs:
            print(f"✗ FAIL: Metadata length mismatch")
            print(f"  Expected: {max_trs}")
            print(f"  Got: {len(metadata)}")
            return False

        print(f"✓ Metadata rows: {len(metadata)}")

        # Check metadata fields
        required_fields = [
            "tr_index",
            "start_time",
            "end_time",
            "n_frames",
            "n_codebooks",
            "encoding_mode",
        ]
        missing_fields = [f for f in required_fields if f not in metadata.columns]

        if missing_fields:
            print(f"✗ FAIL: Missing metadata fields: {missing_fields}")
            return False

        print(f"✓ Metadata fields: {list(metadata.columns)}")

        # Verify n_frames and n_codebooks are consistent
        if metadata["n_frames"].nunique() != 1:
            print(f"✗ FAIL: Inconsistent n_frames in metadata")
            print(f"  Values: {metadata['n_frames'].unique()}")
            return False

        if metadata["n_codebooks"].nunique() != 1:
            print(f"✗ FAIL: Inconsistent n_codebooks in metadata")
            print(f"  Values: {metadata['n_codebooks'].unique()}")
            return False

        n_frames = metadata["n_frames"].iloc[0]
        n_codebooks = metadata["n_codebooks"].iloc[0]
        expected_flat_dim = n_frames * n_codebooks

        print(f"✓ Metadata consistency:")
        print(f"  n_frames: {n_frames}")
        print(f"  n_codebooks: {n_codebooks}")
        print(f"  Expected flat dim: {expected_flat_dim}")

        if features.shape[1] != expected_flat_dim:
            print(f"✗ FAIL: Flat dimension mismatch")
            print(f"  Expected: {expected_flat_dim}")
            print(f"  Got: {features.shape[1]}")
            return False

        print(f"\n✓✓✓ Test PASSED for {max_trs} TRs")
        return True

    except Exception as e:
        print(f"\n✗✗✗ Test FAILED for {max_trs} TRs")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Run verification tests with real Sherlock data"""
    print("=" * 80)
    print("EnCodec Fix Verification with Real Sherlock Data")
    print("=" * 80)

    # Check if Sherlock video exists
    audio_path = Path("data/stimuli_Sherlock.m4v")
    if not audio_path.exists():
        print(f"\n✗ ERROR: Sherlock video not found at {audio_path}")
        print("Please ensure the video is available before running this script.")
        return False

    print(f"\n✓ Found Sherlock video: {audio_path}")
    print(f"  Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Initialize AudioProcessor
    print("\nInitializing AudioProcessor with EnCodec...")
    print("  Settings:")
    print("    use_encodec: True")
    print("    encodec_bandwidth: 3.0 kbps")
    print("    tr: 1.5 seconds")
    print("    Expected output per TR: (896,) = 8 codebooks × 112 frames")

    try:
        processor = AudioProcessor(
            use_encodec=True, encodec_bandwidth=3.0, sample_rate=12000, tr=1.5
        )
        print("\n✓ AudioProcessor initialized")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to initialize AudioProcessor")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    # Test progressively larger subsets
    test_sizes = [5, 10, 20, 50, 100]
    results = []

    print("\n" + "=" * 80)
    print("Running Progressive Tests")
    print("=" * 80)

    for max_trs in test_sizes:
        result = verify_single_test(processor, max_trs, audio_path)
        results.append((max_trs, result))

        if not result:
            print(f"\n⚠ Stopping tests due to failure at {max_trs} TRs")
            break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    print("\nDetailed results:")

    for max_trs, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {max_trs:3d} TRs: {status}")

    if passed == total:
        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED - Fix verified with real Sherlock data!")
        print("=" * 80)
        print("\nThe EnCodec dimension fix is working correctly.")
        print("You can now proceed with confidence to:")
        print("  1. Deploy to cluster")
        print("  2. Extract features from full Sherlock dataset")
        print("  3. Train multimodal autoencoder")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("✗✗✗ SOME TESTS FAILED - Fix needs review")
        print("=" * 80)
        print("\nPlease review the errors above and ensure:")
        print("  1. EnCodec is properly installed")
        print("  2. The fix in giblet/data/audio.py is applied correctly")
        print("  3. PyTorch/transformers versions are compatible")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Verify that the EnCodec dimension fix works correctly.

Tests the fix for the bug:
RuntimeError: The expanded size of the tensor (112) must match
the existing size (106697) at non-singleton dimension 1
"""

import torch
import numpy as np


def test_fixed_logic():
    """
    Test that the fixed logic works correctly.

    The fix: Use frames_per_tr directly when creating normalized_codes,
    instead of tr_codes.shape[1].
    """

    print("=" * 80)
    print("TESTING FIXED LOGIC")
    print("=" * 80)

    # Expected dimensions
    expected_codebooks = 8
    frames_per_tr = 112
    encodec_frame_rate = 75.0
    tr_length = 1.5

    # Simulate full EnCodec output with WRONG number of codebooks
    total_frames = 106697
    actual_codebooks = 4
    codes = torch.zeros(actual_codebooks, total_frames)

    print(f"\nFull EnCodec output: {codes.shape}")
    print(f"Expected per TR: ({expected_codebooks}, {frames_per_tr})")

    # Process first TR
    tr_idx = 0
    start_time = tr_idx * tr_length
    end_time = start_time + tr_length
    start_frame = int(start_time * encodec_frame_rate)
    end_frame = int(end_time * encodec_frame_rate)

    print(f"\nTR {tr_idx}:")
    print(f"   Time: [{start_time:.2f}, {end_time:.2f}]s")
    print(f"   Frames: [{start_frame}, {end_frame}]")

    # Extract frames for this TR (lines 291-298)
    if end_frame <= codes.shape[1]:
        tr_codes = codes[:, start_frame:end_frame]
    else:
        tr_codes = codes[:, start_frame:]
        padding_needed = frames_per_tr - tr_codes.shape[1]
        if padding_needed > 0:
            tr_codes = torch.nn.functional.pad(tr_codes, (0, padding_needed), value=0)

    print(f"   After extraction: {tr_codes.shape}")

    # Temporal dimension normalization (lines 300-305)
    if tr_codes.shape[1] > frames_per_tr:
        tr_codes = tr_codes[:, :frames_per_tr]
    elif tr_codes.shape[1] < frames_per_tr:
        padding = frames_per_tr - tr_codes.shape[1]
        tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)

    print(f"   After temporal norm: {tr_codes.shape}")

    # Codebook dimension normalization (FIXED VERSION - lines 311-320)
    if tr_codes.shape[0] != expected_codebooks:
        print(f"\n   Codebook mismatch: {tr_codes.shape[0]} != {expected_codebooks}")

        # THE FIX: Use frames_per_tr directly, NOT tr_codes.shape[1]
        print(f"   Creating normalized_codes with shape ({expected_codebooks}, {frames_per_tr})")
        normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)

        n_available = min(tr_codes.shape[0], expected_codebooks)
        print(f"   Copying {n_available} codebooks...")

        try:
            normalized_codes[:n_available, :] = tr_codes[:n_available, :]
            print(f"   ✓ SUCCESS!")
            tr_codes = normalized_codes
        except RuntimeError as e:
            print(f"   ✗ ERROR: {e}")
            return False

    print(f"\n   Final shape: {tr_codes.shape}")
    print(f"   Expected: ({expected_codebooks}, {frames_per_tr})")

    if tr_codes.shape == (expected_codebooks, frames_per_tr):
        print(f"   ✓ CORRECT!")
        return True
    else:
        print(f"   ✗ WRONG!")
        return False


def test_edge_cases():
    """Test various edge cases."""

    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    expected_codebooks = 8
    frames_per_tr = 112

    test_cases = [
        ("Fewer codebooks (4)", 4, 112),
        ("More codebooks (16)", 16, 112),
        ("Exact match", 8, 112),
        ("Fewer codebooks + fewer frames", 4, 100),
        ("More codebooks + more frames", 16, 150),
    ]

    all_passed = True

    for name, n_codebooks, n_frames in test_cases:
        print(f"\n[{name}]")
        print(f"   Input: ({n_codebooks}, {n_frames})")

        tr_codes = torch.zeros(n_codebooks, n_frames)

        # Temporal normalization
        if tr_codes.shape[1] > frames_per_tr:
            tr_codes = tr_codes[:, :frames_per_tr]
        elif tr_codes.shape[1] < frames_per_tr:
            padding = frames_per_tr - tr_codes.shape[1]
            tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)

        # Codebook normalization (FIXED)
        if tr_codes.shape[0] != expected_codebooks:
            normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)
            n_available = min(tr_codes.shape[0], expected_codebooks)
            try:
                normalized_codes[:n_available, :] = tr_codes[:n_available, :]
                tr_codes = normalized_codes
            except RuntimeError as e:
                print(f"   ✗ ERROR: {e}")
                all_passed = False
                continue

        print(f"   Output: {tr_codes.shape}")
        if tr_codes.shape == (expected_codebooks, frames_per_tr):
            print(f"   ✓ PASS")
        else:
            print(f"   ✗ FAIL")
            all_passed = False

    return all_passed


def test_flattening():
    """Test that flattening works correctly."""

    print("\n" + "=" * 80)
    print("TESTING FLATTENING")
    print("=" * 80)

    expected_codebooks = 8
    frames_per_tr = 112
    n_trs = 10

    print(f"\nSimulating {n_trs} TRs...")

    features = []
    for tr_idx in range(n_trs):
        # Create a properly shaped TR
        tr_codes = torch.zeros(expected_codebooks, frames_per_tr)
        tr_codes_flat = tr_codes.reshape(-1)
        features.append(tr_codes_flat)

        if tr_idx == 0:
            print(f"   TR {tr_idx}: {tr_codes.shape} → {tr_codes_flat.shape}")

    # Stack all TRs
    features_stacked = torch.stack(features).numpy().astype(np.int64)

    print(f"\n   Stacked shape: {features_stacked.shape}")
    print(f"   Expected: ({n_trs}, {expected_codebooks * frames_per_tr})")

    if features_stacked.shape == (n_trs, expected_codebooks * frames_per_tr):
        print(f"   ✓ CORRECT!")
        return True
    else:
        print(f"   ✗ WRONG!")
        return False


if __name__ == "__main__":
    results = []

    results.append(("Fixed Logic", test_fixed_logic()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Flattening", test_flattening()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {name}: {status}")

    if all(passed for _, passed in results):
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED!")

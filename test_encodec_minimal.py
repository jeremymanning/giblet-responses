"""
Minimal reproduction of EnCodec dimension mismatch bug.

This reproduces the exact error:
RuntimeError: The expanded size of the tensor (112) must match
the existing size (106697) at non-singleton dimension 1
"""

import torch
import numpy as np


def test_dimension_mismatch_bug():
    """
    Reproduce the dimension mismatch at line 317 of audio.py.

    The bug occurs when:
    1. tr_codes has wrong temporal dimension (106697 instead of 112)
    2. We try to create normalized_codes with tr_codes.shape[1]
    3. But then assign from tr_codes which has a different shape
    """

    print("=" * 80)
    print("MINIMAL REPRODUCTION: EnCodec Dimension Mismatch Bug")
    print("=" * 80)

    # Expected dimensions
    expected_codebooks = 8
    frames_per_tr = 112

    # Simulate the bug: tr_codes has wrong temporal dimension
    # This could happen if the slicing logic fails
    print("\n[Scenario 1: Bug - tr_codes has TOTAL frames instead of TR frames]")
    total_frames = 106697  # Example: all frames from full audio
    actual_codebooks = 4   # And maybe fewer codebooks too

    tr_codes_buggy = torch.zeros(actual_codebooks, total_frames)
    print(f"   tr_codes shape: {tr_codes_buggy.shape}")
    print(f"   Expected: ({expected_codebooks}, {frames_per_tr})")

    # This is the problematic code from audio.py:313
    print(f"\n   Creating normalized_codes with shape ({expected_codebooks}, {tr_codes_buggy.shape[1]})")
    normalized_codes = torch.zeros(expected_codebooks, tr_codes_buggy.shape[1], dtype=tr_codes_buggy.dtype)
    print(f"   normalized_codes shape: {normalized_codes.shape}")

    # This is the problematic line 317
    print(f"\n   Attempting: normalized_codes[:n_available, :] = tr_codes[:n_available, :]")
    n_available = min(actual_codebooks, expected_codebooks)

    try:
        normalized_codes[:n_available, :] = tr_codes_buggy[:n_available, :]
        print(f"   ✓ SUCCESS: Assignment worked")
    except RuntimeError as e:
        print(f"   ✗ ERROR: {e}")
        print(f"\n   Analysis:")
        print(f"      LHS (normalized_codes[:n_available, :]): shape ({n_available}, {normalized_codes.shape[1]})")
        print(f"      RHS (tr_codes[:n_available, :]): shape ({n_available}, {tr_codes_buggy.shape[1]})")
        print(f"      These shapes match, so this should work...")

    # The REAL bug: What if tr_codes changes shape after being created?
    print("\n[Scenario 2: The REAL bug - Shape mismatch in how we slice]")

    # Let's simulate what actually happens in the code
    codes = torch.zeros(actual_codebooks, total_frames)  # Full EnCodec output
    print(f"   Full codes shape: {codes.shape}")

    tr_idx = 0
    tr_length = 1.5
    encodec_frame_rate = 75.0
    start_time = tr_idx * tr_length
    end_time = start_time + tr_length
    start_frame = int(start_time * encodec_frame_rate)
    end_frame = int(end_time * encodec_frame_rate)

    print(f"   TR {tr_idx}: start_frame={start_frame}, end_frame={end_frame}")
    print(f"   Expected frames_per_tr: {frames_per_tr}")

    # Extract frames
    if end_frame <= codes.shape[1]:
        tr_codes = codes[:, start_frame:end_frame]
    else:
        tr_codes = codes[:, start_frame:]
        padding_needed = frames_per_tr - tr_codes.shape[1]
        if padding_needed > 0:
            tr_codes = torch.nn.functional.pad(tr_codes, (0, padding_needed), value=0)

    print(f"   After extraction: tr_codes shape = {tr_codes.shape}")

    # Lines 300-305: Temporal dimension normalization
    if tr_codes.shape[1] > frames_per_tr:
        tr_codes = tr_codes[:, :frames_per_tr]
    elif tr_codes.shape[1] < frames_per_tr:
        padding = frames_per_tr - tr_codes.shape[1]
        tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)

    print(f"   After temporal normalization: tr_codes shape = {tr_codes.shape}")

    # Lines 311-318: Codebook dimension normalization
    if tr_codes.shape[0] != expected_codebooks:
        print(f"\n   Codebook mismatch detected: {tr_codes.shape[0]} != {expected_codebooks}")
        print(f"   Creating normalized_codes with shape ({expected_codebooks}, {tr_codes.shape[1]})")

        normalized_codes = torch.zeros(expected_codebooks, tr_codes.shape[1], dtype=tr_codes.dtype)
        n_available = min(tr_codes.shape[0], expected_codebooks)

        print(f"   Attempting assignment...")
        print(f"      normalized_codes[:n_available, :] = tr_codes[:n_available, :]")
        print(f"      LHS shape: ({n_available}, {normalized_codes.shape[1]})")
        print(f"      RHS shape: ({n_available}, {tr_codes.shape[1]})")

        try:
            normalized_codes[:n_available, :] = tr_codes[:n_available, :]
            print(f"   ✓ SUCCESS!")
            tr_codes = normalized_codes
        except RuntimeError as e:
            print(f"   ✗ ERROR: {e}")

    print(f"\n   Final tr_codes shape: {tr_codes.shape}")
    print(f"   Expected: ({expected_codebooks}, {frames_per_tr})")

    if tr_codes.shape == (expected_codebooks, frames_per_tr):
        print(f"   ✓ CORRECT SHAPE!")
    else:
        print(f"   ✗ WRONG SHAPE!")


def test_actual_cause():
    """
    Test what might ACTUALLY be causing the bug.

    Hypothesis: The issue is that somewhere, we're not slicing correctly
    and tr_codes retains the full temporal dimension.
    """

    print("\n" + "=" * 80)
    print("HYPOTHESIS: What if extraction doesn't slice correctly?")
    print("=" * 80)

    # Simulate the ACTUAL problematic scenario
    codes = torch.zeros(4, 106697)  # Full audio: 4 codebooks, 106697 frames
    expected_codebooks = 8
    frames_per_tr = 112

    print(f"\nFull codes: {codes.shape}")

    # What if we accidentally use codes directly instead of slicing?
    tr_codes = codes  # BUG: No slicing!

    print(f"tr_codes (BUGGY - no slicing): {tr_codes.shape}")

    # Now try the temporal normalization
    if tr_codes.shape[1] > frames_per_tr:
        print(f"\nTemporal dimension too large: {tr_codes.shape[1]} > {frames_per_tr}")
        print(f"Attempting to slice...")
        tr_codes = tr_codes[:, :frames_per_tr]
        print(f"After slicing: {tr_codes.shape}")

    # Now try the codebook normalization
    if tr_codes.shape[0] != expected_codebooks:
        print(f"\nCodebook dimension mismatch: {tr_codes.shape[0]} != {expected_codebooks}")
        print(f"Creating normalized_codes...")

        # BUG: What if we create with wrong temporal dimension?
        # Let's say we accidentally use the ORIGINAL codes shape
        normalized_codes = torch.zeros(expected_codebooks, codes.shape[1])  # BUG!
        print(f"normalized_codes: {normalized_codes.shape}")

        n_available = min(tr_codes.shape[0], expected_codebooks)
        print(f"\nAttempting: normalized_codes[:n_available, :] = tr_codes[:n_available, :]")
        print(f"   LHS: ({n_available}, {normalized_codes.shape[1]}) = {(n_available, normalized_codes.shape[1])}")
        print(f"   RHS: ({n_available}, {tr_codes.shape[1]}) = {(n_available, tr_codes.shape[1])}")

        try:
            normalized_codes[:n_available, :] = tr_codes[:n_available, :]
            print(f"✓ SUCCESS")
        except RuntimeError as e:
            print(f"✗ ERROR: {e}")
            print(f"\nTHIS IS THE BUG!")
            print(f"   The issue: normalized_codes was created with shape[1]={normalized_codes.shape[1]}")
            print(f"   But tr_codes has shape[1]={tr_codes.shape[1]}")
            print(f"   The assignment tries to expand tr_codes[{n_available}, {tr_codes.shape[1]}]")
            print(f"   to match normalized_codes[{n_available}, {normalized_codes.shape[1]}]")


if __name__ == "__main__":
    test_dimension_mismatch_bug()
    test_actual_cause()

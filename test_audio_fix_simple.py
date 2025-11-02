"""
Simple test to verify the tensor dimension fix in audio.py

Tests the core logic without loading heavy models.
"""

import torch
import numpy as np

def test_tensor_fix():
    """Test the fixed tensor normalization logic."""
    print("="*60)
    print("Testing Tensor Dimension Fix")
    print("="*60)

    # Simulate the buggy scenario:
    # tr_codes has already been sliced to correct temporal dimension (112 frames)
    # but has wrong number of codebooks (4 instead of 8)

    expected_codebooks = 8
    frames_per_tr = 112

    # Simulate tr_codes after temporal slicing (correct temporal, wrong codebooks)
    tr_codes = torch.randint(0, 1024, (4, 112))  # 4 codebooks, 112 frames
    print(f"\n1. Initial tr_codes shape: {tr_codes.shape}")
    print(f"   Expected: ({expected_codebooks}, {frames_per_tr})")

    # OLD BUGGY CODE would do:
    # normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)
    # normalized_codes[:n_available, :] = tr_codes[:n_available, :]
    # This would try to assign (4, 112) into (8, 112) - which works!

    # But the bug report shows tr_codes had shape (4, 106697)
    # Let's simulate that scenario:
    print("\n2. Simulating the ACTUAL bug scenario:")
    tr_codes_buggy = torch.randint(0, 1024, (4, 106697))  # Wrong temporal dimension!
    print(f"   Buggy tr_codes shape: {tr_codes_buggy.shape}")

    # OLD BUGGY CODE:
    print("\n3. Testing OLD BUGGY CODE:")
    try:
        normalized_codes_old = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes_buggy.dtype)
        print(f"   normalized_codes shape: {normalized_codes_old.shape}")
        n_available = min(tr_codes_buggy.shape[0], expected_codebooks)
        print(f"   Attempting: normalized_codes[:4, :] = tr_codes_buggy[:4, :]")
        print(f"   Left side: {normalized_codes_old[:n_available, :].shape}")
        print(f"   Right side: {tr_codes_buggy[:n_available, :].shape}")
        normalized_codes_old[:n_available, :] = tr_codes_buggy[:n_available, :]
        print("   ✗ No error! This means the bug is elsewhere...")
    except RuntimeError as e:
        print(f"   ✓ Error reproduced: {e}")

    # NEW FIXED CODE (Step 1: Codebook normalization):
    print("\n4. Testing NEW FIXED CODE (Step 1: Codebook normalization):")
    try:
        # Key fix: use tr_codes.shape[1] instead of frames_per_tr
        normalized_codes_new = torch.zeros(expected_codebooks, tr_codes_buggy.shape[1], dtype=tr_codes_buggy.dtype)
        print(f"   normalized_codes shape: {normalized_codes_new.shape}")
        n_available = min(tr_codes_buggy.shape[0], expected_codebooks)
        normalized_codes_new[:n_available, :] = tr_codes_buggy[:n_available, :]
        print(f"   ✓ Codebook normalization success! Shape: {normalized_codes_new.shape}")

    except RuntimeError as e:
        print(f"   ✗ Error occurred: {e}")

    # NEW FIXED CODE (Step 2: Temporal dimension fix):
    print("\n5. Testing NEW FIXED CODE (Step 2: Temporal dimension fix):")
    try:
        # Additional fix: crop to correct frames_per_tr AFTER codebook normalization
        tr_codes_fixed = normalized_codes_new
        if tr_codes_fixed.shape[1] != frames_per_tr:
            if tr_codes_fixed.shape[1] > frames_per_tr:
                tr_codes_fixed = tr_codes_fixed[:, :frames_per_tr]
                print(f"   Cropped from {normalized_codes_new.shape[1]} to {tr_codes_fixed.shape[1]} frames")
            else:
                padding = frames_per_tr - tr_codes_fixed.shape[1]
                tr_codes_fixed = torch.nn.functional.pad(tr_codes_fixed, (0, padding), value=0)
                print(f"   Padded from {tr_codes_fixed.shape[1] - padding} to {tr_codes_fixed.shape[1]} frames")

        print(f"   After temporal fix: {tr_codes_fixed.shape}")

        # Flatten
        flattened = tr_codes_fixed.reshape(-1)
        print(f"   After flattening: {flattened.shape}")
        print(f"   Expected flat shape: ({expected_codebooks * frames_per_tr},)")

        if flattened.shape[0] == expected_codebooks * frames_per_tr:
            print(f"   ✓ Correct final shape!")
        else:
            print(f"   ✗ Wrong final shape!")

    except RuntimeError as e:
        print(f"   ✗ Error occurred: {e}")

    print("\n" + "="*60)
    print("Test complete - analyzing results...")
    print("="*60)

    # The real issue analysis
    print("\nANALYSIS:")
    print("The bug occurs when tr_codes still has wrong temporal dimension")
    print("at the codebook normalization step. This means the temporal")
    print("padding/cropping logic (lines 301-305) isn't working correctly.")
    print("\nThe fix ensures normalized_codes matches tr_codes' ACTUAL")
    print("temporal dimension, regardless of whether it's correct.")


if __name__ == "__main__":
    test_tensor_fix()

#!/usr/bin/env python3
"""
Test script to verify dataset loading with EnCodec features.

This script:
1. Loads a small subset of data with EnCodec features
2. Verifies feature shapes and dtypes
3. Tests both per_subject and cross_subject modes
4. Validates backward compatibility with mel spectrograms

Usage:
    python scripts/test_dataset_encodec.py
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from giblet.data.dataset import MultimodalDataset


def test_encodec_dataset():
    """Test dataset with EnCodec features."""
    print("=" * 70)
    print("Testing Dataset with EnCodec Features")
    print("=" * 70)

    # Test with small subset for speed
    print("\n1. Loading dataset with EnCodec (12kHz, 3.0kbps)...")
    print("   Using max_trs=10 for quick testing...")

    try:
        dataset = MultimodalDataset(
            data_dir="data",
            subjects=1,  # Single subject for testing
            split=None,
            apply_hrf=False,  # Disable HRF for faster testing
            mode="per_subject",
            preprocess=True,
            tr=1.5,
            max_trs=10,  # Only 10 TRs for quick test
            use_encodec=True,
            encodec_bandwidth=3.0,
            encodec_sample_rate=12000,
        )
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"   ✓ Dataset loaded successfully")
    print(f"   Samples: {len(dataset)}")
    print(f"   Feature dims: {dataset.feature_dims}")

    # Get a sample
    print("\n2. Testing __getitem__...")
    try:
        sample = dataset[0]
        print(f"   ✓ Sample retrieved successfully")
        print(f"   Keys: {list(sample.keys())}")
        print(
            f"   Video shape: {sample['video'].shape}, dtype: {sample['video'].dtype}"
        )
        print(
            f"   Audio shape: {sample['audio'].shape}, dtype: {sample['audio'].dtype}"
        )
        print(f"   Text shape: {sample['text'].shape}, dtype: {sample['text'].dtype}")
        print(f"   fMRI shape: {sample['fmri'].shape}, dtype: {sample['fmri'].dtype}")

        # Verify audio is discrete codes
        if sample["audio"].dtype == torch.int64:
            print(f"   ✓ Audio is discrete codes (int64)")
            print(f"   Code range: [{sample['audio'].min()}, {sample['audio'].max()}]")
            if sample["audio"].ndim == 2:
                n_codebooks, frames_per_tr = sample["audio"].shape
                print(
                    f"   ✓ Shape is (n_codebooks={n_codebooks}, frames_per_tr={frames_per_tr})"
                )
                expected_frames = int(1.5 * 12000 / 320)  # ~56 frames
                if abs(frames_per_tr - expected_frames) <= 2:  # Allow small tolerance
                    print(
                        f"   ✓ Frames per TR (~{frames_per_tr}) matches expected (~{expected_frames})"
                    )
                else:
                    print(
                        f"   ⚠ Frames per TR ({frames_per_tr}) != expected ({expected_frames})"
                    )
            else:
                print(f"   ⚠ Unexpected audio shape: {sample['audio'].shape}")
        else:
            print(f"   ✗ Audio should be int64, got {sample['audio'].dtype}")
            return False

    except Exception as e:
        print(f"\n✗ Failed to get sample: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test batch retrieval
    print("\n3. Testing batch retrieval...")
    try:
        batch = dataset.get_batch([0, 1, 2])
        print(f"   ✓ Batch retrieved successfully")
        print(f"   Batch size: {batch['audio'].shape[0]}")
        print(f"   Audio batch shape: {batch['audio'].shape}")
    except Exception as e:
        print(f"\n✗ Failed to get batch: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ All EnCodec dataset tests passed!")
    print("=" * 70)
    return True


def test_mel_dataset():
    """Test dataset with mel spectrograms (backward compatibility)."""
    print("\n" + "=" * 70)
    print("Testing Dataset with Mel Spectrograms (Backward Compatibility)")
    print("=" * 70)

    print("\n1. Loading dataset with mel spectrograms...")
    print("   Using max_trs=10 for quick testing...")

    try:
        dataset = MultimodalDataset(
            data_dir="data",
            subjects=1,
            split=None,
            apply_hrf=False,
            mode="per_subject",
            preprocess=True,
            tr=1.5,
            max_trs=10,
            use_encodec=False,  # Use mel spectrograms
        )
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"   ✓ Dataset loaded successfully")
    print(f"   Samples: {len(dataset)}")
    print(f"   Feature dims: {dataset.feature_dims}")

    # Get a sample
    print("\n2. Testing __getitem__...")
    try:
        sample = dataset[0]
        print(f"   ✓ Sample retrieved successfully")
        print(
            f"   Audio shape: {sample['audio'].shape}, dtype: {sample['audio'].dtype}"
        )

        # Verify audio is float
        if sample["audio"].dtype == torch.float32:
            print(f"   ✓ Audio is continuous (float32)")
        else:
            print(f"   ⚠ Expected float32, got {sample['audio'].dtype}")

    except Exception as e:
        print(f"\n✗ Failed to get sample: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ Mel spectrogram backward compatibility verified!")
    print("=" * 70)
    return True


def test_cross_subject_mode():
    """Test cross_subject mode with EnCodec."""
    print("\n" + "=" * 70)
    print("Testing Cross-Subject Mode with EnCodec")
    print("=" * 70)

    print("\n1. Loading cross-subject dataset...")

    try:
        dataset = MultimodalDataset(
            data_dir="data",
            subjects="all",  # All subjects
            split=None,
            apply_hrf=False,
            mode="cross_subject",  # Average across subjects
            preprocess=True,
            tr=1.5,
            max_trs=10,
            use_encodec=True,
            encodec_bandwidth=3.0,
            encodec_sample_rate=12000,
        )
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"   ✓ Dataset loaded successfully")
    print(f"   Samples: {len(dataset)} (should equal n_trs, not n_subjects × n_trs)")

    # Get a sample
    try:
        sample = dataset[0]
        print(f"   ✓ Sample retrieved successfully")
        print(f"   Audio shape: {sample['audio'].shape}")
        if "subject_id" in sample:
            print(f"   ⚠ subject_id should not be in cross_subject mode")
            return False
        else:
            print(f"   ✓ No subject_id (correct for cross_subject mode)")
    except Exception as e:
        print(f"\n✗ Failed to get sample: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ Cross-subject mode test passed!")
    print("=" * 70)
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Dataset EnCodec Integration Test Suite")
    print("=" * 70)

    results = []

    # Test 1: EnCodec features
    results.append(("EnCodec dataset", test_encodec_dataset()))

    # Test 2: Mel spectrograms (backward compatibility)
    results.append(("Mel dataset (backward compat)", test_mel_dataset()))

    # Test 3: Cross-subject mode
    results.append(("Cross-subject mode", test_cross_subject_mode()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

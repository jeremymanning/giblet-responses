"""
Quick test of EnCodec fix with real Sherlock data.

Tests progressively larger numbers of TRs to verify the fix works.
"""

from pathlib import Path

import pytest

from giblet.data.audio import AudioProcessor


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
def test_sherlock_extraction(data_dir):
    """Test EnCodec extraction with progressively more TRs."""

    print("=" * 80)
    print("TESTING ENCODEC FIX WITH SHERLOCK DATA")
    print("=" * 80)

    video_path = data_dir / "stimuli_Sherlock.m4v"
    if not video_path.exists():
        pytest.skip(f"Sherlock video not found at {video_path}")

    # Initialize processor
    print("\n[1] Initializing AudioProcessor...")
    processor = AudioProcessor(
        use_encodec=True, encodec_bandwidth=3.0, tr=1.5, device="cpu"
    )
    print(f"   EnCodec enabled")

    # Test with increasing numbers of TRs
    test_trs = [5, 10, 20, 50, 100]

    results = []

    for max_trs in test_trs:
        print(f"\n[2] Testing with {max_trs} TRs...")

        try:
            features, metadata = processor.audio_to_features(
                str(video_path), max_trs=max_trs, from_video=True
            )

            print(f"   SUCCESS!")
            print(f"      Shape: {features.shape}")
            print(f"      dtype: {features.dtype}")

            # Verify shape
            expected_shape = (max_trs, 896)  # 8 codebooks × 112 frames
            if features.shape == expected_shape:
                print(f"      Correct shape: {expected_shape}")
                results.append((max_trs, True, None))
            else:
                print(
                    f"      Wrong shape! Expected {expected_shape}, got {features.shape}"
                )
                results.append((max_trs, False, f"Shape mismatch"))

            # Verify all TRs have consistent dimensions
            all_consistent = all(
                features[i].shape == features[0].shape for i in range(len(features))
            )
            if all_consistent:
                print(f"      All TRs consistent")
            else:
                print(f"      Inconsistent TR shapes")
                results.append((max_trs, False, "Inconsistent shapes"))

            # Check metadata
            print(f"      Metadata: {len(metadata)} rows")

        except Exception as e:
            print(f"   ERROR: {e}")
            results.append((max_trs, False, str(e)))

            # Don't continue if we hit an error
            break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for max_trs, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"   {max_trs} TRs: {status}")
        if error:
            print(f"      Error: {error}")

    # Assert all tests passed
    assert all(passed for _, passed, _ in results), "Some tests failed"
    print("\nALL TESTS PASSED!")

"""
Test script to verify the EnCodec audio encoding bug fix.

This tests that:
1. All TRs have consistent dimensions
2. No RuntimeError occurs during encoding
3. Features shape is (n_trs, 896) for 8 codebooks × 112 frames
"""

import sys
from pathlib import Path
import numpy as np

# Add giblet to path
sys.path.insert(0, str(Path(__file__).parent))

from giblet.data.audio import AudioProcessor

def test_encodec_encoding():
    """Test EnCodec encoding produces consistent dimensions."""
    print("="*60)
    print("Testing EnCodec Audio Encoding Fix")
    print("="*60)

    # Initialize processor
    print("\n1. Initializing AudioProcessor...")
    proc = AudioProcessor(
        use_encodec=True,
        encodec_bandwidth=3.0,
        tr=1.5,
        device='cpu'
    )

    # Test with first 20 TRs
    print("\n2. Encoding audio from video (first 20 TRs)...")
    video_path = Path('data/stimuli_Sherlock.m4v')

    if not video_path.exists():
        print(f"ERROR: Video file not found at {video_path}")
        print("Please run ./download_data_from_dropbox.sh first")
        return False

    try:
        features, metadata = proc.audio_to_features(
            video_path,
            max_trs=20,
            from_video=True
        )
        print(f"   ✓ Encoding successful!")

    except RuntimeError as e:
        print(f"   ✗ RuntimeError occurred: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False

    # Check shape
    print(f"\n3. Checking dimensions...")
    print(f"   Features shape: {features.shape}")
    print(f"   Expected shape: (20, 896)")

    if features.shape != (20, 896):
        print(f"   ✗ Shape mismatch!")
        return False
    else:
        print(f"   ✓ Shape matches expected!")

    # Check all TRs have same shape
    print(f"\n4. Verifying all TRs have consistent dimensions...")
    all_same = all(features[i].shape == features[0].shape for i in range(20))

    if not all_same:
        print(f"   ✗ TRs have inconsistent shapes!")
        for i in range(20):
            print(f"      TR {i}: {features[i].shape}")
        return False
    else:
        print(f"   ✓ All TRs have shape {features[0].shape}")

    # Check dtype
    print(f"\n5. Checking dtype...")
    print(f"   Dtype: {features.dtype}")

    if features.dtype != np.int64:
        print(f"   ✗ Expected int64, got {features.dtype}")
        return False
    else:
        print(f"   ✓ Dtype is int64")

    # Check metadata
    print(f"\n6. Checking metadata...")
    print(f"   Metadata shape: {metadata.shape}")
    print(f"   Columns: {list(metadata.columns)}")

    expected_cols = ['tr_index', 'start_time', 'end_time', 'n_frames', 'n_codebooks', 'encoding_mode']
    if list(metadata.columns) != expected_cols:
        print(f"   ✗ Unexpected columns!")
        return False
    else:
        print(f"   ✓ Metadata columns correct")

    # Show sample metadata
    print(f"\n   First 3 TRs metadata:")
    print(metadata.head(3).to_string(index=False))

    # Check codebook count consistency
    print(f"\n7. Checking codebook count consistency...")
    unique_codebooks = metadata['n_codebooks'].unique()
    print(f"   Unique codebook counts: {unique_codebooks}")

    if len(unique_codebooks) != 1 or unique_codebooks[0] != 8:
        print(f"   ✗ Inconsistent or incorrect codebook counts!")
        return False
    else:
        print(f"   ✓ All TRs have 8 codebooks")

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_encodec_encoding()
    sys.exit(0 if success else 1)

"""
Demo of temporal concatenation for video processing.

This script demonstrates the new temporal concatenation approach where each TR
contains all frames from the preceding temporal window [t-TR, t], rather than
an average of those frames.

Usage:
    python examples/video_temporal_concatenation_demo.py

Requires:
    - Sherlock stimulus video at data/stimuli_Sherlock.m4v
    - Run ./download_data_from_dropbox.sh if needed
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from giblet.data.video import VideoProcessor


def main():
    print("=" * 70)
    print("Video Temporal Concatenation Demo")
    print("=" * 70)

    # Check if Sherlock stimulus exists
    video_path = Path("data/stimuli_Sherlock.m4v")

    if not video_path.exists():
        print(f"\nError: Video not found at {video_path}")
        print("This demo requires the Sherlock stimulus video.")
        print("Run ./download_data_from_dropbox.sh to download the dataset.")
        return

    # Test with different TR lengths
    tr_lengths = [1.0, 1.5, 2.0]

    print(f"\nVideo: {video_path.name}")
    print(f"Testing TR lengths: {tr_lengths}")
    print()

    for tr in tr_lengths:
        print(f"\n{'-' * 70}")
        print(f"TR = {tr}s")
        print(f"{'-' * 70}")

        # Create processor
        processor = VideoProcessor(
            target_height=90, target_width=160, tr=tr, normalize=True
        )

        # Get video info
        info = processor.get_video_info(video_path)
        print(f"Video properties:")
        print(f"  FPS: {info['fps']}")
        print(f"  Duration: {info['duration']:.2f}s")
        print(f"  Total frames: {info['total_frames']}")
        print(f"  Expected TRs: {info['n_trs']}")

        # Extract features (just first 5 TRs for demo)
        features, metadata = processor.video_to_features(video_path, max_trs=5)

        print(f"\nFeature extraction results:")
        print(f"  Features shape: {features.shape}")
        print(f"  TRs extracted: {features.shape[0]}")
        print(f"  Features per TR: {features.shape[1]:,}")

        # Calculate breakdown
        frames_per_tr = metadata.iloc[0]["frames_per_tr"]
        features_per_frame = 90 * 160 * 3
        print(f"\nDimension breakdown:")
        print(f"  Frames per TR: {frames_per_tr}")
        print(f"  Features per frame: {features_per_frame:,} (90 × 160 × 3)")
        print(
            f"  Total features: {frames_per_tr} × {features_per_frame:,} = {features.shape[1]:,}"
        )

        # Verify consistency
        all_same = all(
            features[i].shape[0] == features[0].shape[0] for i in range(len(features))
        )
        print(f"\nDimension consistency: {'✓ PASS' if all_same else '✗ FAIL'}")

        # Show metadata for first few TRs
        print(f"\nMetadata (first 3 TRs):")
        print(metadata.head(3).to_string(index=False))

    print(f"\n{'=' * 70}")
    print("Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

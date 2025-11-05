#!/usr/bin/env python3
"""
Precompute EnCodec features for Sherlock stimulus.

This script processes the Sherlock video, extracts audio, encodes it with EnCodec,
and saves the features to disk for fast loading during training.

Usage:
    python scripts/precompute_encodec_features.py [--bandwidth BANDWIDTH] [--sample-rate RATE]

Default configuration (approved for Issue #24):
    - Sample rate: 12 kHz
    - Bandwidth: 3.0 kbps
    - ~56 frames per TR (1.5s)
    - 8 codebooks
    - Discrete integer codes [0, 1023]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from giblet.data.audio import AudioProcessor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Precompute EnCodec features for Sherlock stimulus"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=3.0,
        help="EnCodec bandwidth in kbps (default: 3.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=12000,
        help="Audio sample rate in Hz (default: 12000)",
    )
    parser.add_argument(
        "--max-trs",
        type=int,
        default=None,
        help="Maximum number of TRs to process (default: all 920)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recomputation even if cache exists"
    )

    args = parser.parse_args()

    # Paths
    data_dir = Path(args.data_dir)
    video_path = data_dir / "stimuli_Sherlock.m4v"
    cache_dir = (
        data_dir / "cache" / f"encodec_{args.sample_rate//1000}khz_{args.bandwidth}kbps"
    )
    cache_path = cache_dir / f"{video_path.stem}_encodec.npz"

    # Check if already exists
    if cache_path.exists() and not args.force:
        print(f"✓ EnCodec features already cached at: {cache_path}")
        print(f"  Size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("\nUse --force to recompute.")
        return

    # Verify video exists
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        print("\nPlease ensure Sherlock stimulus is downloaded:")
        print("  ./download_data_from_dropbox.sh")
        sys.exit(1)

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("EnCodec Feature Precomputation")
    print("=" * 70)
    print(f"Video:        {video_path}")
    print(f"Sample rate:  {args.sample_rate} Hz")
    print(f"Bandwidth:    {args.bandwidth} kbps")
    print("TR:           1.5 seconds")
    print("Max TRs:      {}".format(args.max_trs if args.max_trs else "all (~920)"))
    print(f"Cache path:   {cache_path}")
    print("=" * 70)

    # Initialize AudioProcessor with EnCodec
    print("\nInitializing EnCodec model...")
    processor = AudioProcessor(
        use_encodec=True,
        encodec_bandwidth=args.bandwidth,
        sample_rate=args.sample_rate,
        tr=1.5,
    )

    # Process audio
    print("\nProcessing audio with EnCodec...")
    print("This may take several minutes for the full 23-minute video...")
    start_time = time.time()

    try:
        features, metadata = processor.audio_to_features(
            video_path, max_trs=args.max_trs, from_video=True
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total time:        {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"TRs processed:     {features.shape[0]}")
    print(f"Feature shape:     {features.shape}")
    print(f"Feature dtype:     {features.dtype}")
    print(f"Codebooks:         {features.shape[1]}")
    print(f"Frames per TR:     {features.shape[2]}")
    print(f"Code range:        [{features.min()}, {features.max()}]")

    # Verify codes are valid
    assert features.dtype in [
        np.int32,
        np.int64,
    ], f"Expected integer dtype, got {features.dtype}"
    assert features.min() >= 0, f"Codes should be >= 0, got min={features.min()}"
    assert features.max() <= 1023, f"Codes should be <= 1023, got max={features.max()}"
    print("✓ Code validation passed")

    # Save to cache
    print(f"\nSaving to cache: {cache_path}")
    np.savez_compressed(
        cache_path,
        features=features,
        tr_indices=metadata["tr_index"].values,
        start_times=metadata["start_time"].values,
        end_times=metadata["end_time"].values,
        n_frames=metadata["n_frames"].values,
    )

    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"✓ Cached {cache_size_mb:.1f} MB")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Mean code value:   {features.mean():.2f}")
    print(f"Std code value:    {features.std():.2f}")
    print(f"Unique codes:      {len(np.unique(features))}")
    print(f"Memory usage:      {features.nbytes / 1024 / 1024:.1f} MB (in RAM)")
    print(f"Compression:       {cache_size_mb:.1f} MB (on disk, compressed)")
    print(f"Compression ratio: {features.nbytes / cache_path.stat().st_size:.2f}x")

    # Print per-codebook statistics
    print("\nPer-Codebook Statistics:")
    for i in range(features.shape[1]):
        codebook = features[:, i, :]
        print(
            f"  Codebook {i}: min={codebook.min()}, max={codebook.max()}, "
            f"mean={codebook.mean():.1f}, std={codebook.std():.1f}, "
            f"unique={len(np.unique(codebook))}"
        )

    print("\n" + "=" * 70)
    print("✓ Precomputation Complete!")
    print("=" * 70)
    print("\nEnCodec features ready for training.")
    print(f"Cache location: {cache_path}")
    print("\nDataset will automatically load from cache during training.")


if __name__ == "__main__":
    main()

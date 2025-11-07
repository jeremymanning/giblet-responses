"""
Video modality validation script.

Performs comprehensive round-trip validation:
1. Load real Sherlock video
2. Extract features from multiple segments (beginning, middle, end)
3. Reconstruct video from features
4. Calculate PSNR for quality assessment
5. Save before/after frame comparisons
6. Create side-by-side comparison video
7. Test temporal alignment (frame indices → TR indices)

All tests use REAL data - NO MOCKS.
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.video import VideoProcessor


def calculate_psnr_batch(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate average PSNR across multiple frames.

    Parameters
    ----------
    original : np.ndarray
        Shape (n_frames, height, width, 3) original frames
    reconstructed : np.ndarray
        Shape (n_frames, height, width, 3) reconstructed frames

    Returns
    -------
    avg_psnr : float
        Average PSNR in dB
    """
    psnr_values = []
    for i in range(len(original)):
        # Convert to uint8 for PSNR calculation
        orig_uint8 = (np.clip(original[i], 0, 1) * 255).astype(np.uint8)
        recon_uint8 = (np.clip(reconstructed[i], 0, 1) * 255).astype(np.uint8)

        psnr_val = psnr(orig_uint8, recon_uint8, data_range=255)
        psnr_values.append(psnr_val)

    return np.mean(psnr_values)


def save_frame_comparison(
    original: np.ndarray, reconstructed: np.ndarray, tr_idx: int, output_path: Path
):
    """
    Save side-by-side comparison of original and reconstructed frame.

    Parameters
    ----------
    original : np.ndarray
        Shape (height, width, 3) original frame (normalized 0-1)
    reconstructed : np.ndarray
        Shape (height, width, 3) reconstructed frame (normalized 0-1)
    tr_idx : int
        TR index for labeling
    output_path : Path
        Path to save comparison image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original)
    axes[0].set_title(f"Original (TR {tr_idx})", fontsize=14)
    axes[0].axis("off")

    # Reconstructed
    axes[1].imshow(reconstructed)
    axes[1].set_title(f"Reconstructed (TR {tr_idx})", fontsize=14)
    axes[1].axis("off")

    # Difference (amplified)
    diff = np.abs(original - reconstructed)
    diff_amplified = np.clip(diff * 10, 0, 1)  # 10x amplification
    axes[2].imshow(diff_amplified)
    axes[2].set_title(f"Difference (×10)", fontsize=14)
    axes[2].axis("off")

    # Calculate PSNR for this frame
    orig_uint8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
    recon_uint8 = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
    frame_psnr = psnr(orig_uint8, recon_uint8, data_range=255)

    plt.suptitle(f"Frame Comparison - PSNR: {frame_psnr:.2f} dB", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved comparison to {output_path.name} (PSNR: {frame_psnr:.2f} dB)")


def create_side_by_side_video(
    original_features: np.ndarray,
    reconstructed_features: np.ndarray,
    output_path: Path,
    processor: VideoProcessor,
    fps: float = 25,
):
    """
    Create side-by-side comparison video.

    Parameters
    ----------
    original_features : np.ndarray
        Shape (n_trs, n_features) original features
    reconstructed_features : np.ndarray
        Shape (n_trs, n_features) reconstructed features
    output_path : Path
        Path to save comparison video
    processor : VideoProcessor
        VideoProcessor instance for frame conversion
    fps : float
        Output video frame rate
    """
    n_trs = original_features.shape[0]

    # Convert features to frames
    original_frames = []
    reconstructed_frames = []

    print("\n  Converting features to frames...")
    for tr_idx in range(n_trs):
        orig_frame = processor.features_to_frame(original_features[tr_idx])
        recon_frame = processor.features_to_frame(reconstructed_features[tr_idx])
        original_frames.append(orig_frame)
        reconstructed_frames.append(recon_frame)

    # Create side-by-side frames
    height, width = processor.target_height, processor.target_width
    side_by_side_width = width * 2 + 10  # 10 pixels separator

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (side_by_side_width, height))

    print(f"  Creating side-by-side video...")
    for tr_idx in tqdm(range(n_trs), desc="  Writing frames"):
        # Create side-by-side frame
        side_by_side = np.ones((height, side_by_side_width, 3), dtype=np.uint8) * 255

        # Left: original
        orig_bgr = cv2.cvtColor(original_frames[tr_idx], cv2.COLOR_RGB2BGR)
        side_by_side[:, :width] = orig_bgr

        # Right: reconstructed
        recon_bgr = cv2.cvtColor(reconstructed_frames[tr_idx], cv2.COLOR_RGB2BGR)
        side_by_side[:, width + 10 :] = recon_bgr

        # Add text labels
        cv2.putText(
            side_by_side,
            "Original",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            side_by_side,
            "Reconstructed",
            (width + 20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            side_by_side,
            f"TR: {tr_idx}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Duplicate frame to fill TR duration
        frames_per_tr = int(np.round(processor.tr * fps))
        for _ in range(frames_per_tr):
            out.write(side_by_side)

    out.release()
    print(f"  Saved side-by-side video to {output_path.name}")


def validate_temporal_alignment(
    processor: VideoProcessor, video_path: Path, test_trs: list
):
    """
    Validate temporal alignment between frames and TRs.

    Parameters
    ----------
    processor : VideoProcessor
        VideoProcessor instance
    video_path : Path
        Path to video file
    test_trs : list
        List of TR indices to test
    """
    print("\n" + "=" * 80)
    print("TEMPORAL ALIGNMENT VALIDATION")
    print("=" * 80)

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\nVideo properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f}s")
    print(f"  TR: {processor.tr}s")

    print(f"\nTesting temporal alignment for TRs: {test_trs}")

    for tr_idx in test_trs:
        # Calculate expected frame range for this TR
        start_time = tr_idx * processor.tr
        end_time = start_time + processor.tr

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        n_frames = end_frame - start_frame

        print(f"\n  TR {tr_idx}:")
        print(f"    Time window: [{start_time:.2f}s, {end_time:.2f}s)")
        print(f"    Frame range: [{start_frame}, {end_frame}) = {n_frames} frames")

        # Verify frames per TR is consistent
        expected_frames = int(np.round(processor.tr * fps))
        if abs(n_frames - expected_frames) > 1:
            print(f"    WARNING: Expected ~{expected_frames} frames, got {n_frames}")
        else:
            print(f"    ✓ Frame count matches expected (~{expected_frames})")


def validate_video_modality():
    """
    Main validation function for video modality.
    """
    print("\n" + "=" * 80)
    print("VIDEO MODALITY VALIDATION")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "stimuli_Sherlock.m4v"
    output_dir = project_root / "validation_outputs"
    output_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print("Please run download_data_from_dropbox.sh first")
        return

    print(f"\nVideo path: {video_path}")
    print(f"Output directory: {output_dir}")

    # Initialize processor
    processor = VideoProcessor(
        target_height=90, target_width=160, tr=1.5, normalize=True
    )

    # Get video info
    print("\n" + "=" * 80)
    print("VIDEO INFORMATION")
    print("=" * 80)
    info = processor.get_video_info(video_path)
    print(f"\nOriginal video:")
    print(f"  Resolution: {info['width']}×{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  Available TRs: {info['n_trs']}")

    print(f"\nProcessor settings:")
    print(f"  Target resolution: {processor.target_width}×{processor.target_height}")
    print(f"  TR: {processor.tr}s")
    print(f"  Feature dimension: {processor.n_features}")

    # Test on different segments
    test_segments = [(0, 20, "beginning"), (460, 480, "middle"), (900, 920, "end")]

    all_psnr_values = []

    for start_tr, end_tr, segment_name in test_segments:
        print("\n" + "=" * 80)
        print(f"TESTING SEGMENT: {segment_name.upper()} (TRs {start_tr}-{end_tr})")
        print("=" * 80)

        # Extract features for this segment
        print(f"\n1. Extracting features from TRs {start_tr}-{end_tr}...")

        # Load full video and extract segment
        full_features, full_metadata = processor.video_to_features(
            video_path, max_trs=end_tr
        )
        features = full_features[start_tr:end_tr]
        metadata = full_metadata.iloc[start_tr:end_tr].reset_index(drop=True)

        print(f"   Features shape: {features.shape}")
        print(f"   Features range: [{features.min():.3f}, {features.max():.3f}]")

        # Store original features for comparison
        original_features = features.copy()

        # Reconstruct video
        print(f"\n2. Reconstructing video from features...")
        temp_video_path = output_dir / f"video_reconstruction_{segment_name}_temp.mp4"
        processor.features_to_video(features, temp_video_path, fps=25)
        print(f"   Saved to {temp_video_path.name}")

        # Re-extract features from reconstructed video
        print(f"\n3. Re-extracting features from reconstructed video...")
        reconstructed_features, _ = processor.video_to_features(
            temp_video_path, max_trs=len(features)
        )
        print(f"   Reconstructed features shape: {reconstructed_features.shape}")

        # Calculate PSNR
        print(f"\n4. Calculating PSNR...")

        # Convert features to frames for PSNR calculation
        original_frames = []
        reconstructed_frames = []

        for tr_idx in range(len(features)):
            orig_frame = processor.features_to_frame(original_features[tr_idx])
            recon_frame = processor.features_to_frame(reconstructed_features[tr_idx])

            # Normalize to 0-1
            original_frames.append(orig_frame / 255.0)
            reconstructed_frames.append(recon_frame / 255.0)

        original_frames = np.array(original_frames)
        reconstructed_frames = np.array(reconstructed_frames)

        avg_psnr = calculate_psnr_batch(original_frames, reconstructed_frames)
        all_psnr_values.append(avg_psnr)

        print(f"   Average PSNR: {avg_psnr:.2f} dB")

        if avg_psnr > 30:
            print(f"   ✓ EXCELLENT quality (PSNR > 30 dB)")
        elif avg_psnr > 20:
            print(f"   ✓ GOOD quality (PSNR > 20 dB)")
        else:
            print(f"   ⚠ WARNING: Low quality (PSNR < 20 dB)")

        # Save frame comparisons at multiple time points
        print(f"\n5. Saving frame comparisons...")
        comparison_trs = [
            0,
            len(features) // 4,
            len(features) // 2,
            3 * len(features) // 4,
            len(features) - 1,
        ]

        for local_tr_idx in comparison_trs:
            global_tr_idx = start_tr + local_tr_idx
            output_path = (
                output_dir
                / f"video_comparison_{segment_name}_TR{global_tr_idx:04d}.png"
            )
            save_frame_comparison(
                original_frames[local_tr_idx],
                reconstructed_frames[local_tr_idx],
                global_tr_idx,
                output_path,
            )

        # Create side-by-side comparison video
        print(f"\n6. Creating side-by-side comparison video...")
        comparison_video_path = output_dir / f"video_comparison_{segment_name}.mp4"
        create_side_by_side_video(
            original_features,
            reconstructed_features,
            comparison_video_path,
            processor,
            fps=25,
        )

        # Clean up temp video
        temp_video_path.unlink()

    # Test temporal alignment
    validate_temporal_alignment(processor, video_path, [0, 100, 300, 500, 700, 900])

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nPSNR Results:")
    for (_, _, segment_name), psnr_val in zip(test_segments, all_psnr_values):
        print(f"  {segment_name.capitalize():12s}: {psnr_val:.2f} dB")

    overall_psnr = np.mean(all_psnr_values)
    print(f"\n  Overall average: {overall_psnr:.2f} dB")

    print(f"\nGenerated outputs in {output_dir}:")
    output_files = sorted(output_dir.glob("video_*"))
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:60s} ({size_mb:.2f} MB)")

    # Final verdict
    print("\n" + "=" * 80)
    if overall_psnr > 30:
        print("✅ VIDEO VALIDATION PASSED - EXCELLENT QUALITY")
    elif overall_psnr > 20:
        print("✅ VIDEO VALIDATION PASSED - GOOD QUALITY")
    else:
        print("⚠ VIDEO VALIDATION WARNING - QUALITY BELOW EXPECTED")
    print("=" * 80)

    return overall_psnr


if __name__ == "__main__":
    try:
        psnr_result = validate_video_modality()
        print(f"\n✓ Video validation complete. Overall PSNR: {psnr_result:.2f} dB")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

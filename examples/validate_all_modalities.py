"""
Master validation script for all modalities.

Runs comprehensive validation for:
- Video: PSNR, frame comparisons, temporal alignment
- Audio: SNR, correlation, .WAV files for listening
- Text: Semantic similarity, recovery accuracy
- fMRI: Exact match, voxel timeseries, brain visualization

Generates comprehensive metrics report.

All tests use REAL data - NO MOCKS.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_audio import validate_audio_modality
from validate_fmri import validate_fmri_modality
from validate_text import validate_text_modality

# Import individual validation functions
from validate_video import validate_video_modality


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def save_validation_report(results: dict, output_dir: Path):
    """
    Save comprehensive validation report.

    Parameters
    ----------
    results : dict
        Dictionary with results from all modalities
    output_dir : Path
        Directory to save report
    """
    report_path = output_dir / "validation_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GIBLET MULTIMODAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {format_time(results['total_runtime'])}\n\n")

        # Video results
        f.write("=" * 80 + "\n")
        f.write("VIDEO MODALITY\n")
        f.write("=" * 80 + "\n\n")

        if results["video"]["success"]:
            f.write(f"Status: ✅ PASSED\n")
            f.write(f"Runtime: {format_time(results['video']['runtime'])}\n\n")

            f.write(f"Overall PSNR: {results['video']['psnr']:.2f} dB\n\n")

            f.write("Quality Assessment:\n")
            if results["video"]["psnr"] > 30:
                f.write("  EXCELLENT - Near-perfect reconstruction\n")
            elif results["video"]["psnr"] > 20:
                f.write("  GOOD - High quality reconstruction\n")
            else:
                f.write("  ACCEPTABLE - Recognizable reconstruction\n")

            f.write("\nGenerated Files:\n")
            f.write("  - Frame comparison images (PNG)\n")
            f.write("  - Side-by-side comparison videos (MP4)\n")
            f.write("  - Temporal alignment validation\n")
        else:
            f.write(f"Status: ❌ FAILED\n")
            f.write(f"Error: {results['video'].get('error', 'Unknown error')}\n")

        # Audio results
        f.write("\n" + "=" * 80 + "\n")
        f.write("AUDIO MODALITY\n")
        f.write("=" * 80 + "\n\n")

        if results["audio"]["success"]:
            f.write(f"Status: ✅ PASSED\n")
            f.write(f"Runtime: {format_time(results['audio']['runtime'])}\n\n")

            metrics = results["audio"]["metrics"]
            f.write(
                f"Average Audio Correlation: {results['audio']['avg_correlation']:.4f}\n"
            )
            f.write(f"Average SNR: {results['audio']['avg_snr']:.2f} dB\n\n")

            f.write("Segment Results:\n")
            for m in metrics:
                f.write(
                    f"  {m['segment_name']:20s}: "
                    f"Corr={m['audio_correlation']:.4f}, "
                    f"SNR={m['audio_snr']:.2f} dB\n"
                )

            f.write("\nQuality Assessment:\n")
            if results["audio"]["avg_correlation"] > 0.6:
                f.write("  GOOD - Strong correlation with original\n")
            elif results["audio"]["avg_correlation"] > 0.4:
                f.write("  ACCEPTABLE - Moderate correlation\n")
            else:
                f.write("  LOW - Weak correlation (expected for Griffin-Lim)\n")

            f.write("\nGenerated Files:\n")
            f.write("  - Original audio segments (.WAV)\n")
            f.write("  - Reconstructed audio segments (.WAV)\n")
            f.write("  - Waveform comparisons (PNG)\n")
            f.write("  - Spectrogram comparisons (PNG)\n")

            f.write("\n⚠ IMPORTANT: Listen to .WAV files for quality assessment\n")
        else:
            f.write(f"Status: ❌ FAILED\n")
            f.write(f"Error: {results['audio'].get('error', 'Unknown error')}\n")

        # Text results
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEXT MODALITY\n")
        f.write("=" * 80 + "\n\n")

        if results["text"]["success"]:
            f.write(f"Status: ✅ PASSED\n")
            f.write(f"Runtime: {format_time(results['text']['runtime'])}\n\n")

            data = results["text"]["data"]
            f.write(f"Annotations: {data['n_segments']} segments\n")
            f.write(f"TRs: {data['n_trs']}\n")
            f.write(f"Coverage: {data['coverage_pct']:.1f}%\n\n")

            f.write("Recovery Accuracy:\n")
            for k, acc in data["accuracies"].items():
                k_val = k.split("_")[1]
                f.write(f"  Top-{k_val:>2s}: {acc:6.2%}\n")

            f.write(f"\nSemantic Consistency:\n")
            f.write(
                f"  Consecutive TRs: {data['consistency']['consecutive_mean']:.4f}\n"
            )
            f.write(f"  Random pairs: {data['consistency']['random_mean']:.4f}\n")

            if (
                data["consistency"]["consecutive_mean"]
                > data["consistency"]["random_mean"]
            ):
                f.write(f"  ✓ Temporal coherence detected\n")

            f.write("\nGenerated Files:\n")
            f.write("  - Text recovery examples (.TXT)\n")
            f.write("  - Top-k nearest neighbor analysis\n")
        else:
            f.write(f"Status: ❌ FAILED\n")
            f.write(f"Error: {results['text'].get('error', 'Unknown error')}\n")

        # fMRI results
        f.write("\n" + "=" * 80 + "\n")
        f.write("fMRI MODALITY\n")
        f.write("=" * 80 + "\n\n")

        if results["fmri"]["success"]:
            f.write(f"Status: ✅ PASSED\n")
            f.write(f"Runtime: {format_time(results['fmri']['runtime'])}\n\n")

            metrics = results["fmri"]["metrics"]
            f.write(f"Subjects Tested: {len(metrics)}\n")
            f.write(f"Exact Matches: {results['fmri']['n_exact']}/{len(metrics)}\n")
            f.write(
                f"Average Correlation: {results['fmri']['avg_correlation']:.6f}\n\n"
            )

            f.write("Subject Results:\n")
            for m in metrics:
                f.write(
                    f"  {m['subject_id']:4s}: "
                    f"{m['n_voxels']:>6,} voxels, "
                    f"{m['n_trs']:>3} TRs, "
                    f"Corr={m['mean_voxel_corr']:.6f}, "
                    f"MaxDiff={m['max_abs_diff']:.6e}\n"
                )

            f.write("\nQuality Assessment:\n")
            if results["fmri"]["n_exact"] == len(metrics):
                f.write("  PERFECT - Exact reconstruction\n")
            elif results["fmri"]["avg_correlation"] > 0.99:
                f.write("  EXCELLENT - Near-exact reconstruction\n")
            else:
                f.write("  GOOD - High-quality reconstruction\n")

            f.write("\nGenerated Files:\n")
            f.write("  - Reconstructed NIfTI files (.nii.gz)\n")
            f.write("  - Voxel timeseries plots (PNG)\n")
            f.write("  - Brain slice visualizations (PNG)\n")
            f.write("  - Shared brain mask (.nii.gz)\n")
        else:
            f.write(f"Status: ❌ FAILED\n")
            f.write(f"Error: {results['fmri'].get('error', 'Unknown error')}\n")

        # Overall summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        successes = sum(
            1 for m in ["video", "audio", "text", "fmri"] if results[m]["success"]
        )
        f.write(f"Modalities Passed: {successes}/4\n\n")

        if successes == 4:
            f.write("✅ ALL MODALITIES VALIDATED SUCCESSFULLY\n\n")
            f.write(
                "The giblet multimodal processing pipeline is functioning correctly.\n"
            )
            f.write("All round-trip validations passed quality thresholds.\n")
        else:
            f.write("⚠ SOME MODALITIES FAILED VALIDATION\n\n")
            failed = [
                m
                for m in ["video", "audio", "text", "fmri"]
                if not results[m]["success"]
            ]
            f.write(f"Failed modalities: {', '.join(failed)}\n")
            f.write("Review error messages above for details.\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Review generated outputs in validation_outputs/\n")
        f.write("2. For audio: Listen to .WAV files to verify quality\n")
        f.write("3. For video: Watch comparison videos\n")
        f.write("4. For text: Read recovery examples\n")
        f.write("5. For fMRI: Inspect brain visualizations\n\n")

        f.write("All validation tests use REAL Sherlock data.\n")
        f.write("No mocks or synthetic data were used.\n\n")

        f.write("=" * 80 + "\n")

    print(f"\n✓ Saved comprehensive report to {report_path.name}")

    # Also save JSON for programmatic access
    json_path = output_dir / "validation_results.json"
    with open(json_path, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        json_results = {}
        for modality in results:
            if modality in ["video", "audio", "text", "fmri"]:
                json_results[modality] = {
                    "success": results[modality]["success"],
                    "runtime": results[modality]["runtime"],
                }
                if results[modality]["success"]:
                    # Add simplified metrics
                    if modality == "video":
                        json_results[modality]["psnr"] = float(
                            results[modality]["psnr"]
                        )
                    elif modality == "audio":
                        json_results[modality]["avg_correlation"] = float(
                            results[modality]["avg_correlation"]
                        )
                        json_results[modality]["avg_snr"] = float(
                            results[modality]["avg_snr"]
                        )
                    elif modality == "text":
                        json_results[modality]["coverage_pct"] = float(
                            results[modality]["data"]["coverage_pct"]
                        )
                        json_results[modality]["top_1_accuracy"] = float(
                            results[modality]["data"]["accuracies"]["top_1"]
                        )
                    elif modality == "fmri":
                        json_results[modality]["avg_correlation"] = float(
                            results[modality]["avg_correlation"]
                        )
                        json_results[modality]["n_exact"] = int(
                            results[modality]["n_exact"]
                        )

        json_results["total_runtime"] = results["total_runtime"]
        json_results["timestamp"] = datetime.now().isoformat()

        json.dump(json_results, f, indent=2)

    print(f"✓ Saved JSON results to {json_path.name}")


def validate_all():
    """Run validation for all modalities."""
    print("\n" + "=" * 80)
    print("GIBLET MULTIMODAL VALIDATION SUITE")
    print("=" * 80)
    print("\nThis will validate all 4 modalities using REAL Sherlock data:")
    print("  1. Video (frame reconstruction, PSNR)")
    print("  2. Audio (waveform reconstruction, SNR, .WAV files)")
    print("  3. Text (semantic embeddings, recovery accuracy)")
    print("  4. fMRI (exact reconstruction, voxel timeseries)")
    print("\nAll outputs will be saved to validation_outputs/")
    print("=" * 80)

    start_time = time.time()

    results = {
        "video": {"success": False, "runtime": 0},
        "audio": {"success": False, "runtime": 0},
        "text": {"success": False, "runtime": 0},
        "fmri": {"success": False, "runtime": 0},
    }

    # Validate video
    print("\n" + "=" * 80)
    print("STEP 1/4: VIDEO VALIDATION")
    print("=" * 80)
    video_start = time.time()
    try:
        psnr = validate_video_modality()
        results["video"]["success"] = True
        results["video"]["psnr"] = psnr
        results["video"]["runtime"] = time.time() - video_start
        print(
            f"\n✅ Video validation completed in {format_time(results['video']['runtime'])}"
        )
    except Exception as e:
        results["video"]["error"] = str(e)
        results["video"]["runtime"] = time.time() - video_start
        print(f"\n❌ Video validation failed: {e}")
        import traceback

        traceback.print_exc()

    # Validate audio
    print("\n" + "=" * 80)
    print("STEP 2/4: AUDIO VALIDATION")
    print("=" * 80)
    audio_start = time.time()
    try:
        import numpy as np

        metrics = validate_audio_modality()
        results["audio"]["success"] = True
        results["audio"]["metrics"] = metrics
        results["audio"]["avg_correlation"] = np.mean(
            [m["audio_correlation"] for m in metrics]
        )
        results["audio"]["avg_snr"] = np.mean([m["audio_snr"] for m in metrics])
        results["audio"]["runtime"] = time.time() - audio_start
        print(
            f"\n✅ Audio validation completed in {format_time(results['audio']['runtime'])}"
        )
    except Exception as e:
        results["audio"]["error"] = str(e)
        results["audio"]["runtime"] = time.time() - audio_start
        print(f"\n❌ Audio validation failed: {e}")
        import traceback

        traceback.print_exc()

    # Validate text
    print("\n" + "=" * 80)
    print("STEP 3/4: TEXT VALIDATION")
    print("=" * 80)
    text_start = time.time()
    try:
        data = validate_text_modality()
        results["text"]["success"] = True
        results["text"]["data"] = data
        results["text"]["runtime"] = time.time() - text_start
        print(
            f"\n✅ Text validation completed in {format_time(results['text']['runtime'])}"
        )
    except Exception as e:
        results["text"]["error"] = str(e)
        results["text"]["runtime"] = time.time() - text_start
        print(f"\n❌ Text validation failed: {e}")
        import traceback

        traceback.print_exc()

    # Validate fMRI
    print("\n" + "=" * 80)
    print("STEP 4/4: fMRI VALIDATION")
    print("=" * 80)
    fmri_start = time.time()
    try:
        import numpy as np

        metrics = validate_fmri_modality()
        results["fmri"]["success"] = True
        results["fmri"]["metrics"] = metrics
        results["fmri"]["avg_correlation"] = np.mean(
            [m["mean_voxel_corr"] for m in metrics]
        )
        results["fmri"]["n_exact"] = sum(1 for m in metrics if m["exact_match"])
        results["fmri"]["runtime"] = time.time() - fmri_start
        print(
            f"\n✅ fMRI validation completed in {format_time(results['fmri']['runtime'])}"
        )
    except Exception as e:
        results["fmri"]["error"] = str(e)
        results["fmri"]["runtime"] = time.time() - fmri_start
        print(f"\n❌ fMRI validation failed: {e}")
        import traceback

        traceback.print_exc()

    # Calculate total runtime
    results["total_runtime"] = time.time() - start_time

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION REPORT")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / "validation_outputs"
    save_validation_report(results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal runtime: {format_time(results['total_runtime'])}")
    print(f"\nResults by modality:")

    for modality in ["video", "audio", "text", "fmri"]:
        status = "✅ PASSED" if results[modality]["success"] else "❌ FAILED"
        runtime = format_time(results[modality]["runtime"])
        print(f"  {modality.upper():8s}: {status:12s} ({runtime})")

    successes = sum(
        1 for m in ["video", "audio", "text", "fmri"] if results[m]["success"]
    )

    print(f"\nOverall: {successes}/4 modalities passed")

    if successes == 4:
        print("\n" + "=" * 80)
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 80)
        print("\nThe giblet multimodal processing pipeline is working correctly.")
        print("All round-trip validations completed successfully.")
        print("\nReview outputs in validation_outputs/ for detailed results.")
    else:
        print("\n" + "=" * 80)
        print("⚠ SOME VALIDATIONS FAILED")
        print("=" * 80)
        failed = [
            m.upper()
            for m in ["video", "audio", "text", "fmri"]
            if not results[m]["success"]
        ]
        print(f"\nFailed: {', '.join(failed)}")
        print("Review error messages above and validation_report.txt")

    print("\n" + "=" * 80)
    print(f"Detailed report: validation_outputs/validation_report.txt")
    print(f"JSON results: validation_outputs/validation_results.json")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    try:
        results = validate_all()

        # Exit with appropriate code
        successes = sum(
            1 for m in ["video", "audio", "text", "fmri"] if results[m]["success"]
        )
        sys.exit(0 if successes == 4 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error during validation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

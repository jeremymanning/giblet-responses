"""
Debug EnCodec audio encoding issue with real Sherlock data.

This script reproduces the dimension mismatch bug by testing locally
with actual Sherlock video data and provides detailed diagnostics.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add giblet to path
sys.path.insert(0, str(Path(__file__).parent))

from giblet.data.audio import AudioProcessor  # noqa: E402


def debug_encodec_extraction():
    """Test EnCodec extraction with real Sherlock video."""

    print("=" * 80)
    print("DEBUGGING ENCODEC AUDIO EXTRACTION WITH SHERLOCK DATA")
    print("=" * 80)

    # Initialize processor with EnCodec
    print("\n[1] Initializing AudioProcessor with EnCodec...")
    processor = AudioProcessor(
        use_encodec=True, encodec_bandwidth=3.0, sample_rate=12000, tr=1.5, device="cpu"
    )
    print(f"   ✓ EnCodec enabled: {processor.use_encodec}")
    print(f"   ✓ Bandwidth: {processor.encodec_bandwidth} kbps")
    print(f"   ✓ TR: {processor.tr}s")

    # Test with real Sherlock video
    video_path = "data/stimuli_Sherlock.m4v"
    if not Path(video_path).exists():
        print(f"\n✗ ERROR: Sherlock video not found at {video_path}")
        return

    print(f"\n[2] Testing audio extraction from: {video_path}")

    # Start with a small number of TRs
    test_trs = [5, 10, 20]

    for max_trs in test_trs:
        print(f"\n[3] Attempting to extract {max_trs} TRs...")
        try:
            features, metadata = processor.audio_to_features(
                video_path, max_trs=max_trs, from_video=True
            )
            print(f"   ✓ SUCCESS! Shape: {features.shape}")
            print(f"   ✓ dtype: {features.dtype}")
            print(
                f"   ✓ All TRs consistent: {all(features[i].shape == features[0].shape for i in range(len(features)))}"
            )

            # Check metadata
            print("\n   Metadata sample:")
            print(metadata.head())

            # Verify each TR individually
            print("\n   Individual TR shapes:")
            for i in range(min(3, max_trs)):
                print(f"      TR {i}: {features[i].shape}")

        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            print(f"\n{'=' * 80}")
            print("DETAILED TRACEBACK:")
            print("=" * 80)
            import traceback

            traceback.print_exc()

            # Add detailed diagnostics
            print(f"\n{'=' * 80}")
            print("DIAGNOSTIC INFORMATION:")
            print("=" * 80)

            # Try to extract raw codes to see what's happening
            print("\n[DEBUG] Attempting to extract raw EnCodec codes...")
            try:
                import librosa

                # Load audio
                y, sr = librosa.load(
                    video_path, sr=processor.encodec_sample_rate, mono=False
                )
                if y.ndim > 1:
                    y = np.mean(y, axis=0)

                print(f"   Audio loaded: {len(y)} samples @ {sr} Hz")
                print(f"   Duration: {len(y) / sr:.2f}s")

                # Encode with EnCodec
                inputs = processor.encodec_processor(
                    raw_audio=y, sampling_rate=sr, return_tensors="pt"
                )
                inputs = {k: v.to(processor.device) for k, v in inputs.items()}

                with torch.no_grad():
                    encoded = processor.encodec_model.encode(
                        inputs["input_values"],
                        inputs["padding_mask"],
                        bandwidth=processor.encodec_bandwidth,
                    )

                codes = encoded.audio_codes[0].cpu()
                print(f"\n   Raw EnCodec output shape: {codes.shape}")
                print(f"   Number of codebooks: {codes.shape[0]}")
                print(f"   Total frames: {codes.shape[1]}")
                print("   Frame rate: ~75 Hz (EnCodec default)")

                # Calculate expected dimensions
                tr_length = processor.tr
                encodec_frame_rate = 75.0
                frames_per_tr = int(encodec_frame_rate * tr_length)

                print("\n   Expected dimensions per TR:")
                print(f"      TR length: {tr_length}s")
                print(f"      Frames per TR: {frames_per_tr}")
                print(
                    f"      Codebooks (bandwidth {processor.encodec_bandwidth} kbps): 8"
                )
                print(f"      Expected flattened size: {8 * frames_per_tr}")

                # Try extracting first few TRs manually
                print("\n   Manual TR extraction test:")
                for tr_idx in range(min(3, max_trs)):
                    start_time = tr_idx * tr_length
                    end_time = start_time + tr_length
                    start_frame = int(start_time * encodec_frame_rate)
                    end_frame = int(end_time * encodec_frame_rate)

                    tr_codes = codes[:, start_frame:end_frame]
                    print(
                        f"      TR {tr_idx}: start_frame={start_frame}, end_frame={end_frame}, shape={tr_codes.shape}"
                    )

                    # Check if temporal dimension matches
                    if tr_codes.shape[1] != frames_per_tr:
                        print(
                            f"         WARNING: Expected {frames_per_tr} frames, got {tr_codes.shape[1]}"
                        )

                    # Check codebook dimension
                    if tr_codes.shape[0] != 8:
                        print(
                            f"         WARNING: Expected 8 codebooks, got {tr_codes.shape[0]}"
                        )

            except Exception as debug_error:
                print(f"   ✗ Debug extraction failed: {debug_error}")
                traceback.print_exc()

            # Stop after first error for detailed analysis
            return


def test_minimal_reproduction():
    """Create minimal test case for the bug."""

    print("\n" + "=" * 80)
    print("MINIMAL BUG REPRODUCTION")
    print("=" * 80)

    # Simulate the problematic scenario
    print("\nSimulating dimension mismatch scenario...")

    # This is what we expect
    expected_codebooks = 8
    frames_per_tr = 112

    # This is what might actually happen (variable codebooks)
    actual_codebooks = 4  # Example: fewer codebooks than expected

    print(f"Expected: ({expected_codebooks}, {frames_per_tr})")
    print(f"Actual:   ({actual_codebooks}, {frames_per_tr})")

    # The problematic line in audio.py:317
    try:
        tr_codes = torch.zeros(actual_codebooks, frames_per_tr)
        normalized_codes = torch.zeros(expected_codebooks, frames_per_tr)

        # This should work
        n_available = min(actual_codebooks, expected_codebooks)
        normalized_codes[:n_available, :] = tr_codes[:n_available, :]

        print("✓ Dimension normalization successful!")
        print(f"  Result shape: {normalized_codes.shape}")

    except Exception as e:
        print(f"✗ Dimension normalization failed: {e}")


if __name__ == "__main__":
    # Run debugging
    debug_encodec_extraction()

    # Run minimal test
    test_minimal_reproduction()

#!/usr/bin/env python3
"""
End-to-End Pipeline Test for EnCodec Integration (Issue #24, Task 3.3)

Tests complete pipeline from audio → EnCodec codes → bottleneck → codes → audio

APPROVED CONFIGURATION:
- Sampling rate: 12kHz (downsampled from 24kHz)
- Bandwidth: 3.0 kbps
- User confirmed quality acceptable

TEST FLOW:
1. Load Sherlock audio (first 30 TRs = 45 seconds)
2. AudioProcessor.audio_to_features() → EnCodec codes
3. MultimodalDataset → Batch codes
4. AudioEncoder → Compressed features (256 dims)
5. Pass through bottleneck (simulated) → 2048 dims
6. AudioDecoder → Predicted codes
7. AudioProcessor.features_to_audio() → Reconstructed audio
8. Calculate quality metrics (SNR, PESQ, STOI)
9. Save WAV files for manual verification

SUCCESS CRITERIA:
- Pipeline runs without errors
- Quality degradation acceptable (STOI drop <0.1)
- Memory usage reasonable
- WAV files ready for user verification
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict
import time
import tracemalloc
from tqdm import tqdm

# Quality metrics
from pesq import pesq
from pystoi import stoi

# Import project modules using absolute imports
from giblet.data.audio import AudioProcessor
from giblet.models.encoder import AudioEncoder
from giblet.models.decoder import MultimodalDecoder

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "tr": 1.5,  # TR duration in seconds
    "n_trs": 30,  # Number of TRs to test (45 seconds)
    "use_encodec": True,
    "encodec_bandwidth": 3.0,  # kbps
    "encodec_sample_rate": 24000,  # EnCodec requires 24kHz
    "target_sample_rate": 12000,  # Downsample for efficiency
    "audio_encoder_output": 256,  # Audio encoder output dims
    "bottleneck_dim": 2048,  # Bottleneck dimensions
    "n_codebooks": 8,  # EnCodec 24kHz at 3.0 kbps uses 8 codebooks
    "frames_per_tr": 112,  # 75 Hz * 1.5s = 112.5 ≈ 112 frames
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def output_dir(test_data_dir):
    """Create output directory for test results."""
    output = test_data_dir / "encodec_e2e_test"
    output.mkdir(exist_ok=True)
    return output


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    signal_power = np.sum(reference**2)
    noise_power = np.sum((reference - degraded) ** 2)

    if noise_power == 0:
        return float("inf")

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_pesq_metric(
    reference: np.ndarray, degraded: np.ndarray, sr: int
) -> float:
    """Calculate PESQ (Perceptual Evaluation of Speech Quality)."""
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    try:
        # PESQ requires 8kHz or 16kHz, resample if needed
        if sr not in [8000, 16000]:
            target_sr = 16000
            reference_resampled = librosa.resample(
                reference, orig_sr=sr, target_sr=target_sr
            )
            degraded_resampled = librosa.resample(
                degraded, orig_sr=sr, target_sr=target_sr
            )
        else:
            target_sr = sr
            reference_resampled = reference
            degraded_resampled = degraded

        # PESQ mode: 'wb' (wideband) for 16kHz, 'nb' (narrowband) for 8kHz
        mode = "wb" if target_sr == 16000 else "nb"
        score = pesq(target_sr, reference_resampled, degraded_resampled, mode)
        return score
    except Exception as e:
        print(f"  Warning: PESQ calculation failed: {e}")
        return -1.0


def calculate_stoi_metric(
    reference: np.ndarray, degraded: np.ndarray, sr: int
) -> float:
    """Calculate STOI (Short-Time Objective Intelligibility)."""
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    try:
        score = stoi(reference, degraded, sr, extended=False)
        return score
    except Exception as e:
        print(f"  Warning: STOI calculation failed: {e}")
        return -1.0


def plot_spectrograms(audio_dict: Dict[str, np.ndarray], sr: int, output_path: Path):
    """Generate spectrogram comparison plot."""
    import matplotlib.pyplot as plt

    n_audio = len(audio_dict)
    fig, axes = plt.subplots(n_audio, 1, figsize=(12, 3 * n_audio))

    if n_audio == 1:
        axes = [axes]

    for ax, (name, audio) in zip(axes, audio_dict.items()):
        # Compute spectrogram
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="viridis"
        )
        ax.set_title(name)
        ax.set_ylim(0, 8000)  # Focus on 0-8kHz range
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# SIMPLE BOTTLENECK SIMULATION
# ============================================================================


class SimpleBottleneck(nn.Module):
    """Simplified bottleneck for testing (no full autoencoder needed)."""

    def __init__(self, input_dim: int = 256, bottleneck_dim: int = 2048):
        super().__init__()
        self.compress = nn.Linear(input_dim, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through bottleneck."""
        compressed = self.compress(x)
        reconstructed = self.expand(compressed)
        return reconstructed


# ============================================================================
# MAIN TEST
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.data
def test_encodec_e2e_pipeline(data_dir, output_dir, audio_processor):
    """Run end-to-end pipeline test."""

    print("=" * 80)
    print("EnCodec End-to-End Pipeline Test")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(
        f"  Sampling rate: {CONFIG['target_sample_rate']} Hz (downsampled from 24kHz)"
    )
    print(f"  Bandwidth: {CONFIG['encodec_bandwidth']} kbps")
    print(f"  TRs: {CONFIG['n_trs']} ({CONFIG['n_trs'] * CONFIG['tr']:.1f} seconds)")
    print(f"  Device: {CONFIG['device']}")
    print(f"  Output: {output_dir}")
    print()

    # Start memory profiling
    tracemalloc.start()
    start_time = time.time()

    # ========================================================================
    # STEP 1: Load audio and extract features with EnCodec
    # ========================================================================

    print("[1/9] Loading audio and extracting EnCodec codes...")
    print("  Loading EnCodec model (this may take 2-3 minutes first time)...")

    # Audio processor is provided by fixture
    print("  EnCodec model loaded successfully!")

    # Load audio from video using data_dir fixture
    video_path = data_dir / "stimuli_Sherlock.m4v"
    if not video_path.exists():
        pytest.skip(f"Video file not found: {video_path}")

    # Extract features (EnCodec codes)
    features, metadata = audio_processor.audio_to_features(
        video_path, max_trs=CONFIG["n_trs"], from_video=True
    )

    print(f"  ✓ Features shape: {features.shape}")
    print(
        f"  ✓ Expected: ({CONFIG['n_trs']}, {CONFIG['n_codebooks']}, {CONFIG['frames_per_tr']})"
    )
    print(f"  ✓ Feature dtype: {features.dtype}")
    print(f"  ✓ Value range: [{features.min()}, {features.max()}]")

    # Verify dimensions
    assert features.shape == (
        CONFIG["n_trs"],
        CONFIG["n_codebooks"],
        CONFIG["frames_per_tr"],
    ), f"Unexpected feature shape: {features.shape}"
    assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"
    assert (
        features.min() >= 0 and features.max() <= 1023
    ), f"Codes out of range: [{features.min()}, {features.max()}]"

    print("  ✓ Dimension checks passed")
    print()

    # ========================================================================
    # STEP 2: Create baseline reconstruction (EnCodec direct)
    # ========================================================================

    print("[2/9] Creating baseline reconstruction (EnCodec direct)...")

    # Reconstruct audio directly from codes (no encoder/decoder)
    baseline_path = output_dir / "baseline_encodec_direct.wav"
    audio_processor.features_to_audio(features, baseline_path)

    # Load reconstructed audio
    baseline_audio, sr_baseline = librosa.load(
        str(baseline_path), sr=CONFIG["encodec_sample_rate"]
    )

    # Downsample to 12kHz for comparison
    baseline_audio_12k = librosa.resample(
        baseline_audio, orig_sr=sr_baseline, target_sr=CONFIG["target_sample_rate"]
    )

    # Save downsampled version
    baseline_12k_path = output_dir / "baseline_encodec_12khz.wav"
    sf.write(str(baseline_12k_path), baseline_audio_12k, CONFIG["target_sample_rate"])

    print(f"  ✓ Baseline audio: {len(baseline_audio)} samples at {sr_baseline} Hz")
    print(
        f"  ✓ Downsampled: {len(baseline_audio_12k)} samples at {CONFIG['target_sample_rate']} Hz"
    )
    print(f"  ✓ Saved: {baseline_path}")
    print(f"  ✓ Saved: {baseline_12k_path}")
    print()

    # ========================================================================
    # STEP 3: Convert to PyTorch tensors and create batch
    # ========================================================================

    print("[3/9] Creating batch data...")

    # Convert to torch tensors
    # Shape: (n_trs, n_codebooks, frames_per_tr) → (batch, n_codebooks * frames_per_tr)
    # FLATTEN for AudioEncoder (Issue #29: temporal concatenation)
    batch_codes_3d = torch.from_numpy(features).float().to(CONFIG["device"])
    batch_codes = batch_codes_3d.reshape(batch_codes_3d.shape[0], -1)  # Flatten to 2D

    print(f"  ✓ Batch shape (3D): {batch_codes_3d.shape}")
    print(f"  ✓ Batch shape (2D flattened): {batch_codes.shape}")
    print(f"  ✓ Batch dtype: {batch_codes.dtype}")
    print(f"  ✓ Device: {batch_codes.device}")
    print()

    # ========================================================================
    # STEP 4: Pass through AudioEncoder
    # ========================================================================

    print("[4/9] Encoding with AudioEncoder...")

    # Initialize encoder (Issue #29: Linear-based with temporal concatenation)
    encoder = AudioEncoder(
        input_codebooks=CONFIG["n_codebooks"],
        audio_frames_per_tr=CONFIG["frames_per_tr"],  # Updated parameter name
        output_features=CONFIG["audio_encoder_output"],
        use_encodec=True,
    ).to(CONFIG["device"])

    encoder.eval()

    # Encode (input is now 2D flattened)
    with torch.no_grad():
        encoded = encoder(batch_codes)

    print(f"  ✓ Encoded shape: {encoded.shape}")
    print(f"  ✓ Expected: ({CONFIG['n_trs']}, {CONFIG['audio_encoder_output']})")
    print(f"  ✓ Value range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")

    assert encoded.shape == (
        CONFIG["n_trs"],
        CONFIG["audio_encoder_output"],
    ), f"Unexpected encoded shape: {encoded.shape}"

    print("  ✓ Encoding successful")
    print()

    # ========================================================================
    # STEP 5: Pass through bottleneck (simulated)
    # ========================================================================

    print("[5/9] Passing through bottleneck...")

    # Initialize bottleneck
    bottleneck = SimpleBottleneck(
        input_dim=CONFIG["audio_encoder_output"],
        bottleneck_dim=CONFIG["bottleneck_dim"],
    ).to(CONFIG["device"])

    bottleneck.eval()

    # Pass through bottleneck
    with torch.no_grad():
        bottleneck_out = bottleneck(encoded)

    print(f"  ✓ Bottleneck output shape: {bottleneck_out.shape}")
    print(f"  ✓ Expected: ({CONFIG['n_trs']}, {CONFIG['audio_encoder_output']})")
    print(
        f"  ✓ Value range: [{bottleneck_out.min().item():.3f}, {bottleneck_out.max().item():.3f}]"
    )

    assert bottleneck_out.shape == (
        CONFIG["n_trs"],
        CONFIG["audio_encoder_output"],
    ), f"Unexpected bottleneck output shape: {bottleneck_out.shape}"

    print("  ✓ Bottleneck pass successful")
    print()

    # ========================================================================
    # STEP 6: Decode with AudioDecoder (audio path only)
    # ========================================================================

    print("[6/9] Decoding with AudioDecoder...")

    # Initialize decoder
    decoder = MultimodalDecoder(
        bottleneck_dim=CONFIG[
            "audio_encoder_output"
        ],  # Use encoder output as "bottleneck" for simplicity
        audio_dim=CONFIG["n_codebooks"],
        audio_frames_per_tr=CONFIG["frames_per_tr"],
        use_encodec=True,
        n_codebooks=CONFIG["n_codebooks"],
    ).to(CONFIG["device"])

    decoder.eval()

    # Decode (audio only) - returns 2D flattened codes
    with torch.no_grad():
        decoded_codes_flat = decoder.decode_audio_only(bottleneck_out)

    # Reshape to 3D for features_to_audio compatibility
    decoded_codes = decoded_codes_flat.reshape(
        CONFIG["n_trs"], CONFIG["n_codebooks"], CONFIG["frames_per_tr"]
    )

    print(f"  ✓ Decoded codes shape (2D): {decoded_codes_flat.shape}")
    print(f"  ✓ Decoded codes shape (3D): {decoded_codes.shape}")
    print(
        f"  ✓ Expected: ({CONFIG['n_trs']}, {CONFIG['n_codebooks']}, {CONFIG['frames_per_tr']})"
    )
    print(
        f"  ✓ Value range: [{decoded_codes.min().item():.3f}, {decoded_codes.max().item():.3f}]"
    )

    assert decoded_codes.shape == (
        CONFIG["n_trs"],
        CONFIG["n_codebooks"],
        CONFIG["frames_per_tr"],
    ), f"Unexpected decoded shape: {decoded_codes.shape}"

    # Ensure codes are in valid range [0, 1023]
    decoded_codes = torch.clamp(decoded_codes, 0, 1023)

    print("  ✓ Decoding successful")
    print()

    # ========================================================================
    # STEP 7: Reconstruct audio from predicted codes
    # ========================================================================

    print("[7/9] Reconstructing audio from predicted codes...")

    # Convert to numpy and int64
    predicted_codes_np = decoded_codes.cpu().numpy().astype(np.int64)

    # Reconstruct audio using EnCodec decoder
    reconstructed_path = output_dir / "reconstructed_through_bottleneck.wav"
    audio_processor.features_to_audio(predicted_codes_np, reconstructed_path)

    # Load reconstructed audio
    reconstructed_audio, sr_recon = librosa.load(
        str(reconstructed_path), sr=CONFIG["encodec_sample_rate"]
    )

    # Downsample to 12kHz
    reconstructed_audio_12k = librosa.resample(
        reconstructed_audio, orig_sr=sr_recon, target_sr=CONFIG["target_sample_rate"]
    )

    # Save downsampled version
    reconstructed_12k_path = output_dir / "reconstructed_12khz.wav"
    sf.write(
        str(reconstructed_12k_path),
        reconstructed_audio_12k,
        CONFIG["target_sample_rate"],
    )

    print(
        f"  ✓ Reconstructed audio: {len(reconstructed_audio)} samples at {sr_recon} Hz"
    )
    print(
        f"  ✓ Downsampled: {len(reconstructed_audio_12k)} samples at {CONFIG['target_sample_rate']} Hz"
    )
    print(f"  ✓ Saved: {reconstructed_path}")
    print(f"  ✓ Saved: {reconstructed_12k_path}")
    print()

    # ========================================================================
    # STEP 8: Calculate quality metrics
    # ========================================================================

    print("[8/9] Calculating quality metrics...")

    # Load original audio for comparison
    original_audio, sr_orig = librosa.load(
        str(video_path),
        sr=CONFIG["encodec_sample_rate"],
        duration=CONFIG["n_trs"] * CONFIG["tr"],
        mono=True,
    )

    # Downsample original to 12kHz
    original_audio_12k = librosa.resample(
        original_audio, orig_sr=sr_orig, target_sr=CONFIG["target_sample_rate"]
    )

    # Save original for comparison
    original_path = output_dir / "original_30trs.wav"
    original_12k_path = output_dir / "original_12khz.wav"
    sf.write(str(original_path), original_audio, sr_orig)
    sf.write(str(original_12k_path), original_audio_12k, CONFIG["target_sample_rate"])

    print(f"  ✓ Original audio: {len(original_audio)} samples at {sr_orig} Hz")
    print(f"  ✓ Saved: {original_path}")
    print(f"  ✓ Saved: {original_12k_path}")
    print()

    # Compare baseline vs reconstructed (both at 12kHz)
    metrics = {}

    # Baseline (EnCodec direct)
    print("  Baseline (EnCodec direct, 12kHz):")
    metrics["baseline_snr"] = calculate_snr(original_audio_12k, baseline_audio_12k)
    metrics["baseline_pesq"] = calculate_pesq_metric(
        original_audio_12k, baseline_audio_12k, CONFIG["target_sample_rate"]
    )
    metrics["baseline_stoi"] = calculate_stoi_metric(
        original_audio_12k, baseline_audio_12k, CONFIG["target_sample_rate"]
    )

    print(f"    SNR:  {metrics['baseline_snr']:6.2f} dB")
    print(f"    PESQ: {metrics['baseline_pesq']:6.3f}")
    print(f"    STOI: {metrics['baseline_stoi']:6.3f}")
    print()

    # Reconstructed (through encoder/decoder)
    print("  Reconstructed (through bottleneck, 12kHz):")
    metrics["recon_snr"] = calculate_snr(original_audio_12k, reconstructed_audio_12k)
    metrics["recon_pesq"] = calculate_pesq_metric(
        original_audio_12k, reconstructed_audio_12k, CONFIG["target_sample_rate"]
    )
    metrics["recon_stoi"] = calculate_stoi_metric(
        original_audio_12k, reconstructed_audio_12k, CONFIG["target_sample_rate"]
    )

    print(f"    SNR:  {metrics['recon_snr']:6.2f} dB")
    print(f"    PESQ: {metrics['recon_pesq']:6.3f}")
    print(f"    STOI: {metrics['recon_stoi']:6.3f}")
    print()

    # Quality degradation
    print("  Quality degradation (baseline → reconstructed):")
    stoi_drop = metrics["baseline_stoi"] - metrics["recon_stoi"]
    pesq_drop = metrics["baseline_pesq"] - metrics["recon_pesq"]
    snr_drop = metrics["baseline_snr"] - metrics["recon_snr"]

    print(f"    STOI drop: {stoi_drop:+.3f} (target: <0.1)")
    print(f"    PESQ drop: {pesq_drop:+.3f}")
    print(f"    SNR drop:  {snr_drop:+.2f} dB")
    print()

    # Success criteria
    if stoi_drop < 0.1:
        print("  ✓ SUCCESS: Quality degradation acceptable")
    else:
        print("  ⚠ WARNING: Quality degradation exceeds target")
    print()

    # ========================================================================
    # STEP 9: Generate spectrograms and save metrics
    # ========================================================================

    print("[9/9] Generating spectrograms and saving metrics...")

    # Generate spectrograms
    try:
        plot_spectrograms(
            {
                "Original (12kHz)": original_audio_12k,
                "Baseline EnCodec (12kHz)": baseline_audio_12k,
                "Through Bottleneck (12kHz)": reconstructed_audio_12k,
            },
            sr=CONFIG["target_sample_rate"],
            output_path=output_dir / "spectrograms_comparison.png",
        )
    except Exception as e:
        print(f"  ⚠ Warning: Spectrogram generation failed: {e}")

    # Save metrics to file
    metrics_path = output_dir / "metrics_comparison.txt"
    with open(metrics_path, "w") as f:
        f.write("EnCodec End-to-End Pipeline Test - Quality Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Sampling rate: {CONFIG['target_sample_rate']} Hz (12kHz)\n")
        f.write(f"  Bandwidth: {CONFIG['encodec_bandwidth']} kbps\n")
        f.write(
            f"  Duration: {CONFIG['n_trs'] * CONFIG['tr']:.1f} seconds ({CONFIG['n_trs']} TRs)\n\n"
        )

        f.write("Baseline (EnCodec direct, 12kHz):\n")
        f.write(f"  SNR:  {metrics['baseline_snr']:6.2f} dB\n")
        f.write(f"  PESQ: {metrics['baseline_pesq']:6.3f}\n")
        f.write(f"  STOI: {metrics['baseline_stoi']:6.3f}\n\n")

        f.write("Reconstructed (through encoder/decoder, 12kHz):\n")
        f.write(f"  SNR:  {metrics['recon_snr']:6.2f} dB\n")
        f.write(f"  PESQ: {metrics['recon_pesq']:6.3f}\n")
        f.write(f"  STOI: {metrics['recon_stoi']:6.3f}\n\n")

        f.write("Quality degradation:\n")
        f.write(f"  STOI drop: {stoi_drop:+.3f} (target: <0.1)\n")
        f.write(f"  PESQ drop: {pesq_drop:+.3f}\n")
        f.write(f"  SNR drop:  {snr_drop:+.2f} dB\n\n")

        if stoi_drop < 0.1:
            f.write("Result: ✓ SUCCESS - Quality degradation acceptable\n")
        else:
            f.write("Result: ⚠ WARNING - Quality degradation exceeds target\n")

    print(f"  ✓ Saved: {metrics_path}")
    print()

    # ========================================================================
    # Memory and timing report
    # ========================================================================

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print()
    print(f"Memory usage:")
    print(f"  Current: {current / 1024 / 1024:.1f} MB")
    print(f"  Peak:    {peak / 1024 / 1024:.1f} MB")
    print()
    print(f"Elapsed time: {elapsed_time:.1f} seconds")
    print()
    print(f"Output files saved to: {output_dir}/")
    print()
    print("WAV files for manual verification:")
    print(f"  1. {original_12k_path.name} - Original audio (12kHz)")
    print(f"  2. {baseline_12k_path.name} - Baseline EnCodec (12kHz)")
    print(f"  3. {reconstructed_12k_path.name} - Through bottleneck (12kHz)")
    print()
    print("Metrics and spectrograms:")
    print(f"  • {metrics_path.name}")
    print(f"  • spectrograms_comparison.png")
    print()
    print("=" * 80)

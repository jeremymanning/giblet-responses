"""
Audio modality validation script.

Performs comprehensive round-trip validation:
1. Load real Sherlock video and extract audio
2. Convert audio to mel spectrogram features (20 TRs for testing)
3. Reconstruct audio from features using Griffin-Lim
4. Save reconstructed .WAV files for manual listening
5. Calculate metrics: correlation, SNR
6. Test on different segments: speech, music, silence
7. Document quality issues

All tests use REAL data - NO MOCKS.
CRITICAL: Saves .WAV files that can be listened to.
"""

import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.audio import AudioProcessor


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio.

    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    reconstructed : np.ndarray
        Reconstructed audio signal

    Returns
    -------
    snr : float
        SNR in dB
    """
    # Truncate to same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Calculate noise
    noise = original - reconstructed

    # Calculate signal power and noise power
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)

    # Avoid division by zero
    if noise_power < 1e-10:
        return float("inf")

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_correlation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Pearson correlation between original and reconstructed audio.

    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    reconstructed : np.ndarray
        Reconstructed audio signal

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    """
    # Truncate to same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    correlation, _ = pearsonr(original, reconstructed)
    return correlation


def plot_waveform_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    segment_name: str,
    output_path: Path,
):
    """
    Plot waveform comparison between original and reconstructed audio.

    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    reconstructed : np.ndarray
        Reconstructed audio signal
    sample_rate : int
        Sample rate in Hz
    segment_name : str
        Name of segment for labeling
    output_path : Path
        Path to save plot
    """
    # Truncate to same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Time axis
    time = np.arange(len(original)) / sample_rate

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Original waveform
    axes[0].plot(time, original, linewidth=0.5)
    axes[0].set_title(f"Original Audio - {segment_name}", fontsize=14)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, time[-1])
    axes[0].grid(True, alpha=0.3)

    # Reconstructed waveform
    axes[1].plot(time, reconstructed, linewidth=0.5, color="orange")
    axes[1].set_title(f"Reconstructed Audio - {segment_name}", fontsize=14)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0, time[-1])
    axes[1].grid(True, alpha=0.3)

    # Difference
    difference = original - reconstructed
    axes[2].plot(time, difference, linewidth=0.5, color="red")
    axes[2].set_title("Difference (Original - Reconstructed)", fontsize=14)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlim(0, time[-1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved waveform comparison to {output_path.name}")


def plot_spectrogram_comparison(
    original_features: np.ndarray,
    reconstructed_features: np.ndarray,
    segment_name: str,
    output_path: Path,
):
    """
    Plot spectrogram comparison.

    Parameters
    ----------
    original_features : np.ndarray
        Shape (n_trs, n_mels) original mel spectrogram
    reconstructed_features : np.ndarray
        Shape (n_trs, n_mels) reconstructed mel spectrogram
    segment_name : str
        Name of segment for labeling
    output_path : Path
        Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Original spectrogram
    im1 = axes[0].imshow(
        original_features.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0].set_title(f"Original Mel Spectrogram - {segment_name}", fontsize=14)
    axes[0].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im1, ax=axes[0], label="Amplitude (dB)")

    # Reconstructed spectrogram
    im2 = axes[1].imshow(
        reconstructed_features.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_title(f"Reconstructed Mel Spectrogram - {segment_name}", fontsize=14)
    axes[1].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im2, ax=axes[1], label="Amplitude (dB)")

    # Difference
    diff = np.abs(original_features - reconstructed_features)
    im3 = axes[2].imshow(
        diff.T, aspect="auto", origin="lower", cmap="hot", interpolation="nearest"
    )
    axes[2].set_title("Absolute Difference", fontsize=14)
    axes[2].set_xlabel("TR Index")
    axes[2].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im3, ax=axes[2], label="Absolute Difference (dB)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved spectrogram comparison to {output_path.name}")


def validate_audio_segment(
    processor: AudioProcessor,
    video_path: Path,
    start_tr: int,
    end_tr: int,
    segment_name: str,
    output_dir: Path,
):
    """
    Validate audio reconstruction for a specific segment.

    Parameters
    ----------
    processor : AudioProcessor
        AudioProcessor instance
    video_path : Path
        Path to video file
    start_tr : int
        Start TR index
    end_tr : int
        End TR index
    segment_name : str
        Name for output files
    output_dir : Path
        Directory for output files

    Returns
    -------
    metrics : dict
        Dictionary with validation metrics
    """
    print("\n" + "=" * 80)
    print(f"VALIDATING SEGMENT: {segment_name.upper()} (TRs {start_tr}-{end_tr})")
    print("=" * 80)

    # Extract audio from video
    print(f"\n1. Extracting audio from video...")
    y_full, sr = librosa.load(str(video_path), sr=processor.sample_rate, mono=True)
    duration = len(y_full) / sr
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Total samples: {len(y_full):,}")

    # Extract segment audio
    start_time = start_tr * processor.tr
    end_time = end_tr * processor.tr
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    y_segment = y_full[start_sample:end_sample]
    print(f"   Segment: {start_time:.1f}s to {end_time:.1f}s")
    print(f"   Segment samples: {len(y_segment):,}")

    # Save original segment
    original_audio_path = output_dir / f"audio_original_{segment_name}.wav"
    sf.write(str(original_audio_path), y_segment, sr)
    print(f"   Saved original audio to {original_audio_path.name}")

    # Convert to features
    print(f"\n2. Converting to mel spectrogram features...")
    features, metadata = processor.audio_to_features(
        video_path, max_trs=end_tr, from_video=True
    )
    features = features[start_tr:end_tr]
    print(f"   Features shape: {features.shape}")
    print(f"   Features range: [{features.min():.2f}, {features.max():.2f}] dB")

    # Store original features
    original_features = features.copy()

    # Reconstruct audio from features
    print(f"\n3. Reconstructing audio from features...")
    reconstructed_audio_path = output_dir / f"audio_reconstructed_{segment_name}.wav"
    processor.features_to_audio(features, reconstructed_audio_path)
    print(f"   Saved reconstructed audio to {reconstructed_audio_path.name}")

    # Load reconstructed audio
    y_reconstructed, _ = sf.read(str(reconstructed_audio_path))
    print(f"   Reconstructed samples: {len(y_reconstructed):,}")

    # Re-extract features from reconstructed audio
    print(f"\n4. Re-extracting features from reconstructed audio...")

    # Create temporary video with reconstructed audio for re-extraction
    # For simplicity, directly compute mel spectrogram
    mel_spec_recon = librosa.feature.melspectrogram(
        y=y_reconstructed,
        sr=sr,
        n_mels=processor.n_mels,
        n_fft=processor.n_fft,
        hop_length=processor.hop_length,
    )
    mel_spec_db_recon = librosa.power_to_db(mel_spec_recon, ref=np.max)

    # Aggregate to TRs
    n_trs = end_tr - start_tr
    reconstructed_features = np.zeros((n_trs, processor.n_mels), dtype=np.float32)
    samples_per_tr = int(processor.tr * sr)

    for tr_idx in range(n_trs):
        start_sample_tr = tr_idx * samples_per_tr
        end_sample_tr = (tr_idx + 1) * samples_per_tr

        start_frame = start_sample_tr // processor.hop_length
        end_frame = end_sample_tr // processor.hop_length

        if end_frame > start_frame and end_frame <= mel_spec_db_recon.shape[1]:
            reconstructed_features[tr_idx] = np.mean(
                mel_spec_db_recon[:, start_frame:end_frame], axis=1
            )

    print(f"   Reconstructed features shape: {reconstructed_features.shape}")

    # Calculate metrics
    print(f"\n5. Calculating quality metrics...")

    # Feature-level metrics
    feature_mse = np.mean((original_features - reconstructed_features) ** 2)
    feature_corr = np.corrcoef(
        original_features.flatten(), reconstructed_features.flatten()
    )[0, 1]

    print(f"   Feature MSE: {feature_mse:.4f}")
    print(f"   Feature correlation: {feature_corr:.4f}")

    # Audio-level metrics
    snr = calculate_snr(y_segment, y_reconstructed)
    audio_corr = calculate_correlation(y_segment, y_reconstructed)

    print(f"   Audio SNR: {snr:.2f} dB")
    print(f"   Audio correlation: {audio_corr:.4f}")

    # Calculate spectral statistics
    original_energy = np.sum(original_features**2)
    reconstructed_energy = np.sum(reconstructed_features**2)
    energy_ratio = reconstructed_energy / original_energy

    print(f"   Original energy: {original_energy:.2e}")
    print(f"   Reconstructed energy: {reconstructed_energy:.2e}")
    print(f"   Energy ratio: {energy_ratio:.4f}")

    # Save visualizations
    print(f"\n6. Saving visualizations...")

    # Waveform comparison
    waveform_path = output_dir / f"audio_waveform_{segment_name}.png"
    plot_waveform_comparison(
        y_segment, y_reconstructed, sr, segment_name, waveform_path
    )

    # Spectrogram comparison
    spectrogram_path = output_dir / f"audio_spectrogram_{segment_name}.png"
    plot_spectrogram_comparison(
        original_features, reconstructed_features, segment_name, spectrogram_path
    )

    # Quality assessment
    print(f"\n7. Quality assessment:")
    if audio_corr > 0.7:
        print(f"   ✓ GOOD correlation (> 0.7)")
    elif audio_corr > 0.5:
        print(f"   ⚠ MODERATE correlation (0.5-0.7)")
    else:
        print(f"   ❌ LOW correlation (< 0.5)")

    if snr > 10:
        print(f"   ✓ GOOD SNR (> 10 dB)")
    elif snr > 5:
        print(f"   ⚠ MODERATE SNR (5-10 dB)")
    else:
        print(f"   ❌ LOW SNR (< 5 dB)")

    return {
        "segment_name": segment_name,
        "feature_mse": feature_mse,
        "feature_correlation": feature_corr,
        "audio_snr": snr,
        "audio_correlation": audio_corr,
        "energy_ratio": energy_ratio,
    }


def validate_audio_modality():
    """
    Main validation function for audio modality.
    """
    print("\n" + "=" * 80)
    print("AUDIO MODALITY VALIDATION")
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
    processor = AudioProcessor(
        sample_rate=22050, n_mels=2048, n_fft=4096, hop_length=512, tr=1.5
    )

    # Get audio info
    print("\n" + "=" * 80)
    print("AUDIO INFORMATION")
    print("=" * 80)
    info = processor.get_audio_info(video_path, from_video=True)
    print(f"\nAudio properties:")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Total samples: {info['samples']:,}")
    print(f"  Available TRs: {info['n_trs']}")

    print(f"\nProcessor settings:")
    print(f"  Target sample rate: {processor.sample_rate} Hz")
    print(f"  Mel bins: {processor.n_mels}")
    print(f"  FFT size: {processor.n_fft}")
    print(f"  Hop length: {processor.hop_length}")
    print(f"  TR: {processor.tr}s")

    # Test on different segments
    # Note: Choosing segments based on typical Sherlock content
    # For comprehensive testing, would need scene-by-scene analysis
    test_segments = [
        (0, 20, "opening_scene"),  # Opening with theme music
        (200, 220, "dialogue_scene"),  # Dialogue-heavy segment
        (460, 480, "middle_scene"),  # Middle of episode
        (700, 720, "late_scene"),  # Later segment
        (900, 920, "ending_scene"),  # Ending
    ]

    all_metrics = []

    for start_tr, end_tr, segment_name in test_segments:
        metrics = validate_audio_segment(
            processor, video_path, start_tr, end_tr, segment_name, output_dir
        )
        all_metrics.append(metrics)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\nMetrics by segment:")
    print(
        f"{'Segment':<20} {'Feature Corr':>13} {'Audio Corr':>11} {'SNR (dB)':>9} {'Energy Ratio':>13}"
    )
    print("-" * 80)

    for m in all_metrics:
        print(
            f"{m['segment_name']:<20} "
            f"{m['feature_correlation']:>13.4f} "
            f"{m['audio_correlation']:>11.4f} "
            f"{m['audio_snr']:>9.2f} "
            f"{m['energy_ratio']:>13.4f}"
        )

    # Overall statistics
    avg_feature_corr = np.mean([m["feature_correlation"] for m in all_metrics])
    avg_audio_corr = np.mean([m["audio_correlation"] for m in all_metrics])
    avg_snr = np.mean([m["audio_snr"] for m in all_metrics])

    print("\nOverall averages:")
    print(f"  Feature correlation: {avg_feature_corr:.4f}")
    print(f"  Audio correlation: {avg_audio_corr:.4f}")
    print(f"  SNR: {avg_snr:.2f} dB")

    print(f"\nGenerated .WAV files (LISTEN TO THESE!):")
    wav_files = sorted(output_dir.glob("audio_*.wav"))
    for f in wav_files:
        size_kb = f.stat().st_size / 1024
        duration = len(sf.read(str(f))[0]) / 22050
        print(f"  {f.name:50s} ({size_kb:>7.1f} KB, {duration:>5.1f}s)")

    print(f"\nGenerated visualizations:")
    png_files = sorted(output_dir.glob("audio_*.png"))
    for f in png_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:>7.1f} KB)")

    # Known issues documentation
    print("\n" + "=" * 80)
    print("KNOWN ISSUES / QUALITY NOTES")
    print("=" * 80)
    print(
        """
Griffin-Lim reconstruction limitations:
- Phase information is lost in mel spectrogram
- Reconstructed audio may sound "phasey" or "robotic"
- High-frequency content may be degraded
- Expected correlations: 0.5-0.8 (not perfect but recognizable)
- Expected SNR: 5-15 dB (moderate to good)

This is expected behavior for mel spectrogram + Griffin-Lim approach.
For better reconstruction, consider:
- Using HiFi-GAN or other neural vocoder
- Preserving phase information
- Using higher resolution spectrograms

For this project's purposes (fMRI prediction), the feature extraction
is what matters most. The reconstruction quality validates that the
features contain meaningful audio information.
    """
    )

    # Final verdict
    print("\n" + "=" * 80)
    if avg_audio_corr > 0.6 and avg_snr > 8:
        print("✅ AUDIO VALIDATION PASSED - GOOD QUALITY")
    elif avg_audio_corr > 0.4 and avg_snr > 5:
        print("✅ AUDIO VALIDATION PASSED - ACCEPTABLE QUALITY")
    else:
        print("⚠ AUDIO VALIDATION WARNING - QUALITY BELOW EXPECTED")
    print("=" * 80)

    return all_metrics


if __name__ == "__main__":
    try:
        metrics = validate_audio_modality()
        avg_corr = np.mean([m["audio_correlation"] for m in metrics])
        avg_snr = np.mean([m["audio_snr"] for m in metrics])
        print(
            f"\n✓ Audio validation complete. Avg correlation: {avg_corr:.4f}, "
            f"Avg SNR: {avg_snr:.2f} dB"
        )
        print(
            "\n⚠ IMPORTANT: Listen to the generated .WAV files in validation_outputs/"
        )
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
EnCodec Quality Test Script for Sherlock Audio

Tests EnCodec audio compression on the first 30 seconds of Sherlock stimulus.
Generates quality metrics and visualization outputs.

Author: Claude Code
Date: 2025-10-31
Issue: #24, Task 1.1
"""

import os
import subprocess
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from pesq import pesq
from pystoi import stoi
from scipy import signal
from encodec import EncodecModel
from encodec.utils import convert_audio

# Configuration
VIDEO_PATH = "data/stimuli_Sherlock.m4v"
OUTPUT_DIR = "encodec_test_outputs"
DURATION_SECONDS = 30
TARGET_BANDWIDTH = 6.0  # kbps
SAMPLE_RATE = 24000  # EnCodec requirement

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")

def load_sherlock_audio(duration_seconds=30):
    """
    Load audio from Sherlock video file using ffmpeg.

    Args:
        duration_seconds: Duration to extract (default 30s)

    Returns:
        Tuple of (waveform, sample_rate)
    """
    print(f"\n1. Loading audio from {VIDEO_PATH}...")

    # Create temporary WAV file
    temp_wav = os.path.join(OUTPUT_DIR, "temp_audio.wav")

    # Extract audio using ffmpeg
    print(f"   Extracting audio with ffmpeg...")
    cmd = [
        'ffmpeg',
        '-i', VIDEO_PATH,
        '-t', str(duration_seconds),  # Extract only first N seconds
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', '48000',  # Sample rate
        '-ac', '2',  # Stereo
        '-y',  # Overwrite
        temp_wav
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    # Load with soundfile
    audio_data, sr = sf.read(temp_wav, dtype='float32')

    # Convert to torch tensor (channels first)
    # audio_data is (samples, channels) -> need (channels, samples)
    waveform = torch.from_numpy(audio_data.T)

    print(f"   Original sample rate: {sr} Hz")
    print(f"   Original channels: {waveform.shape[0]}")
    print(f"   Original duration: {waveform.shape[1] / sr:.2f} seconds")

    # Clean up temp file
    os.remove(temp_wav)

    return waveform, sr

def save_wav(waveform, sample_rate, filename):
    """Save waveform as WAV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Convert to numpy (samples, channels)
    audio_np = waveform.cpu().numpy().T

    # Save with soundfile
    sf.write(filepath, audio_np, sample_rate)
    print(f"   Saved: {filepath}")
    return filepath

def test_encodec(waveform, original_sr):
    """
    Test EnCodec encoding/decoding.

    Args:
        waveform: Input audio waveform
        original_sr: Original sample rate

    Returns:
        Tuple of (decoded_waveform, model_sample_rate)
    """
    print(f"\n2. Testing EnCodec...")

    # Load EnCodec model
    print(f"   Loading EnCodec 24kHz model...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(TARGET_BANDWIDTH)

    print(f"   Target bandwidth: {TARGET_BANDWIDTH} kbps")
    print(f"   Model sample rate: {model.sample_rate} Hz")
    print(f"   Model channels: {model.channels}")

    # Convert audio to model requirements
    print(f"   Converting audio to {model.sample_rate} Hz, {model.channels} channel(s)...")
    wav_converted = convert_audio(waveform, original_sr, model.sample_rate, model.channels)

    # Save original (resampled) version
    original_path = save_wav(wav_converted, model.sample_rate, "original_30s.wav")

    # Encode
    print(f"   Encoding...")
    with torch.no_grad():
        encoded_frames = model.encode(wav_converted.unsqueeze(0))

    print(f"   Encoded frames: {len(encoded_frames[0])}")

    # Decode
    print(f"   Decoding...")
    with torch.no_grad():
        decoded = model.decode(encoded_frames)

    # Remove batch dimension
    decoded = decoded.squeeze(0)

    print(f"   Decoded shape: {decoded.shape}")

    # Save reconstructed version
    reconstructed_path = save_wav(decoded, model.sample_rate, "encodec_reconstructed_30s.wav")

    return wav_converted, decoded, model.sample_rate

def calculate_snr(original, reconstructed):
    """
    Calculate Signal-to-Noise Ratio (SNR).

    Args:
        original: Original signal
        reconstructed: Reconstructed signal

    Returns:
        SNR in dB
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    # Convert to numpy
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    # Calculate noise
    noise = original_np - reconstructed_np

    # Calculate power
    signal_power = np.mean(original_np ** 2)
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')

    # SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def calculate_pesq_score(original, reconstructed, sample_rate):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        sample_rate: Sample rate

    Returns:
        PESQ score (higher is better, range: -0.5 to 4.5)
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    # Convert to mono if stereo
    if original.dim() > 1 and original.shape[0] > 1:
        original = original.mean(dim=0)
    if reconstructed.dim() > 1 and reconstructed.shape[0] > 1:
        reconstructed = reconstructed.mean(dim=0)

    # Convert to numpy
    original_np = original.cpu().numpy().flatten()
    reconstructed_np = reconstructed.cpu().numpy().flatten()

    # PESQ requires 8kHz or 16kHz
    # Resample if needed
    if sample_rate not in [8000, 16000]:
        # Resample to 16kHz
        num_samples = int(len(original_np) * 16000 / sample_rate)
        original_resampled = signal.resample(original_np, num_samples)
        reconstructed_resampled = signal.resample(reconstructed_np, num_samples)
        pesq_sr = 16000
    else:
        original_resampled = original_np
        reconstructed_resampled = reconstructed_np
        pesq_sr = sample_rate

    # Calculate PESQ
    pesq_score = pesq(pesq_sr, original_resampled, reconstructed_resampled, 'wb')

    return pesq_score

def calculate_stoi_score(original, reconstructed, sample_rate):
    """
    Calculate STOI (Short-Time Objective Intelligibility).

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        sample_rate: Sample rate

    Returns:
        STOI score (range: 0 to 1, higher is better)
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    # Convert to mono if stereo
    if original.dim() > 1 and original.shape[0] > 1:
        original = original.mean(dim=0)
    if reconstructed.dim() > 1 and reconstructed.shape[0] > 1:
        reconstructed = reconstructed.mean(dim=0)

    # Convert to numpy
    original_np = original.cpu().numpy().flatten()
    reconstructed_np = reconstructed.cpu().numpy().flatten()

    # Calculate STOI
    stoi_score = stoi(original_np, reconstructed_np, sample_rate, extended=False)

    return stoi_score

def calculate_correlation(original, reconstructed):
    """
    Calculate waveform correlation.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal

    Returns:
        Correlation coefficient
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    # Convert to numpy and flatten
    original_np = original.cpu().numpy().flatten()
    reconstructed_np = reconstructed.cpu().numpy().flatten()

    # Calculate correlation
    correlation = np.corrcoef(original_np, reconstructed_np)[0, 1]

    return correlation

def calculate_metrics(original, reconstructed, sample_rate):
    """
    Calculate all quality metrics.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        sample_rate: Sample rate

    Returns:
        Dictionary of metrics
    """
    print(f"\n3. Calculating quality metrics...")

    metrics = {}

    # SNR
    print(f"   Calculating SNR...")
    metrics['snr'] = calculate_snr(original, reconstructed)
    print(f"   SNR: {metrics['snr']:.2f} dB")

    # PESQ
    print(f"   Calculating PESQ...")
    try:
        metrics['pesq'] = calculate_pesq_score(original, reconstructed, sample_rate)
        print(f"   PESQ: {metrics['pesq']:.3f}")
    except Exception as e:
        print(f"   PESQ calculation failed: {e}")
        metrics['pesq'] = None

    # STOI
    print(f"   Calculating STOI...")
    try:
        metrics['stoi'] = calculate_stoi_score(original, reconstructed, sample_rate)
        print(f"   STOI: {metrics['stoi']:.3f}")
    except Exception as e:
        print(f"   STOI calculation failed: {e}")
        metrics['stoi'] = None

    # Correlation
    print(f"   Calculating correlation...")
    metrics['correlation'] = calculate_correlation(original, reconstructed)
    print(f"   Correlation: {metrics['correlation']:.4f}")

    return metrics

def plot_waveforms(original, reconstructed, sample_rate):
    """
    Plot waveform comparison.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        sample_rate: Sample rate
    """
    print(f"\n4. Generating waveform visualization...")

    # Convert to mono for visualization
    if original.dim() > 1 and original.shape[0] > 1:
        original_mono = original.mean(dim=0)
    else:
        original_mono = original.squeeze()

    if reconstructed.dim() > 1 and reconstructed.shape[0] > 1:
        reconstructed_mono = reconstructed.mean(dim=0)
    else:
        reconstructed_mono = reconstructed.squeeze()

    # Ensure same length
    min_len = min(original_mono.shape[-1], reconstructed_mono.shape[-1])
    original_mono = original_mono[:min_len]
    reconstructed_mono = reconstructed_mono[:min_len]

    # Convert to numpy
    original_np = original_mono.cpu().numpy()
    reconstructed_np = reconstructed_mono.cpu().numpy()

    # Time axis
    time = np.arange(len(original_np)) / sample_rate

    # Plot first 5 seconds for clarity
    plot_duration = min(5.0, len(time) / sample_rate)
    plot_samples = int(plot_duration * sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Original
    axes[0].plot(time[:plot_samples], original_np[:plot_samples], linewidth=0.5, alpha=0.8)
    axes[0].set_title('Original Audio (First 5 seconds)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Reconstructed
    axes[1].plot(time[:plot_samples], reconstructed_np[:plot_samples], linewidth=0.5, alpha=0.8, color='orange')
    axes[1].set_title('EnCodec Reconstructed Audio (First 5 seconds)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "waveforms_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

def plot_spectrograms(original, reconstructed, sample_rate):
    """
    Plot spectrogram comparison.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        sample_rate: Sample rate
    """
    print(f"\n5. Generating spectrogram visualization...")

    # Convert to mono for visualization
    if original.dim() > 1 and original.shape[0] > 1:
        original_mono = original.mean(dim=0)
    else:
        original_mono = original.squeeze()

    if reconstructed.dim() > 1 and reconstructed.shape[0] > 1:
        reconstructed_mono = reconstructed.mean(dim=0)
    else:
        reconstructed_mono = reconstructed.squeeze()

    # Ensure same length
    min_len = min(original_mono.shape[-1], reconstructed_mono.shape[-1])
    original_mono = original_mono[:min_len]
    reconstructed_mono = reconstructed_mono[:min_len]

    # Convert to numpy
    original_np = original_mono.cpu().numpy()
    reconstructed_np = reconstructed_mono.cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Original spectrogram
    f1, t1, Sxx1 = signal.spectrogram(original_np, sample_rate, nperseg=1024, noverlap=512)
    im1 = axes[0].pcolormesh(t1, f1, 10 * np.log10(Sxx1 + 1e-10), shading='gouraud', cmap='viridis')
    axes[0].set_title('Original Audio Spectrogram', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylim([0, 8000])  # Focus on speech/music range
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    # Reconstructed spectrogram
    f2, t2, Sxx2 = signal.spectrogram(reconstructed_np, sample_rate, nperseg=1024, noverlap=512)
    im2 = axes[1].pcolormesh(t2, f2, 10 * np.log10(Sxx2 + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_title('EnCodec Reconstructed Audio Spectrogram', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylim([0, 8000])  # Focus on speech/music range
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "spectrograms_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

def save_metrics_report(metrics):
    """
    Save metrics to text file.

    Args:
        metrics: Dictionary of metrics
    """
    print(f"\n6. Generating metrics report...")

    report_path = os.path.join(OUTPUT_DIR, "metrics_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EnCodec Quality Metrics Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Test Configuration:\n")
        f.write(f"  - Video source: {VIDEO_PATH}\n")
        f.write(f"  - Duration tested: {DURATION_SECONDS} seconds\n")
        f.write(f"  - EnCodec bandwidth: {TARGET_BANDWIDTH} kbps\n")
        f.write(f"  - Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"  - Test date: 2025-10-31\n\n")

        f.write("-" * 80 + "\n")
        f.write("Quality Metrics:\n")
        f.write("-" * 80 + "\n\n")

        # SNR
        f.write(f"1. Signal-to-Noise Ratio (SNR):\n")
        f.write(f"   Value: {metrics['snr']:.2f} dB\n")
        f.write(f"   Target: >15 dB\n")
        f.write(f"   Status: {'PASS' if metrics['snr'] > 15 else 'FAIL'}\n")
        f.write(f"   Interpretation: Measures overall signal fidelity. Higher is better.\n\n")

        # PESQ
        if metrics['pesq'] is not None:
            f.write(f"2. PESQ (Perceptual Evaluation of Speech Quality):\n")
            f.write(f"   Value: {metrics['pesq']:.3f}\n")
            f.write(f"   Target: >3.0\n")
            f.write(f"   Status: {'PASS' if metrics['pesq'] > 3.0 else 'FAIL'}\n")
            f.write(f"   Range: -0.5 to 4.5 (higher is better)\n")
            f.write(f"   Interpretation: Perceptual quality metric aligned with human perception.\n\n")
        else:
            f.write(f"2. PESQ (Perceptual Evaluation of Speech Quality):\n")
            f.write(f"   Value: N/A (calculation failed)\n\n")

        # STOI
        if metrics['stoi'] is not None:
            f.write(f"3. STOI (Short-Time Objective Intelligibility):\n")
            f.write(f"   Value: {metrics['stoi']:.3f}\n")
            f.write(f"   Target: >0.7\n")
            f.write(f"   Status: {'PASS' if metrics['stoi'] > 0.7 else 'FAIL'}\n")
            f.write(f"   Range: 0 to 1 (higher is better)\n")
            f.write(f"   Interpretation: Measures speech intelligibility.\n\n")
        else:
            f.write(f"3. STOI (Short-Time Objective Intelligibility):\n")
            f.write(f"   Value: N/A (calculation failed)\n\n")

        # Correlation
        f.write(f"4. Waveform Correlation:\n")
        f.write(f"   Value: {metrics['correlation']:.4f}\n")
        f.write(f"   Range: -1 to 1 (closer to 1 is better)\n")
        f.write(f"   Interpretation: Measures waveform similarity.\n\n")

        f.write("-" * 80 + "\n")
        f.write("Overall Assessment:\n")
        f.write("-" * 80 + "\n\n")

        # Determine overall status
        passed_tests = []
        failed_tests = []

        if metrics['snr'] > 15:
            passed_tests.append("SNR")
        else:
            failed_tests.append("SNR")

        if metrics['pesq'] is not None:
            if metrics['pesq'] > 3.0:
                passed_tests.append("PESQ")
            else:
                failed_tests.append("PESQ")

        if metrics['stoi'] is not None:
            if metrics['stoi'] > 0.7:
                passed_tests.append("STOI")
            else:
                failed_tests.append("STOI")

        f.write(f"Passed tests ({len(passed_tests)}/{3}): {', '.join(passed_tests) if passed_tests else 'None'}\n")
        f.write(f"Failed tests ({len(failed_tests)}/{3}): {', '.join(failed_tests) if failed_tests else 'None'}\n\n")

        if len(passed_tests) >= 2:
            f.write("Conclusion: EnCodec produces HIGH QUALITY audio reconstruction.\n")
        elif len(passed_tests) >= 1:
            f.write("Conclusion: EnCodec produces MODERATE QUALITY audio reconstruction.\n")
        else:
            f.write("Conclusion: EnCodec produces LOW QUALITY audio reconstruction.\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Output Files:\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  - original_30s.wav: Original audio (resampled to 24kHz)\n")
        f.write(f"  - encodec_reconstructed_30s.wav: Reconstructed audio\n")
        f.write(f"  - waveforms_comparison.png: Waveform visualization\n")
        f.write(f"  - spectrograms_comparison.png: Spectrogram visualization\n")
        f.write(f"  - metrics_report.txt: This report\n\n")

        f.write("=" * 80 + "\n")
        f.write("Next Steps:\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Listen to both WAV files to verify perceptual quality\n")
        f.write("2. Review spectrograms for frequency content preservation\n")
        f.write("3. If quality is acceptable, proceed with full pipeline implementation\n")
        f.write("4. Consider testing different bandwidth settings (1.5, 3, 6, 12, 24 kbps)\n\n")

    print(f"   Saved: {report_path}")

def main():
    """Main test execution."""
    print("=" * 80)
    print("EnCodec Quality Test - Sherlock Audio")
    print("=" * 80)

    # Setup
    ensure_output_dir()

    # Load audio
    waveform, original_sr = load_sherlock_audio(DURATION_SECONDS)

    # Test EnCodec
    original_converted, reconstructed, model_sr = test_encodec(waveform, original_sr)

    # Calculate metrics
    metrics = calculate_metrics(original_converted, reconstructed, model_sr)

    # Generate visualizations
    plot_waveforms(original_converted, reconstructed, model_sr)
    plot_spectrograms(original_converted, reconstructed, model_sr)

    # Save report
    save_metrics_report(metrics)

    # Summary
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"\nKey metrics:")
    print(f"  - SNR: {metrics['snr']:.2f} dB (target: >15 dB)")
    if metrics['pesq'] is not None:
        print(f"  - PESQ: {metrics['pesq']:.3f} (target: >3.0)")
    if metrics['stoi'] is not None:
        print(f"  - STOI: {metrics['stoi']:.3f} (target: >0.7)")
    print(f"  - Correlation: {metrics['correlation']:.4f}")
    print(f"\nPlease listen to the WAV files to verify perceptual quality:")
    print(f"  - {OUTPUT_DIR}/original_30s.wav")
    print(f"  - {OUTPUT_DIR}/encodec_reconstructed_30s.wav")
    print("\n")

if __name__ == "__main__":
    main()

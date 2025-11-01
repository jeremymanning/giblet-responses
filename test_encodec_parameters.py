#!/usr/bin/env python3
"""
EnCodec Parameter Sweep for Issue #24, Task 1.2

Evaluates different EnCodec parameter settings for quality vs. compression tradeoff.

Tests:
1. Bandwidth settings: 1.5, 3.0, 6.0, 12.0, 24.0 kbps
2. Model variants: 24kHz (mono), 48kHz (stereo)
3. Three different 10-second segments from Sherlock stimulus

Metrics:
- SNR (Signal-to-Noise Ratio) in dB
- PESQ (Perceptual Evaluation of Speech Quality): 1.0-4.5
- STOI (Short-Time Objective Intelligibility): 0.0-1.0
- File size / compression ratio
- Encoded dimensions
- Computational time

Outputs:
- encodec_parameter_sweep/ directory with:
  - WAV files for each configuration
  - Comparison table (CSV and markdown)
  - Spectrograms for visual comparison
  - Recommendation report
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Import EnCodec
import encodec
from encodec import EncodecModel
from encodec.utils import convert_audio

# Import audio quality metrics
from pesq import pesq
from pystoi import stoi

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EnCodecParameterEvaluator:
    """Evaluate EnCodec with different parameter settings."""

    def __init__(self,
                 stimulus_path: str,
                 output_dir: str = "encodec_parameter_sweep",
                 device: str = "cpu"):
        """
        Initialize evaluator.

        Args:
            stimulus_path: Path to Sherlock stimulus video/audio
            output_dir: Directory for output files
            device: Device to run on ('cpu' or 'cuda')
        """
        self.stimulus_path = Path(stimulus_path)
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define test segments (10 seconds each)
        self.segments = {
            'speech': (0.0, 10.0),           # Speech-heavy dialogue
            'music': (600.0, 610.0),          # Music/background sounds
            'mixed': (1200.0, 1210.0)         # Mixed dialogue + sound effects
        }

        # Define bandwidth settings to test (in kbps)
        self.bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]

        # Define model variants
        self.models = {
            '24khz': 'encodec_24khz',  # Mono, 24kHz
            '48khz': 'encodec_48khz'   # Stereo, 48kHz
        }

        # Store results
        self.results = []

    def load_audio(self) -> Tuple[torch.Tensor, int]:
        """
        Load audio from Sherlock stimulus.

        Returns:
            (audio, sample_rate): Audio tensor and sample rate
        """
        print(f"\nLoading audio from: {self.stimulus_path}")

        # Load audio using soundfile (more reliable than torchaudio for WAV files)
        import soundfile as sf
        audio_np, sr = sf.read(str(self.stimulus_path))

        # Convert to torch tensor and transpose to [channels, samples]
        audio = torch.from_numpy(audio_np.T).float()

        print(f"  Original shape: {audio.shape}, Sample rate: {sr} Hz")
        print(f"  Duration: {audio.shape[1] / sr:.2f} seconds")

        return audio, sr

    def extract_segment(self,
                       audio: torch.Tensor,
                       sr: int,
                       start_sec: float,
                       end_sec: float) -> torch.Tensor:
        """
        Extract a time segment from audio.

        Args:
            audio: Full audio tensor [channels, samples]
            sr: Sample rate
            start_sec: Start time in seconds
            end_sec: End time in seconds

        Returns:
            segment: Audio segment [channels, samples]
        """
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        return audio[:, start_sample:end_sample]

    def calculate_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.

        Args:
            original: Original audio
            reconstructed: Reconstructed audio

        Returns:
            SNR in dB
        """
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def calculate_pesq(self,
                      original: np.ndarray,
                      reconstructed: np.ndarray,
                      sr: int) -> float:
        """
        Calculate PESQ (Perceptual Evaluation of Speech Quality).

        Args:
            original: Original audio (1D)
            reconstructed: Reconstructed audio (1D)
            sr: Sample rate (must be 8000 or 16000 for PESQ)

        Returns:
            PESQ score (1.0-4.5, higher is better)
        """
        try:
            # PESQ requires 8kHz or 16kHz
            if sr not in [8000, 16000]:
                # Resample to 16kHz
                import scipy.signal
                target_sr = 16000
                num_samples = int(len(original) * target_sr / sr)
                original = scipy.signal.resample(original, num_samples)
                reconstructed = scipy.signal.resample(reconstructed, num_samples)
                sr = target_sr

            # Calculate PESQ (mode: 'wb' for wideband, 'nb' for narrowband)
            mode = 'wb' if sr == 16000 else 'nb'
            score = pesq(sr, original, reconstructed, mode)
            return score
        except Exception as e:
            print(f"    Warning: PESQ calculation failed: {e}")
            return -1.0

    def calculate_stoi(self,
                      original: np.ndarray,
                      reconstructed: np.ndarray,
                      sr: int) -> float:
        """
        Calculate STOI (Short-Time Objective Intelligibility).

        Args:
            original: Original audio (1D)
            reconstructed: Reconstructed audio (1D)
            sr: Sample rate

        Returns:
            STOI score (0.0-1.0, higher is better)
        """
        try:
            # STOI works with sample rates >= 10kHz
            score = stoi(original, reconstructed, sr, extended=False)
            return score
        except Exception as e:
            print(f"    Warning: STOI calculation failed: {e}")
            return -1.0

    def encode_decode(self,
                     audio: torch.Tensor,
                     sr: int,
                     model_name: str,
                     bandwidth: float) -> Tuple[torch.Tensor, Dict]:
        """
        Encode and decode audio with EnCodec.

        Args:
            audio: Audio tensor [channels, samples]
            sr: Sample rate
            model_name: Model variant ('encodec_24khz' or 'encodec_48khz')
            bandwidth: Target bandwidth in kbps

        Returns:
            (reconstructed_audio, metadata): Reconstructed audio and encoding metadata
        """
        # Load model
        model = EncodecModel.encodec_model_24khz() if model_name == 'encodec_24khz' \
                else EncodecModel.encodec_model_48khz()

        # Check if bandwidth is supported
        # 24kHz: [1.5, 3.0, 6.0, 12.0, 24.0]
        # 48kHz: [3.0, 6.0, 12.0, 24.0]
        try:
            model.set_target_bandwidth(bandwidth)
        except ValueError as e:
            raise ValueError(f"Bandwidth {bandwidth} not supported by {model_name}: {e}")

        model.to(self.device)

        # Get model sample rate
        model_sr = model.sample_rate

        # Convert audio to model format
        audio_converted = convert_audio(audio, sr, model_sr, model.channels)
        audio_converted = audio_converted.unsqueeze(0).to(self.device)  # Add batch dimension

        # Encode
        start_time = time.time()
        with torch.no_grad():
            encoded_frames = model.encode(audio_converted)
        encode_time = time.time() - start_time

        # Get encoded dimensions
        codes = encoded_frames[0][0]  # [num_codebooks, num_frames]
        encoded_shape = codes.shape

        # Calculate compression ratio
        original_bytes = audio.numel() * 4  # float32 = 4 bytes
        # EnCodec codes are integers, typically stored as int64
        encoded_bytes = codes.numel() * 8  # int64 = 8 bytes (conservative estimate)
        compression_ratio = original_bytes / encoded_bytes

        # Decode
        start_time = time.time()
        with torch.no_grad():
            reconstructed = model.decode(encoded_frames)
        decode_time = time.time() - start_time

        # Remove batch dimension and move to CPU
        reconstructed = reconstructed.squeeze(0).cpu()

        # Convert back to original sample rate if needed
        if model_sr != sr:
            reconstructed = torchaudio.transforms.Resample(model_sr, sr)(reconstructed)

        # Ensure same length as original
        if reconstructed.shape[1] > audio.shape[1]:
            reconstructed = reconstructed[:, :audio.shape[1]]
        elif reconstructed.shape[1] < audio.shape[1]:
            # Pad with zeros
            pad_size = audio.shape[1] - reconstructed.shape[1]
            reconstructed = torch.nn.functional.pad(reconstructed, (0, pad_size))

        metadata = {
            'encoded_shape': encoded_shape,
            'compression_ratio': compression_ratio,
            'original_bytes': original_bytes,
            'encoded_bytes': encoded_bytes,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'total_time': encode_time + decode_time
        }

        return reconstructed, metadata

    def evaluate_configuration(self,
                              segment_name: str,
                              audio: torch.Tensor,
                              sr: int,
                              model_name: str,
                              bandwidth: float) -> Dict:
        """
        Evaluate a single EnCodec configuration.

        Args:
            segment_name: Name of audio segment
            audio: Audio tensor [channels, samples]
            sr: Sample rate
            model_name: Model variant
            bandwidth: Target bandwidth in kbps

        Returns:
            result: Dictionary with all metrics
        """
        # Encode and decode
        reconstructed, metadata = self.encode_decode(audio, sr, model_name, bandwidth)

        # Convert to numpy for metric calculation (use first channel if stereo)
        original_np = audio[0].numpy() if audio.shape[0] > 1 else audio.squeeze().numpy()
        reconstructed_np = reconstructed[0].numpy() if reconstructed.shape[0] > 1 \
                          else reconstructed.squeeze().numpy()

        # Calculate metrics
        snr = self.calculate_snr(original_np, reconstructed_np)
        pesq_score = self.calculate_pesq(original_np, reconstructed_np, sr)
        stoi_score = self.calculate_stoi(original_np, reconstructed_np, sr)

        # Save reconstructed audio
        filename = f"{segment_name}_{model_name}_bw{bandwidth:.1f}kbps.wav"
        filepath = self.output_dir / filename

        # Use soundfile for saving (more reliable than torchaudio)
        import soundfile as sf
        reconstructed_np = reconstructed.numpy().T  # Transpose to [samples, channels]
        sf.write(str(filepath), reconstructed_np, sr)

        result = {
            'segment': segment_name,
            'model': model_name,
            'bandwidth_kbps': bandwidth,
            'snr_db': snr,
            'pesq': pesq_score,
            'stoi': stoi_score,
            'encoded_shape': str(metadata['encoded_shape']),
            'compression_ratio': metadata['compression_ratio'],
            'original_bytes': metadata['original_bytes'],
            'encoded_bytes': metadata['encoded_bytes'],
            'encode_time_sec': metadata['encode_time'],
            'decode_time_sec': metadata['decode_time'],
            'total_time_sec': metadata['total_time'],
            'output_file': filename
        }

        return result

    def plot_spectrograms(self,
                         segment_name: str,
                         original: torch.Tensor,
                         reconstructed_configs: List[Tuple[str, torch.Tensor]],
                         sr: int):
        """
        Plot spectrograms for visual comparison.

        Args:
            segment_name: Name of audio segment
            original: Original audio tensor
            reconstructed_configs: List of (config_name, audio) tuples
            sr: Sample rate
        """
        n_configs = len(reconstructed_configs) + 1  # +1 for original
        fig, axes = plt.subplots(n_configs, 1, figsize=(12, 3 * n_configs))

        if n_configs == 1:
            axes = [axes]

        # Plot original (convert to 1D numpy array)
        original_np = original[0].numpy() if original.shape[0] > 1 else original.squeeze().numpy()
        # Ensure 1D
        if original_np.ndim > 1:
            original_np = original_np.flatten()
        self._plot_single_spectrogram(axes[0], original_np, sr, "Original")

        # Plot reconstructed versions
        for i, (config_name, audio) in enumerate(reconstructed_configs, start=1):
            audio_np = audio[0].numpy() if audio.shape[0] > 1 else audio.squeeze().numpy()
            # Ensure 1D
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            self._plot_single_spectrogram(axes[i], audio_np, sr, config_name)

        plt.tight_layout()
        output_path = self.output_dir / f"{segment_name}_spectrograms.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved spectrogram: {output_path}")

    def _plot_single_spectrogram(self, ax, audio: np.ndarray, sr: int, title: str):
        """Plot a single spectrogram."""
        import librosa.display

        # Ensure audio is 1D numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        if audio.ndim > 1:
            audio = audio.flatten()

        # Compute mel spectrogram using librosa
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                           fmax=8000, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Plot
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                       sr=sr, fmax=8000, ax=ax, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=ax, label='Power (dB)', format='%+2.0f dB')

    def run_evaluation(self):
        """Run complete parameter sweep evaluation."""
        print("=" * 80)
        print("EnCodec Parameter Sweep Evaluation")
        print("=" * 80)

        # Load audio
        full_audio, sr = self.load_audio()

        # Test each segment
        for segment_name, (start, end) in self.segments.items():
            print(f"\n{'=' * 80}")
            print(f"Processing segment: {segment_name} ({start:.1f}s - {end:.1f}s)")
            print(f"{'=' * 80}")

            # Extract segment
            audio_segment = self.extract_segment(full_audio, sr, start, end)

            # Save original segment
            original_path = self.output_dir / f"{segment_name}_original.wav"
            import soundfile as sf
            original_np = audio_segment.numpy().T  # Transpose to [samples, channels]
            sf.write(str(original_path), original_np, sr)

            # Test each configuration
            reconstructed_for_plotting = []

            for model_key, model_name in self.models.items():
                for bandwidth in self.bandwidths:
                    config_name = f"{model_key}_bw{bandwidth:.1f}kbps"
                    print(f"\n  Testing: {config_name}")

                    try:
                        result = self.evaluate_configuration(
                            segment_name, audio_segment, sr, model_name, bandwidth
                        )
                        self.results.append(result)

                        print(f"    SNR: {result['snr_db']:.2f} dB")
                        print(f"    PESQ: {result['pesq']:.3f}")
                        print(f"    STOI: {result['stoi']:.3f}")
                        print(f"    Compression: {result['compression_ratio']:.2f}x")
                        print(f"    Time: {result['total_time_sec']:.3f}s")

                        # Collect for plotting (only 24kHz model to avoid clutter)
                        if model_key == '24khz':
                            reconstructed_path = self.output_dir / result['output_file']
                            import soundfile as sf
                            reconstructed_np, _ = sf.read(str(reconstructed_path))
                            reconstructed = torch.from_numpy(reconstructed_np.T).float()
                            reconstructed_for_plotting.append((config_name, reconstructed))

                    except ValueError as e:
                        # Bandwidth not supported
                        print(f"    SKIPPED: {e}")
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        import traceback
                        traceback.print_exc()

            # Plot spectrograms (24kHz model only)
            if reconstructed_for_plotting:
                print(f"\n  Generating spectrograms...")
                try:
                    self.plot_spectrograms(segment_name, audio_segment,
                                          reconstructed_for_plotting, sr)
                except Exception as e:
                    print(f"    Warning: Spectrogram generation failed: {e}")
                    print(f"    Skipping spectrograms for {segment_name}")

    def generate_report(self):
        """Generate comparison tables and recommendation report."""
        print(f"\n{'=' * 80}")
        print("Generating Report")
        print(f"{'=' * 80}")

        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Save full results as CSV
        csv_path = self.output_dir / "encodec_parameter_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")

        # Generate markdown table
        md_path = self.output_dir / "encodec_parameter_comparison.md"
        with open(md_path, 'w') as f:
            f.write("# EnCodec Parameter Comparison\n\n")
            f.write("## Summary Statistics\n\n")

            # Summary by bandwidth (24kHz model only, averaged across segments)
            df_24khz = df[df['model'] == 'encodec_24khz']
            summary = df_24khz.groupby('bandwidth_kbps').agg({
                'snr_db': 'mean',
                'pesq': 'mean',
                'stoi': 'mean',
                'compression_ratio': 'mean',
                'total_time_sec': 'mean'
            }).round(3)

            f.write("### By Bandwidth (24kHz model, averaged across segments)\n\n")
            f.write(summary.to_markdown())
            f.write("\n\n")

            # Full results table
            f.write("## Detailed Results\n\n")

            # Format numeric columns
            display_cols = [
                'segment', 'model', 'bandwidth_kbps',
                'snr_db', 'pesq', 'stoi',
                'compression_ratio', 'total_time_sec'
            ]
            df_display = df[display_cols].copy()
            df_display['snr_db'] = df_display['snr_db'].round(2)
            df_display['pesq'] = df_display['pesq'].round(3)
            df_display['stoi'] = df_display['stoi'].round(3)
            df_display['compression_ratio'] = df_display['compression_ratio'].round(2)
            df_display['total_time_sec'] = df_display['total_time_sec'].round(3)

            f.write(df_display.to_markdown(index=False))
            f.write("\n\n")

            # Generate recommendation
            f.write("## Recommendation\n\n")

            # Decision criteria thresholds
            min_acceptable = {'snr_db': 12, 'pesq': 2.5, 'stoi': 0.6}
            target = {'snr_db': 15, 'pesq': 3.0, 'stoi': 0.7}

            # Find best bandwidth for 24kHz model
            df_24khz = df[df['model'] == 'encodec_24khz']
            avg_metrics = df_24khz.groupby('bandwidth_kbps').agg({
                'snr_db': 'mean',
                'pesq': 'mean',
                'stoi': 'mean',
                'compression_ratio': 'mean'
            })

            # Check which bandwidths meet minimum requirements
            meets_min = (
                (avg_metrics['snr_db'] >= min_acceptable['snr_db']) &
                (avg_metrics['pesq'] >= min_acceptable['pesq']) &
                (avg_metrics['stoi'] >= min_acceptable['stoi'])
            )

            meets_target = (
                (avg_metrics['snr_db'] >= target['snr_db']) &
                (avg_metrics['pesq'] >= target['pesq']) &
                (avg_metrics['stoi'] >= target['stoi'])
            )

            if meets_target.any():
                # Find lowest bandwidth that meets target
                recommended_bw = avg_metrics[meets_target].index.min()
                tier = "TARGET"
            elif meets_min.any():
                # Find lowest bandwidth that meets minimum
                recommended_bw = avg_metrics[meets_min].index.min()
                tier = "MINIMUM ACCEPTABLE"
            else:
                # Use highest bandwidth available
                recommended_bw = avg_metrics.index.max()
                tier = "BEST AVAILABLE (below minimum)"

            f.write(f"**Recommended Setting:** EnCodec 24kHz, {recommended_bw} kbps\n\n")
            f.write(f"**Quality Tier:** {tier}\n\n")
            f.write(f"**Average Metrics:**\n")
            f.write(f"- SNR: {avg_metrics.loc[recommended_bw, 'snr_db']:.2f} dB\n")
            f.write(f"- PESQ: {avg_metrics.loc[recommended_bw, 'pesq']:.3f}\n")
            f.write(f"- STOI: {avg_metrics.loc[recommended_bw, 'stoi']:.3f}\n")
            f.write(f"- Compression: {avg_metrics.loc[recommended_bw, 'compression_ratio']:.2f}x\n\n")

            f.write(f"**Justification:**\n")
            if tier == "TARGET":
                f.write(f"- Meets all target quality thresholds (SNR >15dB, PESQ >3.0, STOI >0.7)\n")
                f.write(f"- {recommended_bw} kbps provides best balance of quality and compression\n")
            elif tier == "MINIMUM ACCEPTABLE":
                f.write(f"- Meets minimum acceptable thresholds (SNR >12dB, PESQ >2.5, STOI >0.6)\n")
                f.write(f"- {recommended_bw} kbps is lowest bandwidth achieving acceptable quality\n")
            else:
                f.write(f"- Warning: No bandwidth setting meets minimum quality thresholds\n")
                f.write(f"- {recommended_bw} kbps provides best available quality\n")
                f.write(f"- Consider alternative approaches (e.g., Complex FFT, higher bandwidth)\n")

            f.write(f"\n**Implementation Notes:**\n")
            f.write(f"- Use `EncodecModel.encodec_model_24khz()` (mono, 24kHz)\n")
            f.write(f"- Set target bandwidth: `model.set_target_bandwidth({recommended_bw})`\n")
            f.write(f"- Expected encoded shape: ~[8, N_frames] for {recommended_bw} kbps\n")
            f.write(f"- Integration: Replace mel spectrogram pipeline with EnCodec codes\n")

        print(f"Saved Markdown report: {md_path}")

        # Print summary to console
        print(f"\n{'=' * 80}")
        print("RECOMMENDATION SUMMARY")
        print(f"{'=' * 80}")
        with open(md_path, 'r') as f:
            lines = f.readlines()
            in_rec_section = False
            for line in lines:
                if line.startswith("## Recommendation"):
                    in_rec_section = True
                if in_rec_section:
                    print(line.rstrip())


def main():
    """Main entry point."""
    # Path to Sherlock stimulus audio (extracted from video)
    stimulus_path = "/Users/jmanning/giblet-responses/data/stimuli_Sherlock_audio.wav"

    if not os.path.exists(stimulus_path):
        print(f"ERROR: Audio stimulus not found at {stimulus_path}")
        print("Extracting audio from video...")
        import subprocess
        video_path = "/Users/jmanning/giblet-responses/data/stimuli_Sherlock.m4v"
        if not os.path.exists(video_path):
            print(f"ERROR: Video not found at {video_path}")
            print("Please run ./download_data_from_dropbox.sh first")
            return 1

        # Extract audio using ffmpeg
        result = subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '48000', '-ac', '2', stimulus_path, '-y'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: Failed to extract audio: {result.stderr}")
            return 1
        print(f"Audio extracted successfully to {stimulus_path}")
    else:
        print(f"Using existing audio file: {stimulus_path}")

    # Create evaluator
    evaluator = EnCodecParameterEvaluator(
        stimulus_path=stimulus_path,
        output_dir="encodec_parameter_sweep",
        device="cpu"  # Use "cuda" if GPU available
    )

    # Run evaluation
    evaluator.run_evaluation()

    # Generate report
    evaluator.generate_report()

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {evaluator.output_dir}")
    print("\nFiles generated:")
    print("  - encodec_parameter_comparison.csv (full results)")
    print("  - encodec_parameter_comparison.md (summary + recommendation)")
    print("  - *.wav files (original and reconstructed audio)")
    print("  - *_spectrograms.png (visual comparisons)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

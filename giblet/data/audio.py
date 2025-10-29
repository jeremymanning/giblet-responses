"""
Audio processing module for Sherlock fMRI project.

Handles bidirectional conversion between audio and mel spectrogram features:
- Audio → Features: Extract mel spectrogram aligned to fMRI TRs
- Features → Audio: Reconstruct audio using HiFi-GAN vocoder

All temporal alignment uses TR = 1.5 seconds.
"""

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
from tqdm import tqdm

# Optional: torchaudio for HiFi-GAN vocoder
try:
    import torch
    import torchaudio
    VOCODER_AVAILABLE = True
except (ImportError, OSError) as e:
    VOCODER_AVAILABLE = False
    print(f"Warning: torchaudio not available ({e}). Audio reconstruction will use Griffin-Lim.")


class AudioProcessor:
    """
    Process audio for multimodal autoencoder training.

    Handles:
    - Audio extraction from video files
    - Resampling to 22,050 Hz (HiFi-GAN standard)
    - Mel spectrogram computation (128 mels)
    - Temporal aggregation to match fMRI TR (1.5s bins)
    - Audio reconstruction using HiFi-GAN vocoder

    Parameters
    ----------
    sample_rate : int, default=22050
        Target sample rate (Hz)
    n_mels : int, default=128
        Number of mel frequency bins
    n_fft : int, default=1024
        FFT window size
    hop_length : int, default=256
        Number of samples between successive frames
    tr : float, default=1.5
        fMRI repetition time in seconds
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        tr: float = 1.5
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tr = tr
        self.n_features = n_mels

        # Initialize HiFi-GAN vocoder (lazy loading)
        self._vocoder = None
        self._vocoder_sample_rate = None

    def _load_vocoder(self):
        """Load HiFi-GAN vocoder for audio reconstruction."""
        if not VOCODER_AVAILABLE:
            raise RuntimeError("torchaudio not available. Cannot load HiFi-GAN vocoder.")

        if self._vocoder is None:
            print("Loading HiFi-GAN vocoder...")
            # Use pretrained HiFi-GAN vocoder
            bundle = torchaudio.pipelines.HIFIGAN_VOCODER_V3_LJSPEECH
            self._vocoder = bundle.get_vocoder()
            self._vocoder_sample_rate = bundle.sample_rate
            self._vocoder.eval()
            print(f"Vocoder loaded (sample rate: {self._vocoder_sample_rate} Hz)")

    def audio_to_features(
        self,
        audio_source: Union[str, Path],
        max_trs: Optional[int] = None,
        from_video: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert audio to mel spectrogram features aligned to fMRI TRs.

        Parameters
        ----------
        audio_source : str or Path
            Path to audio/video file
        max_trs : int, optional
            Maximum number of TRs to extract
        from_video : bool, default=True
            If True, extract audio from video file

        Returns
        -------
        features : np.ndarray
            Shape (n_trs, n_mels) mel spectrogram in dB scale
        metadata : pd.DataFrame
            DataFrame with: tr_index, start_time, end_time

        Notes
        -----
        - Uses librosa to extract mel spectrogram
        - Converts to log scale (dB)
        - Aggregates spectrogram frames within each TR bin
        """
        audio_source = Path(audio_source)

        # Load audio
        if from_video:
            # Extract audio from video
            y, sr = librosa.load(str(audio_source), sr=self.sample_rate, mono=False)
            # If stereo, convert to mono by averaging channels
            if y.ndim > 1:
                y = np.mean(y, axis=0)
        else:
            y, sr = librosa.load(str(audio_source), sr=self.sample_rate)

        duration = len(y) / sr

        # Calculate number of TRs
        n_trs = int(np.floor(duration / self.tr))
        if max_trs is not None:
            n_trs = min(n_trs, max_trs)

        # Compute full mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Aggregate to TRs
        features = np.zeros((n_trs, self.n_mels), dtype=np.float32)
        tr_metadata = []

        samples_per_tr = int(self.tr * sr)
        frames_per_sample = self.hop_length

        for tr_idx in range(n_trs):
            # Time window
            start_time = tr_idx * self.tr
            end_time = start_time + self.tr

            # Sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Spectrogram frame indices
            start_frame = start_sample // self.hop_length
            end_frame = end_sample // self.hop_length

            # Average spectrogram frames within TR
            if end_frame > start_frame:
                features[tr_idx] = np.mean(
                    mel_spec_db[:, start_frame:end_frame],
                    axis=1
                )
            else:
                features[tr_idx] = mel_spec_db[:, start_frame]

            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time
            })

        metadata_df = pd.DataFrame(tr_metadata)

        return features, metadata_df

    def features_to_audio(
        self,
        features: np.ndarray,
        output_path: Union[str, Path],
        use_vocoder: bool = True
    ) -> None:
        """
        Reconstruct audio from mel spectrogram features.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_mels) mel spectrogram in dB scale
        output_path : str or Path
            Path for output audio file
        use_vocoder : bool, default=True
            If True, use HiFi-GAN vocoder for high-quality reconstruction
            If False, use Griffin-Lim algorithm (faster but lower quality)

        Notes
        -----
        - HiFi-GAN produces much better quality than Griffin-Lim
        - Reconstructed audio will be at self.sample_rate Hz
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trs = features.shape[0]

        if use_vocoder:
            # Use HiFi-GAN vocoder for high-quality reconstruction
            self._load_vocoder()

            # Expand TR-level features to frames
            # Each TR needs to be expanded to fill self.tr seconds
            samples_per_tr = int(self.tr * self.sample_rate)
            frames_per_tr = samples_per_tr // self.hop_length

            # Build full mel spectrogram by repeating each TR
            full_mel_spec_db = np.zeros(
                (self.n_mels, n_trs * frames_per_tr),
                dtype=np.float32
            )

            for tr_idx in range(n_trs):
                start_frame = tr_idx * frames_per_tr
                end_frame = start_frame + frames_per_tr
                # Repeat the TR's features across all frames in that TR
                full_mel_spec_db[:, start_frame:end_frame] = features[tr_idx:tr_idx+1].T

            # Convert from dB back to power
            mel_spec = librosa.db_to_power(full_mel_spec_db)

            # Convert to tensor for vocoder
            mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

            # Generate audio using vocoder
            with torch.no_grad():
                waveform = self._vocoder(mel_spec_tensor)

            # Save audio
            waveform_np = waveform.squeeze().cpu().numpy()
            sf.write(
                str(output_path),
                waveform_np,
                self._vocoder_sample_rate
            )

        else:
            # Fallback: Griffin-Lim algorithm (lower quality)
            # Similar expansion as above
            samples_per_tr = int(self.tr * self.sample_rate)
            frames_per_tr = samples_per_tr // self.hop_length

            full_mel_spec_db = np.zeros(
                (self.n_mels, n_trs * frames_per_tr),
                dtype=np.float32
            )

            for tr_idx in range(n_trs):
                start_frame = tr_idx * frames_per_tr
                end_frame = start_frame + frames_per_tr
                full_mel_spec_db[:, start_frame:end_frame] = features[tr_idx:tr_idx+1].T

            # Convert from dB to power
            mel_spec = librosa.db_to_power(full_mel_spec_db)

            # Invert mel spectrogram to audio
            # Note: Griffin-Lim produces slightly shorter output than exact TR duration
            # This is acceptable for reconstruction purposes
            y = librosa.feature.inverse.mel_to_audio(
                mel_spec,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Save
            sf.write(str(output_path), y, self.sample_rate)

    def get_audio_info(
        self,
        audio_source: Union[str, Path],
        from_video: bool = True
    ) -> dict:
        """
        Get audio metadata.

        Parameters
        ----------
        audio_source : str or Path
            Path to audio/video file
        from_video : bool, default=True
            If True, extract from video file

        Returns
        -------
        info : dict
            Dictionary with: sample_rate, duration, samples, n_trs
        """
        if from_video:
            y, sr = librosa.load(str(audio_source), sr=self.sample_rate, mono=False)
            if y.ndim > 1:
                y = np.mean(y, axis=0)
        else:
            y, sr = librosa.load(str(audio_source), sr=self.sample_rate)

        duration = len(y) / sr
        n_trs = int(np.floor(duration / self.tr))

        return {
            'sample_rate': sr,
            'duration': duration,
            'samples': len(y),
            'n_trs': n_trs
        }

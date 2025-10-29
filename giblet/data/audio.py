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

# PyTorch is optional for future neural vocoder support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioProcessor:
    """
    Process audio for multimodal autoencoder training.

    Handles:
    - Audio extraction from video files
    - Resampling to 22,050 Hz
    - Mel spectrogram computation (2048 mels for maximum detail preservation)
    - Temporal aggregation to match fMRI TR (1.5s bins)
    - Audio reconstruction using Griffin-Lim algorithm

    Parameters
    ----------
    sample_rate : int, default=22050
        Target sample rate (Hz)
    n_mels : int, default=2048
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
        sample_rate: int = 22050,  # Standard for speech/music
        n_mels: int = 2048,        # Very high resolution for detail preservation
        n_fft: int = 4096,         # Larger FFT for frequency resolution
        hop_length: int = 512,     # Frame advance
        tr: float = 1.5
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tr = tr
        self.n_features = n_mels

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
        output_path: Union[str, Path]
    ) -> None:
        """
        Reconstruct audio from mel spectrogram features using Griffin-Lim.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_mels) mel spectrogram in dB scale
        output_path : str or Path
            Path for output audio file

        Notes
        -----
        - Uses Griffin-Lim algorithm for phase reconstruction
        - Reconstructed audio will be at self.sample_rate Hz
        - Griffin-Lim may produce slightly shorter output than exact TR duration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trs = features.shape[0]

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

        # Convert from dB to power
        mel_spec = librosa.db_to_power(full_mel_spec_db)

        # Invert mel spectrogram to audio using Griffin-Lim
        y = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Save audio
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

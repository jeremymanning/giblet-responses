"""
Audio processing module for multimodal fMRI autoencoder project.

Handles bidirectional conversion between audio and mel spectrogram features:
- Audio → Features: Extract mel spectrogram aligned to fMRI TRs
- Features → Audio: Reconstruct audio using HiFi-GAN vocoder

Default temporal alignment uses TR = 1.5 seconds (configurable).
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

        FIXED: Now preserves temporal structure within each TR instead of averaging.

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
            Shape (n_trs, n_mels, frames_per_tr) mel spectrogram in dB scale
            CHANGED: Now 3D array preserving temporal frames within each TR
        metadata : pd.DataFrame
            DataFrame with: tr_index, start_time, end_time, n_frames

        Notes
        -----
        - Uses librosa to extract mel spectrogram
        - Converts to log scale (dB)
        - Preserves ALL temporal frames within each TR (no averaging)
        - Pads/crops frames to consistent frames_per_tr
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

        # Calculate expected frames per TR
        samples_per_tr = int(self.tr * sr)
        max_frames_per_tr = int(np.ceil(samples_per_tr / self.hop_length))

        # Preserve temporal frames within each TR (NO AVERAGING)
        # Output shape: (n_trs, n_mels, max_frames_per_tr)
        features = np.zeros((n_trs, self.n_mels, max_frames_per_tr), dtype=np.float32)
        tr_metadata = []

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

            # Extract frames for this TR
            if end_frame > start_frame:
                tr_frames = mel_spec_db[:, start_frame:end_frame]  # (n_mels, n_frames)
            else:
                tr_frames = mel_spec_db[:, start_frame:start_frame+1]

            # Pad or crop to max_frames_per_tr
            n_frames = tr_frames.shape[1]
            if n_frames < max_frames_per_tr:
                # Pad with zeros
                padding = max_frames_per_tr - n_frames
                tr_frames = np.pad(tr_frames, ((0, 0), (0, padding)), mode='constant', constant_values=-80.0)
            elif n_frames > max_frames_per_tr:
                # Crop to max_frames_per_tr
                tr_frames = tr_frames[:, :max_frames_per_tr]

            features[tr_idx] = tr_frames  # (n_mels, max_frames_per_tr)

            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_frames': n_frames
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

        FIXED: Now handles 3D features with preserved temporal structure.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_mels, frames_per_tr) mel spectrogram in dB scale
            CHANGED: Now expects 3D array with temporal frames
        output_path : str or Path
            Path for output audio file

        Notes
        -----
        - Uses Griffin-Lim algorithm for phase reconstruction
        - Reconstructed audio will be at self.sample_rate Hz
        - Temporal structure within each TR is preserved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle both 2D (old format) and 3D (new format) features
        if features.ndim == 2:
            print("Warning: 2D features detected (old format). Converting to 3D...")
            # Old format: (n_trs, n_mels)
            n_trs = features.shape[0]
            samples_per_tr = int(self.tr * self.sample_rate)
            frames_per_tr = samples_per_tr // self.hop_length

            # Repeat features across frames (old behavior)
            features_3d = np.zeros((n_trs, self.n_mels, frames_per_tr), dtype=np.float32)
            for tr_idx in range(n_trs):
                features_3d[tr_idx] = features[tr_idx:tr_idx+1].T  # Broadcast
            features = features_3d

        # Now features is 3D: (n_trs, n_mels, frames_per_tr)
        n_trs, n_mels, frames_per_tr = features.shape

        # Concatenate all TRs along time axis
        # Reshape from (n_trs, n_mels, frames_per_tr) to (n_mels, n_trs * frames_per_tr)
        full_mel_spec_db = features.transpose(1, 0, 2).reshape(n_mels, -1)

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

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

# EnCodec neural audio codec (optional)
try:
    from transformers import EncodecModel, AutoProcessor
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False


class AudioProcessor:
    """
    Process audio for multimodal autoencoder training.

    Supports two encoding modes:
    1. Mel spectrogram (legacy): 2048 mels with Griffin-Lim reconstruction
    2. EnCodec neural codec (default): 24kHz mono with learned compression

    Handles:
    - Audio extraction from video files
    - EnCodec encoding: Resampling to 24kHz, encoding at 3.0 kbps
    - Mel spectrogram: Resampling to 22,050 Hz, mel computation
    - Temporal aggregation to match fMRI TR (1.5s bins)
    - Audio reconstruction: EnCodec decoder or Griffin-Lim algorithm

    Parameters
    ----------
    use_encodec : bool, default=True
        If True, use EnCodec neural codec (requires transformers package).
        If False or EnCodec unavailable, fall back to mel spectrogram.
    encodec_bandwidth : float, default=3.0
        EnCodec bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0).
        Lower = more compression, higher = better quality.
        3.0 kbps provides STOI~0.74 (sufficient quality).
    sample_rate : int, default=22050
        Target sample rate (Hz) for mel spectrogram mode
    n_mels : int, default=2048
        Number of mel frequency bins (mel spectrogram mode)
    n_fft : int, default=1024
        FFT window size (mel spectrogram mode)
    hop_length : int, default=256
        Number of samples between successive frames (mel spectrogram mode)
    tr : float, default=1.5
        fMRI repetition time in seconds
    device : str, default='cpu'
        Device for EnCodec model ('cpu' or 'cuda')
    """

    def __init__(
        self,
        use_encodec: bool = True,
        encodec_bandwidth: float = 3.0,
        sample_rate: int = 22050,  # Standard for speech/music
        n_mels: int = 2048,        # Very high resolution for detail preservation
        n_fft: int = 4096,         # Larger FFT for frequency resolution
        hop_length: int = 512,     # Frame advance
        tr: float = 1.5,
        device: str = 'cpu'
    ):
        self.use_encodec = use_encodec and ENCODEC_AVAILABLE
        self.encodec_bandwidth = encodec_bandwidth
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tr = tr
        self.n_features = n_mels

        # Initialize EnCodec model if requested
        if self.use_encodec:
            if not ENCODEC_AVAILABLE:
                print("Warning: EnCodec requested but transformers not available. "
                      "Falling back to mel spectrogram mode.")
                self.use_encodec = False
            else:
                print(f"Loading EnCodec model (24kHz, {encodec_bandwidth} kbps)...")
                self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
                self.encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
                self.encodec_sample_rate = 24000  # EnCodec requires 24kHz
                print("EnCodec model loaded successfully.")

    def audio_to_features(
        self,
        audio_source: Union[str, Path],
        max_trs: Optional[int] = None,
        from_video: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert audio to features (EnCodec codes or mel spectrogram) aligned to fMRI TRs.

        Two modes:
        1. EnCodec mode (use_encodec=True): Returns integer codes
        2. Mel spectrogram mode (use_encodec=False): Returns dB-scale mel features

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
            EnCodec mode: Shape (n_trs, n_codebooks, frames_per_tr), dtype int64
                         n_codebooks = 1 for 24kHz mono
                         frames_per_tr = 75 * TR (75 Hz frame rate)
            Mel mode: Shape (n_trs, n_mels, frames_per_tr), dtype float32
        metadata : pd.DataFrame
            DataFrame with: tr_index, start_time, end_time, n_frames, encoding_mode

        Notes
        -----
        EnCodec mode:
        - Resamples to 24kHz mono
        - Encodes with neural codec at specified bandwidth
        - Frame rate: 75 Hz (fixed by EnCodec)
        - Returns integer codebook indices

        Mel spectrogram mode:
        - Uses librosa to extract mel spectrogram
        - Converts to log scale (dB)
        - Preserves temporal frames within each TR
        """
        audio_source = Path(audio_source)

        if self.use_encodec:
            return self._audio_to_features_encodec(audio_source, max_trs, from_video)
        else:
            return self._audio_to_features_mel(audio_source, max_trs, from_video)

    def _audio_to_features_encodec(
        self,
        audio_source: Union[str, Path],
        max_trs: Optional[int] = None,
        from_video: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """EnCodec-based audio encoding."""
        # Load audio
        if from_video:
            y, sr = librosa.load(str(audio_source), sr=self.encodec_sample_rate, mono=False)
            if y.ndim > 1:
                y = np.mean(y, axis=0)
        else:
            y, sr = librosa.load(str(audio_source), sr=self.encodec_sample_rate)

        # Resample if needed
        if sr != self.encodec_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.encodec_sample_rate)
            sr = self.encodec_sample_rate

        duration = len(y) / sr

        # Calculate number of TRs
        n_trs = int(np.floor(duration / self.tr))
        if max_trs is not None:
            n_trs = min(n_trs, max_trs)

        # EnCodec encoding
        inputs = self.encodec_processor(raw_audio=y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            encoded = self.encodec_model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=self.encodec_bandwidth
            )

        # Extract codes: shape is [batch=1, n_codebooks, n_frames]
        codes = encoded.audio_codes[0].cpu()  # [n_codebooks, n_frames]

        # EnCodec frame rate is 75 Hz (fixed)
        encodec_frame_rate = 75.0
        frames_per_tr = int(encodec_frame_rate * self.tr)

        # Align to TRs
        features = []
        tr_metadata = []

        for tr_idx in range(n_trs):
            start_time = tr_idx * self.tr
            end_time = start_time + self.tr

            # Frame indices
            start_frame = int(start_time * encodec_frame_rate)
            end_frame = int(end_time * encodec_frame_rate)

            # Extract frames for this TR
            if end_frame <= codes.shape[1]:
                tr_codes = codes[:, start_frame:end_frame]
            else:
                # Pad if needed
                tr_codes = codes[:, start_frame:]
                padding_needed = frames_per_tr - tr_codes.shape[1]
                if padding_needed > 0:
                    tr_codes = torch.nn.functional.pad(tr_codes, (0, padding_needed), value=0)

            # Ensure consistent shape
            if tr_codes.shape[1] > frames_per_tr:
                tr_codes = tr_codes[:, :frames_per_tr]
            elif tr_codes.shape[1] < frames_per_tr:
                padding = frames_per_tr - tr_codes.shape[1]
                tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)

            features.append(tr_codes)

            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_frames': tr_codes.shape[1],
                'encoding_mode': 'encodec'
            })

        # Stack: (n_trs, n_codebooks, frames_per_tr)
        features = torch.stack(features).numpy().astype(np.int64)
        metadata_df = pd.DataFrame(tr_metadata)

        return features, metadata_df

    def _audio_to_features_mel(
        self,
        audio_source: Union[str, Path],
        max_trs: Optional[int] = None,
        from_video: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Mel spectrogram-based audio encoding (legacy mode)."""
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
                'n_frames': n_frames,
                'encoding_mode': 'mel_spectrogram'
            })

        metadata_df = pd.DataFrame(tr_metadata)

        return features, metadata_df

    def features_to_audio(
        self,
        features: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Reconstruct audio from features (EnCodec codes or mel spectrogram).

        Automatically detects feature format:
        - Integer dtype → EnCodec codes
        - Float dtype → Mel spectrogram

        Parameters
        ----------
        features : np.ndarray
            EnCodec: Shape (n_trs, n_codebooks, frames_per_tr), dtype int64
            Mel: Shape (n_trs, n_mels, frames_per_tr), dtype float32
            Legacy: Shape (n_trs, n_mels), dtype float32 (2D)
        output_path : str or Path
            Path for output audio file

        Notes
        -----
        EnCodec mode:
        - Uses neural decoder for high-quality reconstruction
        - Output sample rate: 24kHz

        Mel spectrogram mode:
        - Uses Griffin-Lim algorithm for phase reconstruction
        - Output sample rate: 22.05kHz (self.sample_rate)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format based on dtype and shape
        if features.dtype in [np.int32, np.int64]:
            # EnCodec codes (integer)
            self._features_to_audio_encodec(features, output_path)
        else:
            # Mel spectrogram (float)
            self._features_to_audio_mel(features, output_path)

    def _features_to_audio_encodec(
        self,
        features: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """Decode EnCodec codes to audio."""
        if not self.use_encodec:
            raise ValueError("EnCodec decoder not available. Use use_encodec=True when initializing AudioProcessor.")

        # Features shape: (n_trs, n_codebooks, frames_per_tr)
        n_trs, n_codebooks, frames_per_tr = features.shape

        # Reshape to (1, n_codebooks, total_frames)
        codes = torch.tensor(features, dtype=torch.long)
        codes = codes.permute(1, 0, 2).reshape(1, n_codebooks, -1).to(self.device)

        # Decode with EnCodec
        with torch.no_grad():
            # EncodedAudioWithBandwidth expects (codes, None)
            # where codes has shape [batch, n_codebooks, n_frames]
            decoded = self.encodec_model.decode(
                [(codes, None)]  # List of tuples (codes, scale)
            )

        # Extract audio: shape is [batch=1, channels=1, samples]
        audio = decoded.audio_values[0, 0].cpu().numpy()

        # Save audio
        sf.write(str(output_path), audio, self.encodec_sample_rate)

    def _features_to_audio_mel(
        self,
        features: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """Decode mel spectrogram to audio using Griffin-Lim."""
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

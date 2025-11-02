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
        from_video: bool = True,
        tr_length: Optional[float] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert audio to features (EnCodec codes or mel spectrogram) aligned to fMRI TRs.

        Two modes:
        1. EnCodec mode (use_encodec=True): Returns flattened integer codes
        2. Mel spectrogram mode (use_encodec=False): Returns dB-scale mel features

        Parameters
        ----------
        audio_source : str or Path
            Path to audio/video file
        max_trs : int, optional
            Maximum number of TRs to extract
        from_video : bool, default=True
            If True, extract audio from video file
        tr_length : float, optional
            TR duration in seconds (overrides self.tr if provided)

        Returns
        -------
        features : np.ndarray
            EnCodec mode: Shape (n_trs, n_codebooks * frames_per_tr), dtype int64
                         Example: TR=1.5s, 8 codebooks, 112 frames → (n_trs, 896)
                         Flattened for consistent dimensions
            Mel mode: Shape (n_trs, n_mels, frames_per_tr), dtype float32
        metadata : pd.DataFrame
            DataFrame with: tr_index, start_time, end_time, n_frames, encoding_mode

        Notes
        -----
        EnCodec mode (temporal concatenation):
        - Resamples to 24kHz mono
        - Encodes with neural codec at specified bandwidth
        - Frame rate: 75 Hz (fixed by EnCodec)
        - Concatenates codes from [t-TR, t] for each TR
        - Flattens to 1D for consistent dimensions across all TRs
        - Returns integer codebook indices

        Mel spectrogram mode:
        - Uses librosa to extract mel spectrogram
        - Converts to log scale (dB)
        - Preserves temporal frames within each TR

        Examples
        --------
        >>> processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)
        >>> features, metadata = processor.audio_to_features('video.mp4', max_trs=100)
        >>> features.shape
        (100, 896)  # 100 TRs × 896 codes (8 codebooks × 112 frames)
        """
        audio_source = Path(audio_source)

        if self.use_encodec:
            return self._audio_to_features_encodec(audio_source, max_trs, from_video, tr_length)
        else:
            return self._audio_to_features_mel(audio_source, max_trs, from_video)

    def _audio_to_features_encodec(
        self,
        audio_source: Union[str, Path],
        max_trs: Optional[int] = None,
        from_video: bool = True,
        tr_length: Optional[float] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        EnCodec-based audio encoding with temporal concatenation.

        For each TR at time t, concatenates EnCodec codes from [t-TR, t].
        This ensures consistent dimensions across all TRs and preserves
        temporal structure.

        Parameters
        ----------
        audio_source : str or Path
            Path to audio/video file
        max_trs : int, optional
            Maximum number of TRs to extract
        from_video : bool, default=True
            If True, extract audio from video file
        tr_length : float, optional
            TR duration in seconds (overrides self.tr if provided)

        Returns
        -------
        features : np.ndarray
            Shape: (n_trs, n_codebooks * frames_per_tr)
            Flattened EnCodec codes for consistent dimensions
            dtype: int64
        metadata : pd.DataFrame
            DataFrame with: tr_index, start_time, end_time, n_frames,
                           n_codebooks, encoding_mode

        Notes
        -----
        EnCodec parameters:
        - Sample rate: 24kHz (self.encodec_sample_rate)
        - Frame rate: 75 Hz (fixed by EnCodec architecture)
        - Bandwidth: self.encodec_bandwidth kbps (default 3.0)
        - Codebooks: Determined by bandwidth (3.0 kbps → 8 codebooks)

        Temporal concatenation:
        - TR=1.5s @ 75Hz → 112 frames per TR
        - 8 codebooks × 112 frames = 896 codes per TR
        - Output: (n_trs, 896) flattened int64 array

        Dimension mismatch fix:
        - Previous issue: Variable codebook counts (4 vs 0) across TRs
        - Solution: Enforce consistent codebook count using bandwidth setting
        - All TRs now have same shape regardless of audio content
        """
        # Use provided TR length or fall back to instance default
        tr_length = tr_length if tr_length is not None else self.tr

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
        n_trs = int(np.floor(duration / tr_length))
        if max_trs is not None:
            n_trs = min(n_trs, max_trs)

        # EnCodec encoding with explicit bandwidth setting
        # This ensures consistent codebook count across all frames
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

        # Determine expected codebook count based on bandwidth
        # EnCodec 24kHz model: 3.0 kbps → 8 codebooks
        # See: https://github.com/facebookresearch/encodec
        bandwidth_to_codebooks = {
            1.5: 2,
            3.0: 8,
            6.0: 16,
            12.0: 32,
            24.0: 32
        }
        expected_codebooks = bandwidth_to_codebooks.get(self.encodec_bandwidth, codes.shape[0])

        # EnCodec frame rate is 75 Hz (fixed)
        encodec_frame_rate = 75.0
        frames_per_tr = int(encodec_frame_rate * tr_length)

        # Align to TRs with temporal concatenation
        features = []
        tr_metadata = []

        for tr_idx in range(n_trs):
            # Temporal window: [t-TR, t]
            start_time = tr_idx * tr_length
            end_time = start_time + tr_length

            # Frame indices
            start_frame = int(start_time * encodec_frame_rate)
            end_frame = int(end_time * encodec_frame_rate)

            # Extract frames for this TR window
            if end_frame <= codes.shape[1]:
                tr_codes = codes[:, start_frame:end_frame]
            else:
                # Pad if needed (end of audio)
                tr_codes = codes[:, start_frame:]
                padding_needed = frames_per_tr - tr_codes.shape[1]
                if padding_needed > 0:
                    tr_codes = torch.nn.functional.pad(tr_codes, (0, padding_needed), value=0)

            # Ensure consistent temporal dimension FIRST
            if tr_codes.shape[1] > frames_per_tr:
                tr_codes = tr_codes[:, :frames_per_tr]
            elif tr_codes.shape[1] < frames_per_tr:
                padding = frames_per_tr - tr_codes.shape[1]
                tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)

            # CRITICAL FIX: Ensure consistent codebook dimension
            # Must be done AFTER temporal dimension is fixed
            # This fixes the "RuntimeError: The expanded size of the tensor (112) must match
            # the existing size (106697)" error
            if tr_codes.shape[0] != expected_codebooks:
                # Create properly shaped tensor with KNOWN correct temporal dimension
                # Use frames_per_tr directly, NOT tr_codes.shape[1], because tr_codes
                # was already normalized to frames_per_tr in the previous step (lines 300-305)
                normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)
                # Copy available codebooks (pad with zeros if fewer, crop if more)
                n_available = min(tr_codes.shape[0], expected_codebooks)
                # Both tensors now have matching temporal dimension (frames_per_tr)
                normalized_codes[:n_available, :] = tr_codes[:n_available, :]
                tr_codes = normalized_codes

            # Flatten to 1D: (n_codebooks, frames_per_tr) → (n_codebooks * frames_per_tr,)
            # This ensures consistent dimensions for torch.stack() and training
            tr_codes_flat = tr_codes.reshape(-1)

            features.append(tr_codes_flat)

            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_frames': frames_per_tr,
                'n_codebooks': expected_codebooks,
                'encoding_mode': 'encodec'
            })

        # Stack: (n_trs, n_codebooks * frames_per_tr)
        # All TRs now have identical shape
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
        output_path: Union[str, Path],
        n_codebooks: Optional[int] = None,
        frames_per_tr: Optional[int] = None
    ) -> None:
        """
        Reconstruct audio from features (EnCodec codes or mel spectrogram).

        Automatically detects feature format:
        - Integer dtype → EnCodec codes
        - Float dtype → Mel spectrogram

        Parameters
        ----------
        features : np.ndarray
            EnCodec (new): Shape (n_trs, n_codebooks * frames_per_tr), dtype int64
                          Flattened codes from temporal concatenation
            EnCodec (old): Shape (n_trs, n_codebooks, frames_per_tr), dtype int64
                          Legacy 3D format (still supported)
            Mel: Shape (n_trs, n_mels, frames_per_tr), dtype float32
            Legacy: Shape (n_trs, n_mels), dtype float32 (2D)
        output_path : str or Path
            Path for output audio file
        n_codebooks : int, optional
            Number of codebooks (required for flattened EnCodec format)
            If not provided, will attempt to infer from bandwidth setting
        frames_per_tr : int, optional
            Frames per TR (required for flattened EnCodec format)
            If not provided, will use default based on self.tr

        Notes
        -----
        EnCodec mode:
        - Uses neural decoder for high-quality reconstruction
        - Output sample rate: 24kHz
        - Supports both flattened (2D) and legacy (3D) formats

        Mel spectrogram mode:
        - Uses Griffin-Lim algorithm for phase reconstruction
        - Output sample rate: 22.05kHz (self.sample_rate)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format based on dtype and shape
        if features.dtype in [np.int32, np.int64]:
            # EnCodec codes (integer)
            self._features_to_audio_encodec(features, output_path, n_codebooks, frames_per_tr)
        else:
            # Mel spectrogram (float)
            self._features_to_audio_mel(features, output_path)

    def _features_to_audio_encodec(
        self,
        features: np.ndarray,
        output_path: Union[str, Path],
        n_codebooks: Optional[int] = None,
        frames_per_tr: Optional[int] = None
    ) -> None:
        """
        Decode EnCodec codes to audio.

        Supports both flattened (2D) and legacy (3D) formats.

        Parameters
        ----------
        features : np.ndarray
            Shape: (n_trs, n_codebooks * frames_per_tr) [new flattened format]
                or (n_trs, n_codebooks, frames_per_tr) [legacy 3D format]
        output_path : str or Path
            Output audio file path
        n_codebooks : int, optional
            Number of codebooks (for 2D format). If not provided, inferred from bandwidth.
        frames_per_tr : int, optional
            Frames per TR (for 2D format). If not provided, calculated from self.tr.
        """
        if not self.use_encodec:
            raise ValueError("EnCodec decoder not available. Use use_encodec=True when initializing AudioProcessor.")

        # Detect format: 2D (flattened) or 3D (legacy)
        if features.ndim == 2:
            # New flattened format: (n_trs, n_codebooks * frames_per_tr)
            n_trs, flat_dim = features.shape

            # Infer n_codebooks if not provided
            if n_codebooks is None:
                bandwidth_to_codebooks = {
                    1.5: 2,
                    3.0: 8,
                    6.0: 16,
                    12.0: 32,
                    24.0: 32
                }
                n_codebooks = bandwidth_to_codebooks.get(self.encodec_bandwidth, 8)

            # Infer frames_per_tr if not provided
            if frames_per_tr is None:
                encodec_frame_rate = 75.0
                frames_per_tr = int(encodec_frame_rate * self.tr)

            # Verify dimensions
            expected_flat_dim = n_codebooks * frames_per_tr
            if flat_dim != expected_flat_dim:
                # Try to infer from actual dimension
                if flat_dim % frames_per_tr == 0:
                    n_codebooks = flat_dim // frames_per_tr
                elif flat_dim % n_codebooks == 0:
                    frames_per_tr = flat_dim // n_codebooks
                else:
                    raise ValueError(
                        f"Cannot reshape features with dim {flat_dim} into "
                        f"({n_codebooks}, {frames_per_tr}). "
                        f"Expected: {expected_flat_dim}"
                    )

            # Reshape to 3D: (n_trs, flat_dim) → (n_trs, n_codebooks, frames_per_tr)
            features_3d = features.reshape(n_trs, n_codebooks, frames_per_tr)

        elif features.ndim == 3:
            # Legacy 3D format: (n_trs, n_codebooks, frames_per_tr)
            features_3d = features
            n_trs, n_codebooks, frames_per_tr = features.shape

        else:
            raise ValueError(f"Expected 2D or 3D features, got shape {features.shape}")

        # Reshape to (1, n_codebooks, total_frames)
        codes = torch.tensor(features_3d, dtype=torch.long)
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

"""
Decoder module for multimodal fMRI autoencoder.

Architecture mirrors the encoder in reverse (Layers 8-13):
- Layer 8: Expand from bottleneck (2048 → 8000)
- Layer 9: Feature expansion (8000 → 4096)
- Layer 10: Feature deconvolution + ReLU (4096 → 2048)
- Layer 11: Unpool features (2048 → 1536)
- Layer 12A/B/C: Separate video/audio/text paths
- Layer 13: Output video + audio + text

Input: 2048-dimensional bottleneck features
Output: Separate video (43,200), audio (2048), text (1024) features
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultimodalDecoder(nn.Module):
    """
    Decoder for reconstructing video, audio, and text from fMRI features.

    Symmetric architecture to the encoder, taking bottleneck features and
    expanding them through multiple layers to reconstruct the original
    video frames, audio mel spectrograms, and text embeddings.

    Architecture (Layers 8-13):
    - Layer 8: 2048 → 8000 (mirror Encoder Layer 6)
    - Layer 9: 8000 → 4096 (mirror Encoder Layer 5)
    - Layer 10: 4096 → 2048 (mirror Encoder Layer 4)
    - Layer 11: 2048 → 1536 (mirror Encoder Layer 3)
    - Layer 12A/B/C: Modality decoders (mirror Encoder Layer 2A/B/C)
    - Layer 13: Output reconstruction (mirror Encoder Layer 1)

    Parameters
    ----------
    bottleneck_dim : int, default=2048
        Dimension of the bottleneck/middle layer (Layer 7)
    video_dim : int, default=43200
        Output dimension for video (160×90×3 = 43,200)
    audio_dim : int, default=2048
        Output dimension for audio (2048 mels for mel spectrograms)
    audio_frames_per_tr : int, default=65
        Number of temporal frames per TR (65 for mel @ 44Hz, 112 for EnCodec @ 75Hz)
    text_dim : int, default=1024
        Output dimension for text (1024 embeddings)
    dropout : float, default=0.3
        Dropout rate for regularization
    use_encodec : bool, default=False
        If True, predict EnCodec discrete codes; if False, predict mel spectrograms
    n_codebooks : int, default=8
        Number of EnCodec codebooks (only used if use_encodec=True)

    Attributes
    ----------
    layer8 : nn.Sequential
        Expansion from bottleneck (2048 → 8000)
    layer9 : nn.Sequential
        Feature expansion (8000 → 4096)
    layer10 : nn.Sequential
        Feature deconvolution (4096 → 2048)
    layer11 : nn.Sequential
        Feature unpooling (2048 → 1536)
    layer12_video : nn.Sequential
        Video-specific decoder path
    layer12_audio : nn.Sequential
        Audio-specific decoder path
    layer12_text : nn.Sequential
        Text-specific decoder path
    layer13_video : nn.Sequential
        Final video output layer
    layer13_audio : nn.Linear
        Final audio output layer
    layer13_text : nn.Linear
        Final text output layer

    Examples
    --------
    >>> decoder = MultimodalDecoder(bottleneck_dim=2048)
    >>> bottleneck = torch.randn(32, 2048)  # batch_size=32
    >>> video, audio, text = decoder(bottleneck)
    >>> video.shape, audio.shape, text.shape
    (torch.Size([32, 43200]), torch.Size([32, 2048]), torch.Size([32, 1024]))
    """

    def __init__(
        self,
        bottleneck_dim: int = 2048,
        video_dim: int = 43200,
        audio_dim: int = 2048,
        audio_frames_per_tr: int = 65,  # NEW: temporal frames per TR
        text_dim: int = 1024,
        dropout: float = 0.3,
        use_encodec: bool = False,  # NEW: Use EnCodec discrete codes
        n_codebooks: int = 8,  # NEW: Number of EnCodec codebooks
    ):
        super().__init__()

        self.bottleneck_dim = bottleneck_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.audio_frames_per_tr = audio_frames_per_tr
        self.text_dim = text_dim
        self.use_encodec = use_encodec
        self.n_codebooks = n_codebooks

        # Layer 8: Expand from bottleneck (2048 → 8000)
        # Mirror of Encoder Layer 6 (8000 → 2048)
        self.layer8 = nn.Sequential(
            nn.Linear(2048, 8000), nn.BatchNorm1d(8000), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 9: Feature expansion (8000 → 4096)
        # Mirror of Encoder Layer 5 (4096 → 8000)
        self.layer9 = nn.Sequential(
            nn.Linear(8000, 4096), nn.BatchNorm1d(4096), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 10: Feature deconvolution (4096 → 2048)
        # Mirror of Encoder Layer 4 (2048 → 4096)
        self.layer10 = nn.Sequential(
            nn.Linear(4096, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 11: Unpool features (2048 → 1536)
        # Mirror of Encoder Layer 3 (1536 → 2048)
        self.layer11 = nn.Sequential(
            nn.Linear(2048, 1536), nn.BatchNorm1d(1536), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 12A/B/C: Modality-specific decoder paths
        # Mirror of Encoder Layer 2A/B/C

        # Layer 12A: Video decoder path (1536 → video features)
        # Needs large capacity for 43,200 outputs
        self.layer12_video = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 12B: Audio decoder path (1536 → audio features)
        # UPDATED: Includes temporal upsampling for 3D output
        self.layer12_audio = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # EnCodec-specific or Mel-specific temporal processing
        if use_encodec:
            # For EnCodec: Predict discrete codes directly
            # No temporal upsampling needed - predict codes at target resolution
            self.audio_temporal_init_frames = None
            self.audio_temporal_upsample = None
            self.audio_temporal_adjust = None
        else:
            # For Mel spectrograms: Temporal upsampling for 3D output
            # Start with 8 frames, upsample to 16, 32, then target frames_per_tr
            # Calculate number of upsampling stages needed
            self.audio_temporal_init_frames = 8
            self.audio_temporal_upsample = nn.Sequential(
                # 8 → 16 frames
                nn.ConvTranspose1d(
                    audio_dim, audio_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm1d(audio_dim),
                nn.ReLU(),
                # 16 → 32 frames
                nn.ConvTranspose1d(
                    audio_dim, audio_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm1d(audio_dim),
                nn.ReLU(),
                # 32 → 64 frames (close to target ~65)
                nn.ConvTranspose1d(
                    audio_dim, audio_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm1d(audio_dim),
                nn.ReLU(),
            )
            # Final adjustment layer to get exact frames_per_tr
            self.audio_temporal_adjust = nn.Conv1d(
                audio_dim, audio_dim, kernel_size=3, padding=1
            )

        # Layer 12C: Text decoder path (1536 → text features)
        # Moderate size for 1024 embeddings
        self.layer12_text = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 13: Output reconstruction
        # Mirror of Encoder Layer 1

        # Layer 13A: Video output (4096 → 43,200)
        # Sigmoid for [0, 1] pixel range
        self.layer13_video = nn.Sequential(nn.Linear(4096, video_dim), nn.Sigmoid())

        # Layer 13B: Audio output
        # EnCodec: Outputs discrete codes (n_codebooks × frames_per_tr)
        # Mel: Outputs initial temporal features that will be upsampled
        if use_encodec:
            # Predict EnCodec codes: 8 codebooks × frames_per_tr
            # Update frames_per_tr to 112 for EnCodec (75 Hz × 1.5s)
            if audio_frames_per_tr == 65:
                # Default mel value, update for EnCodec
                self.audio_frames_per_tr = 112
            self.layer13_audio = nn.Linear(2048, n_codebooks * self.audio_frames_per_tr)
        else:
            # Predict mel spectrogram temporal features
            self.layer13_audio = nn.Linear(
                2048, audio_dim * self.audio_temporal_init_frames
            )

        # Layer 13C: Text output (1024 → 1024)
        # No activation (normalized in loss/post-processing)
        self.layer13_text = nn.Linear(1024, text_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, bottleneck: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, 2048) bottleneck features from Layer 7

        Returns
        -------
        video : torch.Tensor
            Shape (batch_size, video_dim) reconstructed video features
        audio : torch.Tensor
            If use_encodec=True: Shape (batch_size, n_codebooks, frames_per_tr)
                EnCodec codes in [0, 1023] range (continuous during training, discrete during inference)
            If use_encodec=False: Shape (batch_size, audio_dim, frames_per_tr)
                Mel spectrogram features (continuous)
        text : torch.Tensor
            Shape (batch_size, text_dim) reconstructed text features
        """
        # Layer 8: Expand from bottleneck (2048 → 8000)
        x = self.layer8(bottleneck)

        # Layer 9: Feature expansion (8000 → 4096)
        x = self.layer9(x)

        # Layer 10: Feature deconvolution (4096 → 2048)
        x = self.layer10(x)

        # Layer 11: Unpool features (2048 → 1536)
        x = self.layer11(x)

        # Layer 12A/B/C: Separate modality decoder paths
        video_features = self.layer12_video(x)  # 1536 → 4096
        audio_features = self.layer12_audio(x)  # 1536 → 2048
        text_features = self.layer12_text(x)  # 1536 → 1024

        # Layer 13: Output reconstruction
        video = self.layer13_video(video_features)  # 4096 → 43,200

        # Audio decoding: EnCodec or Mel spectrogram
        if self.use_encodec:
            # EnCodec: Predict discrete codes
            audio = self.layer13_audio(
                audio_features
            )  # 2048 → n_codebooks * frames_per_tr
            batch_size = audio.size(0)
            audio = audio.view(
                batch_size, self.n_codebooks, self.audio_frames_per_tr
            )  # (B, 8, 112)

            # Scale to valid code range [0, 1023]
            # Use sigmoid to map to [0, 1], then scale to [0, 1023]
            audio = torch.sigmoid(audio) * 1023.0

            # During inference, round to nearest integer
            if not self.training:
                audio = torch.round(audio)
                # Clip to ensure valid range (defensive)
                audio = torch.clamp(audio, 0, 1023)

            # Squeeze temporal dimension for single-frame audio
            if self.audio_frames_per_tr == 1:
                audio = audio.squeeze(2)  # (B, n_codebooks, 1) → (B, n_codebooks)
        else:
            # Mel spectrogram: Temporal upsampling
            audio = self.layer13_audio(audio_features)  # 2048 → audio_dim * init_frames
            batch_size = audio.size(0)
            audio = audio.view(
                batch_size, self.audio_dim, self.audio_temporal_init_frames
            )  # (B, mels, 8)
            audio = self.audio_temporal_upsample(audio)  # (B, mels, 64)
            audio = self.audio_temporal_adjust(audio)  # (B, mels, 64)

            # Crop or pad to exact frames_per_tr
            current_frames = audio.size(2)
            if current_frames > self.audio_frames_per_tr:
                audio = audio[:, :, : self.audio_frames_per_tr]
            elif current_frames < self.audio_frames_per_tr:
                padding = self.audio_frames_per_tr - current_frames
                audio = torch.nn.functional.pad(audio, (0, padding))

            # Squeeze temporal dimension for single-frame audio
            if self.audio_frames_per_tr == 1:
                audio = audio.squeeze(2)  # (B, mels, 1) → (B, mels)

        text = self.layer13_text(text_features)  # 1024 → 1,024

        return video, audio, text

    def decode_video_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only video, skipping audio and text paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, 2048) bottleneck features

        Returns
        -------
        video : torch.Tensor
            Shape (batch_size, video_dim) reconstructed video features
        """
        x = self.layer8(bottleneck)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        video_features = self.layer12_video(x)
        video = self.layer13_video(video_features)
        return video

    def decode_audio_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only audio, skipping video and text paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, 2048) bottleneck features

        Returns
        -------
        audio : torch.Tensor
            If use_encodec=True: Shape (batch_size, n_codebooks, frames_per_tr) EnCodec codes
            If use_encodec=False: Shape (batch_size, audio_dim, frames_per_tr) mel spectrograms
        """
        x = self.layer8(bottleneck)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        audio_features = self.layer12_audio(x)

        # Audio decoding: EnCodec or Mel
        if self.use_encodec:
            # EnCodec: Predict discrete codes
            audio = self.layer13_audio(audio_features)
            batch_size = audio.size(0)
            audio = audio.view(batch_size, self.n_codebooks, self.audio_frames_per_tr)
            audio = torch.sigmoid(audio) * 1023.0

            if not self.training:
                audio = torch.round(audio)
                audio = torch.clamp(audio, 0, 1023)
        else:
            # Mel: Temporal upsampling
            audio = self.layer13_audio(audio_features)
            batch_size = audio.size(0)
            audio = audio.view(
                batch_size, self.audio_dim, self.audio_temporal_init_frames
            )
            audio = self.audio_temporal_upsample(audio)
            audio = self.audio_temporal_adjust(audio)

            # Crop or pad to exact frames_per_tr
            current_frames = audio.size(2)
            if current_frames > self.audio_frames_per_tr:
                audio = audio[:, :, : self.audio_frames_per_tr]
            elif current_frames < self.audio_frames_per_tr:
                padding = self.audio_frames_per_tr - current_frames
                audio = torch.nn.functional.pad(audio, (0, padding))

        return audio

    def decode_text_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only text, skipping video and audio paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, 2048) bottleneck features

        Returns
        -------
        text : torch.Tensor
            Shape (batch_size, text_dim) reconstructed text features
        """
        x = self.layer8(bottleneck)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        text_features = self.layer12_text(x)
        text = self.layer13_text(text_features)
        return text

    def get_layer_outputs(self, bottleneck: torch.Tensor) -> dict:
        """
        Get intermediate outputs from all layers for analysis/debugging.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, 2048) bottleneck features

        Returns
        -------
        outputs : dict
            Dictionary with keys: 'layer8', 'layer9', 'layer10', 'layer11',
            'layer12_video', 'layer12_audio', 'layer12_text',
            'video', 'audio', 'text'
        """
        outputs = {}

        # Layer 8
        x = self.layer8(bottleneck)
        outputs["layer8"] = x.detach()

        # Layer 9
        x = self.layer9(x)
        outputs["layer9"] = x.detach()

        # Layer 10
        x = self.layer10(x)
        outputs["layer10"] = x.detach()

        # Layer 11
        x = self.layer11(x)
        outputs["layer11"] = x.detach()

        # Layer 12A/B/C
        video_features = self.layer12_video(x)
        audio_features = self.layer12_audio(x)
        text_features = self.layer12_text(x)
        outputs["layer12_video"] = video_features.detach()
        outputs["layer12_audio"] = audio_features.detach()
        outputs["layer12_text"] = text_features.detach()

        # Layer 13
        video = self.layer13_video(video_features)
        audio = self.layer13_audio(audio_features)
        text = self.layer13_text(text_features)
        outputs["video"] = video.detach()
        outputs["audio"] = audio.detach()
        outputs["text"] = text.detach()

        return outputs

    def count_parameters(self) -> dict:
        """
        Count trainable parameters in each component.

        Returns
        -------
        param_counts : dict
            Dictionary with parameter counts for each layer/path
        """

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "layer8": count_params(self.layer8),
            "layer9": count_params(self.layer9),
            "layer10": count_params(self.layer10),
            "layer11": count_params(self.layer11),
            "layer12_video": count_params(self.layer12_video),
            "layer12_audio": count_params(self.layer12_audio),
            "layer12_text": count_params(self.layer12_text),
            "layer13_video": count_params(self.layer13_video),
            "layer13_audio": count_params(self.layer13_audio),
            "layer13_text": count_params(self.layer13_text),
            "total": count_params(self),
        }

"""
Decoder module for multimodal fMRI autoencoder.

Architecture mirrors the encoder in reverse (Layers 7-11):
- Layer 7: Expand from bottleneck
- Layer 8: Feature deconvolution + ReLU
- Layer 9: Unpool features
- Layer 10A/B/C: Separate video/audio/text paths
- Layer 11: Output video + audio + text

Input: fMRI voxel-space features (~5000-10000 dimensions)
Output: Separate video (43,200), audio (128), text (1024) features
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

    Parameters
    ----------
    bottleneck_dim : int
        Dimension of the bottleneck/middle layer (fMRI feature space)
    video_dim : int, default=43200
        Output dimension for video (160×90×3 = 43,200)
    audio_dim : int, default=128
        Output dimension for audio (128 mels)
    text_dim : int, default=1024
        Output dimension for text (1024 embeddings)
    hidden_dim : int, default=2048
        Hidden layer dimension for intermediate processing
    dropout : float, default=0.3
        Dropout rate for regularization

    Attributes
    ----------
    layer7 : nn.Sequential
        Expansion from bottleneck
    layer8 : nn.Sequential
        Feature deconvolution with ReLU
    layer9 : nn.Sequential
        Feature unpooling
    layer10_video : nn.Sequential
        Video-specific path
    layer10_audio : nn.Sequential
        Audio-specific path
    layer10_text : nn.Sequential
        Text-specific path
    layer11_video : nn.Linear
        Final video output layer
    layer11_audio : nn.Linear
        Final audio output layer
    layer11_text : nn.Linear
        Final text output layer

    Examples
    --------
    >>> decoder = MultimodalDecoder(bottleneck_dim=5000)
    >>> bottleneck = torch.randn(32, 5000)  # batch_size=32
    >>> video, audio, text = decoder(bottleneck)
    >>> video.shape, audio.shape, text.shape
    (torch.Size([32, 43200]), torch.Size([32, 128]), torch.Size([32, 1024]))
    """

    def __init__(
        self,
        bottleneck_dim: int,
        video_dim: int = 43200,
        audio_dim: int = 128,
        text_dim: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.3
    ):
        super().__init__()

        self.bottleneck_dim = bottleneck_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Calculate intermediate dimensions
        # Layer 9 output -> Layer 8 input
        intermediate_dim_1 = hidden_dim * 2  # ~4096
        # Layer 8 output -> Layer 7 input
        intermediate_dim_2 = hidden_dim * 4  # ~8192

        # Layer 7: Expand from bottleneck
        # Maps from fMRI voxel space to larger feature space
        self.layer7 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 8: Feature deconvolution + ReLU
        # Expands features further with nonlinear transformation
        self.layer8 = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim_1),
            nn.BatchNorm1d(intermediate_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 9: Unpool features
        # Further expansion before modality split
        self.layer9 = nn.Sequential(
            nn.Linear(intermediate_dim_1, intermediate_dim_2),
            nn.BatchNorm1d(intermediate_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 10A/B/C: Separate paths for each modality
        # Video path: needs large capacity for 43,200 outputs
        self.layer10_video = nn.Sequential(
            nn.Linear(intermediate_dim_2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Audio path: smaller, targeted for 128 mel features
        self.layer10_audio = nn.Sequential(
            nn.Linear(intermediate_dim_2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Text path: moderate size for 1024 embeddings
        self.layer10_text = nn.Sequential(
            nn.Linear(intermediate_dim_2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 11: Output layers for each modality
        # Video: output 43,200 features with sigmoid for [0, 1] range
        self.layer11_video = nn.Sequential(
            nn.Linear(hidden_dim * 2, video_dim),
            nn.Sigmoid()  # Video pixels in [0, 1] range
        )

        # Audio: output 128 mel features (no activation, dB scale)
        self.layer11_audio = nn.Linear(hidden_dim // 2, audio_dim)

        # Text: output 1024 embeddings (normalized in loss/post-processing)
        self.layer11_text = nn.Linear(hidden_dim, text_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        bottleneck: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) bottleneck features

        Returns
        -------
        video : torch.Tensor
            Shape (batch_size, video_dim) reconstructed video features
        audio : torch.Tensor
            Shape (batch_size, audio_dim) reconstructed audio features
        text : torch.Tensor
            Shape (batch_size, text_dim) reconstructed text features
        """
        # Layer 7: Expand from bottleneck
        x = self.layer7(bottleneck)

        # Layer 8: Feature deconvolution + ReLU
        x = self.layer8(x)

        # Layer 9: Unpool features
        x = self.layer9(x)

        # Layer 10A/B/C: Separate modality paths
        video_features = self.layer10_video(x)
        audio_features = self.layer10_audio(x)
        text_features = self.layer10_text(x)

        # Layer 11: Generate outputs
        video = self.layer11_video(video_features)
        audio = self.layer11_audio(audio_features)
        text = self.layer11_text(text_features)

        return video, audio, text

    def decode_video_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only video, skipping audio and text paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) bottleneck features

        Returns
        -------
        video : torch.Tensor
            Shape (batch_size, video_dim) reconstructed video features
        """
        x = self.layer7(bottleneck)
        x = self.layer8(x)
        x = self.layer9(x)
        video_features = self.layer10_video(x)
        video = self.layer11_video(video_features)
        return video

    def decode_audio_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only audio, skipping video and text paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) bottleneck features

        Returns
        -------
        audio : torch.Tensor
            Shape (batch_size, audio_dim) reconstructed audio features
        """
        x = self.layer7(bottleneck)
        x = self.layer8(x)
        x = self.layer9(x)
        audio_features = self.layer10_audio(x)
        audio = self.layer11_audio(audio_features)
        return audio

    def decode_text_only(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Decode only text, skipping video and audio paths for efficiency.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) bottleneck features

        Returns
        -------
        text : torch.Tensor
            Shape (batch_size, text_dim) reconstructed text features
        """
        x = self.layer7(bottleneck)
        x = self.layer8(x)
        x = self.layer9(x)
        text_features = self.layer10_text(x)
        text = self.layer11_text(text_features)
        return text

    def get_layer_outputs(
        self,
        bottleneck: torch.Tensor
    ) -> dict:
        """
        Get intermediate outputs from all layers for analysis/debugging.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) bottleneck features

        Returns
        -------
        outputs : dict
            Dictionary with keys: 'layer7', 'layer8', 'layer9',
            'layer10_video', 'layer10_audio', 'layer10_text',
            'video', 'audio', 'text'
        """
        outputs = {}

        # Layer 7
        x = self.layer7(bottleneck)
        outputs['layer7'] = x.detach()

        # Layer 8
        x = self.layer8(x)
        outputs['layer8'] = x.detach()

        # Layer 9
        x = self.layer9(x)
        outputs['layer9'] = x.detach()

        # Layer 10A/B/C
        video_features = self.layer10_video(x)
        audio_features = self.layer10_audio(x)
        text_features = self.layer10_text(x)
        outputs['layer10_video'] = video_features.detach()
        outputs['layer10_audio'] = audio_features.detach()
        outputs['layer10_text'] = text_features.detach()

        # Layer 11
        video = self.layer11_video(video_features)
        audio = self.layer11_audio(audio_features)
        text = self.layer11_text(text_features)
        outputs['video'] = video.detach()
        outputs['audio'] = audio.detach()
        outputs['text'] = text.detach()

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
            'layer7': count_params(self.layer7),
            'layer8': count_params(self.layer8),
            'layer9': count_params(self.layer9),
            'layer10_video': count_params(self.layer10_video),
            'layer10_audio': count_params(self.layer10_audio),
            'layer10_text': count_params(self.layer10_text),
            'layer11_video': count_params(self.layer11_video),
            'layer11_audio': count_params(self.layer11_audio),
            'layer11_text': count_params(self.layer11_text),
            'total': count_params(self)
        }

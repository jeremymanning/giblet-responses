"""
Encoder module for multimodal fMRI autoencoder.

Implements the encoder half of the autoencoder architecture that maps
stimulus features (video + audio + text) to brain activity (fMRI voxels).

Architecture follows issue #2 specification:
- Layer 1: Input (video + audio + text concatenated after processing)
- Layer 2A: Video convolutions (Conv2D to reduce spatial dimensions)
- Layer 2B: Audio convolutions (Conv1D on mel spectrogram)
- Layer 2C: Text linear mapping (Linear on embeddings)
- Layer 3: Pool all features
- Layer 4: Feature space convolution + ReLU
- Layer 5: Linear mapping to 85,810 brain voxels
- Layer 6: Bottleneck convolution (middle layer ~5000-10000 dims)

This module implements Layers 1-6 of the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VideoEncoder(nn.Module):
    """
    Encode video frames using 2D convolutions.

    Reduces 160×90×3 frames to a lower-dimensional representation.

    Parameters
    ----------
    input_channels : int, default=3
        Number of input channels (RGB)
    input_height : int, default=90
        Frame height
    input_width : int, default=160
        Frame width
    output_features : int, default=1024
        Dimensionality of output features
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 90,
        input_width: int = 160,
        output_features: int = 1024
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_features = output_features

        # Convolutional layers to reduce spatial dimensions
        # 160×90×3 → 80×45×32 → 40×23×64 → 20×12×128 → 10×6×256
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Calculate flattened size after convolutions
        # After 4 stride-2 convolutions: height // 16, width // 16
        self.flat_height = (input_height + 15) // 16  # Round up
        self.flat_width = (input_width + 15) // 16
        self.flat_features = 256 * self.flat_height * self.flat_width

        # Linear layer to compress to output features
        self.fc = nn.Linear(self.flat_features, output_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through video encoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, 3, height, width) video frames

        Returns
        -------
        features : torch.Tensor
            Shape (batch_size, output_features) encoded features
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear projection
        x = self.fc(x)
        x = self.dropout(x)
        x = F.relu(x)

        return x


class AudioEncoder(nn.Module):
    """
    Encode audio mel spectrograms using 1D convolutions.

    Processes 128 mel bins to extract audio features.

    Parameters
    ----------
    input_mels : int, default=128
        Number of mel frequency bins
    output_features : int, default=256
        Dimensionality of output features
    """

    def __init__(
        self,
        input_mels: int = 128,
        output_features: int = 256
    ):
        super().__init__()

        self.input_mels = input_mels
        self.output_features = output_features

        # 1D convolutions over frequency dimension
        # 128 → 64 → 32 → 16
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Calculate flattened size
        self.flat_length = (input_mels + 7) // 8  # After 3 stride-2 convs
        self.flat_features = 128 * self.flat_length

        # Linear projection
        self.fc = nn.Linear(self.flat_features, output_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through audio encoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_mels) mel spectrogram

        Returns
        -------
        features : torch.Tensor
            Shape (batch_size, output_features) encoded features
        """
        # Reshape for 1D conv: (batch, 1, mels)
        x = x.unsqueeze(1)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear projection
        x = self.fc(x)
        x = self.dropout(x)
        x = F.relu(x)

        return x


class TextEncoder(nn.Module):
    """
    Encode text embeddings using linear layers.

    Maps 1024-dim embeddings to lower-dimensional representation.

    Parameters
    ----------
    input_dim : int, default=1024
        Dimensionality of input embeddings (BGE-large-en-v1.5)
    output_features : int, default=256
        Dimensionality of output features
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_features: int = 256
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_features = output_features

        # Linear layers for dimensionality reduction
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text encoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim) text embeddings

        Returns
        -------
        features : torch.Tensor
            Shape (batch_size, output_features) encoded features
        """
        # First linear layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second linear layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return x


class MultimodalEncoder(nn.Module):
    """
    Full encoder for multimodal fMRI autoencoder.

    Maps stimulus features (video + audio + text) to brain activity (fMRI).

    Architecture:
    - Layer 1: Input (video + audio + text)
    - Layer 2A/B/C: Modality-specific encoders (convolutions/linear)
    - Layer 3: Pooled multimodal features
    - Layer 4: Feature space convolution + ReLU
    - Layer 5: Linear mapping to brain voxels
    - Layer 6: Bottleneck (compressed brain representation)

    Parameters
    ----------
    video_height : int, default=90
        Video frame height
    video_width : int, default=160
        Video frame width
    audio_mels : int, default=128
        Number of mel frequency bins
    text_dim : int, default=1024
        Dimensionality of text embeddings
    n_voxels : int, default=85810
        Number of brain voxels
    bottleneck_dim : int, default=8000
        Dimensionality of bottleneck layer (middle layer)
    video_features : int, default=1024
        Video encoder output features
    audio_features : int, default=256
        Audio encoder output features
    text_features : int, default=256
        Text encoder output features
    """

    def __init__(
        self,
        video_height: int = 90,
        video_width: int = 160,
        audio_mels: int = 128,
        text_dim: int = 1024,
        n_voxels: int = 85810,
        bottleneck_dim: int = 8000,
        video_features: int = 1024,
        audio_features: int = 256,
        text_features: int = 256
    ):
        super().__init__()

        self.video_height = video_height
        self.video_width = video_width
        self.audio_mels = audio_mels
        self.text_dim = text_dim
        self.n_voxels = n_voxels
        self.bottleneck_dim = bottleneck_dim

        # Layer 2A: Video encoder
        self.video_encoder = VideoEncoder(
            input_height=video_height,
            input_width=video_width,
            output_features=video_features
        )

        # Layer 2B: Audio encoder
        self.audio_encoder = AudioEncoder(
            input_mels=audio_mels,
            output_features=audio_features
        )

        # Layer 2C: Text encoder
        self.text_encoder = TextEncoder(
            input_dim=text_dim,
            output_features=text_features
        )

        # Layer 3: Pooled features dimension
        self.pooled_dim = video_features + audio_features + text_features

        # Layer 4: Feature space convolution (1D conv over concatenated features)
        # This acts as a learned weighted combination of modalities
        self.feature_conv = nn.Sequential(
            nn.Linear(self.pooled_dim, self.pooled_dim),
            nn.BatchNorm1d(self.pooled_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Layer 5 + 6: Direct compression to bottleneck, then expansion to voxels
        # This is more parameter-efficient than going pooled → voxels → bottleneck
        # Following autoencoder principle: compress first, then expand if needed

        # Compress to bottleneck (middle layer)
        self.to_bottleneck = nn.Sequential(
            nn.Linear(self.pooled_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Expand from bottleneck to voxel space (used for training)
        # This layer is only used when we need voxel-level predictions
        self.bottleneck_to_voxels = nn.Sequential(
            nn.Linear(bottleneck_dim, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16384, n_voxels)
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        return_voxels: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder.

        Parameters
        ----------
        video : torch.Tensor
            Shape (batch_size, 3, height, width) video frames
        audio : torch.Tensor
            Shape (batch_size, n_mels) mel spectrograms
        text : torch.Tensor
            Shape (batch_size, text_dim) text embeddings
        return_voxels : bool, default=False
            If True, also return voxel-space representation

        Returns
        -------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim) compressed representation
        voxels : torch.Tensor, optional
            Shape (batch_size, n_voxels) if return_voxels=True

        Notes
        -----
        The bottleneck output represents the compressed "brain activity"
        that serves as the middle layer of the autoencoder.
        When return_voxels=True, the voxels are expanded from the bottleneck.
        """
        # Layer 2A/B/C: Encode each modality
        video_feat = self.video_encoder(video)      # (B, video_features)
        audio_feat = self.audio_encoder(audio)      # (B, audio_features)
        text_feat = self.text_encoder(text)         # (B, text_features)

        # Layer 3: Pool/concatenate features
        pooled = torch.cat([video_feat, audio_feat, text_feat], dim=1)  # (B, pooled_dim)

        # Layer 4: Feature space convolution + ReLU
        conv_feat = self.feature_conv(pooled)       # (B, pooled_dim)

        # Layer 5 + 6: Compress to bottleneck (middle layer)
        bottleneck = self.to_bottleneck(conv_feat)  # (B, bottleneck_dim)

        # Optionally expand to voxel space
        if return_voxels:
            voxels = self.bottleneck_to_voxels(bottleneck)  # (B, n_voxels)
            return bottleneck, voxels
        return bottleneck, None

    def get_parameter_count(self) -> dict:
        """
        Calculate number of parameters in each component.

        Returns
        -------
        param_dict : dict
            Dictionary with parameter counts for each module
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'video_encoder': count_params(self.video_encoder),
            'audio_encoder': count_params(self.audio_encoder),
            'text_encoder': count_params(self.text_encoder),
            'feature_conv': count_params(self.feature_conv),
            'to_bottleneck': count_params(self.to_bottleneck),
            'bottleneck_to_voxels': count_params(self.bottleneck_to_voxels),
            'total': count_params(self)
        }


def create_encoder(
    video_height: int = 90,
    video_width: int = 160,
    audio_mels: int = 128,
    text_dim: int = 1024,
    n_voxels: int = 85810,
    bottleneck_dim: int = 8000
) -> MultimodalEncoder:
    """
    Factory function to create MultimodalEncoder with default parameters.

    Parameters
    ----------
    video_height : int, default=90
        Video frame height
    video_width : int, default=160
        Video frame width
    audio_mels : int, default=128
        Number of mel frequency bins
    text_dim : int, default=1024
        Dimensionality of text embeddings
    n_voxels : int, default=85810
        Number of brain voxels
    bottleneck_dim : int, default=8000
        Dimensionality of bottleneck layer

    Returns
    -------
    encoder : MultimodalEncoder
        Initialized encoder model
    """
    return MultimodalEncoder(
        video_height=video_height,
        video_width=video_width,
        audio_mels=audio_mels,
        text_dim=text_dim,
        n_voxels=n_voxels,
        bottleneck_dim=bottleneck_dim
    )

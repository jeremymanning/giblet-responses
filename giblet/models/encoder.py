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
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional


class VideoEncoder(nn.Module):
    """
    Encode video features using Linear layers for temporal concatenation.

    Processes flattened temporal concatenation features where each TR contains
    concatenated frames from the temporal window [t-TR, t].

    For TR=1.5s @ 25fps: 38 frames × 160×90×3 = 1,641,600 features per TR

    This architecture replaces Conv2d layers with Linear layers to handle
    the flattened temporal concatenation format. Uses progressive dimensionality
    reduction similar to AudioEncoder and TextEncoder.

    Parameters
    ----------
    input_dim : int, default=1641600
        Dimensionality of flattened temporal concatenation
        (frames_per_tr × height × width × channels)
        Default: 38 × 160 × 90 × 3 = 1,641,600
    output_features : int, default=1024
        Dimensionality of output features
    """

    def __init__(
        self,
        input_dim: int = 1641600,
        output_features: int = 1024,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_features = output_features
        self.gradient_checkpointing = gradient_checkpointing

        # Progressive dimensionality reduction via Linear layers
        # 1,641,600 → 4096 → 2048 → 1024

        # Layer 1: Massive reduction (1.6M → 4K)
        self.fc1 = nn.Linear(input_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.3)

        # Layer 2: Further reduction (4K → 2K)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.3)

        # Layer 3: Final compression (2K → output)
        self.fc3 = nn.Linear(2048, output_features)
        self.bn3 = nn.BatchNorm1d(output_features)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through video encoder.

        Parameters
        ----------
        x : torch.Tensor
            Flattened temporal concatenation: (batch_size, input_dim)
            OR legacy 4D format: (batch_size, 3, height, width) for backward compatibility

        Returns
        -------
        features : torch.Tensor
            Shape (batch_size, output_features) encoded features

        Notes
        -----
        Handles both:
        1. Temporal concatenation (2D): [batch, 1641600] - flattened frames
        2. Legacy single frame (4D): [batch, 3, 90, 160] - will be flattened

        The temporal concatenation format is preferred and expected for training.
        """
        # Handle backward compatibility: if 4D input, flatten it
        if x.dim() == 4:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)

        # Ensure 2D input
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch, features), got {x.dim()}D: {x.shape}"
            )

        # MEMORY OPTIMIZATION (Issue #30): Use gradient checkpointing to reduce activation memory
        # Trades compute for memory by not storing intermediate activations during forward pass
        # Reduces activation memory by ~30-50% at the cost of ~30-40% slower training

        if self.gradient_checkpointing and self.training:
            # Define checkpointed blocks as lambda functions
            # PyTorch checkpoint requires functions that take tensors as input

            def block1(x):
                x = self.fc1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.dropout1(x)
                return x

            def block2(x):
                x = self.fc2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = self.dropout2(x)
                return x

            def block3(x):
                x = self.fc3(x)
                x = self.bn3(x)
                x = F.relu(x)
                x = self.dropout3(x)
                return x

            # Apply checkpointing to each block
            x = checkpoint(block1, x, use_reentrant=False)
            x = checkpoint(block2, x, use_reentrant=False)
            x = checkpoint(block3, x, use_reentrant=False)
        else:
            # Standard forward pass (no checkpointing)
            # Linear layer 1: 1,641,600 → 4096
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout1(x)

            # Linear layer 2: 4096 → 2048
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)

            # Linear layer 3: 2048 → output_features
            x = self.fc3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout3(x)

        return x


class AudioEncoder(nn.Module):
    """
    Encode audio features using Linear layers for temporal concatenation.

    Processes flattened temporal concatenation of audio features (EnCodec codes or mel spectrograms).
    For TR=1.5s @ 75Hz EnCodec: 8 codebooks × 112 frames = 896 features per TR

    Supports two input modes:
    1. EnCodec codes: (batch, n_codebooks * frames_per_tr) - flattened integer codes
    2. Mel spectrograms: (batch, n_mels * frames_per_tr) - flattened float values (legacy)

    Parameters
    ----------
    input_mels : int, default=2048
        Number of mel frequency bins (for mel spectrogram mode, legacy)
    input_codebooks : int, default=8
        Number of EnCodec codebooks (for EnCodec mode)
    frames_per_tr : int, default=65
        Number of temporal frames per TR
        - Mel mode: ~1.5s / 512 hop = 65 frames (legacy)
        - EnCodec mode: ~112 frames per TR at 3.0 kbps
    output_features : int, default=256
        Dimensionality of output features
    use_encodec : bool, default=False
        If True, process EnCodec quantized codes instead of mel spectrograms
    vocab_size : int, default=1024
        Not used (kept for backward compatibility)
    embed_dim : int, default=64
        Not used (kept for backward compatibility)
    """

    def __init__(
        self,
        input_mels: int = 2048,
        input_codebooks: int = 8,
        frames_per_tr: int = 65,
        output_features: int = 256,
        use_encodec: bool = False,
        vocab_size: int = 1024,  # Not used with Linear layers
        embed_dim: int = 64,  # Not used with Linear layers
    ):
        super().__init__()

        self.input_mels = input_mels
        self.input_codebooks = input_codebooks
        self.frames_per_tr = frames_per_tr
        self.output_features = output_features
        self.use_encodec = use_encodec

        # Calculate input dimension for flattened temporal concatenation
        if use_encodec:
            # EnCodec mode: flattened codebooks × frames
            # For 8 codebooks × 112 frames = 896 features
            input_dim = input_codebooks * frames_per_tr
        else:
            # Mel spectrogram mode: flattened mels × frames
            input_dim = input_mels * frames_per_tr

        # Progressive dimensionality reduction using Linear layers
        # Similar to VideoEncoder approach
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through audio encoder using Linear layers.

        Parameters
        ----------
        x : torch.Tensor
            Flattened audio features in one of these formats:
            - EnCodec mode: (batch_size, n_codebooks * frames_per_tr) - e.g., [batch, 896]
            - Mel mode: (batch_size, n_mels * frames_per_tr) - e.g., [batch, 2048*65]

        Returns
        -------
        features : torch.Tensor
            Shape (batch_size, output_features) encoded features

        Notes
        -----
        Handles flattened temporal concatenation directly without needing to
        reshape or embed. Linear layers process integer or float inputs seamlessly.
        """
        # Verify input is 2D flattened format
        if x.dim() != 2:
            raise ValueError(
                f"AudioEncoder expects 2D flattened input [batch, features], "
                f"but got shape {x.shape}"
            )

        # Verify expected input dimension
        if self.use_encodec:
            expected_dim = self.input_codebooks * self.frames_per_tr
        else:
            expected_dim = self.input_mels * self.frames_per_tr

        if x.shape[1] != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim}, but got {x.shape[1]}"
            )

        # Convert to float if integer (for EnCodec codes)
        if x.dtype in [torch.int32, torch.int64, torch.long]:
            x = x.float()

        # 2-layer linear network with batch norm and dropout
        x = F.relu(self.dropout1(self.bn1(self.fc1(x))))
        x = F.relu(self.dropout2(self.bn2(self.fc2(x))))

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

    def __init__(self, input_dim: int = 1024, output_features: int = 256):
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

    Architecture (Encoder half of 13-layer autoencoder):
    - Layer 1: Input (video + audio + text)
    - Layer 2A/B/C: Modality-specific encoders (convolutions/linear)
    - Layer 3: Pooled multimodal features (1536 dims)
    - Layer 4: Feature space convolution + ReLU (1536 dims)
    - Layer 5: First expansion (1536 → 4096 dims)
    - Layer 6: Second expansion (4096 → 8000 dims)
    - Layer 7: BOTTLENECK compression (8000 → 2048 dims, smallest layer)

    Parameters
    ----------
    video_height : int, default=90
        Video frame height
    video_width : int, default=160
        Video frame width
    audio_mels : int, default=2048
        Number of mel frequency bins
    text_dim : int, default=1024
        Dimensionality of text embeddings
    n_voxels : int, default=85810
        Number of brain voxels
    bottleneck_dim : int, default=2048
        Dimensionality of bottleneck layer (middle layer, smallest in autoencoder)
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
        audio_mels: int = 2048,
        audio_codebooks: int = 8,  # NEW: EnCodec codebooks
        audio_frames_per_tr: int = 65,  # Temporal frames per TR
        text_dim: int = 1024,
        n_voxels: int = 85810,
        bottleneck_dim: int = 2048,
        video_features: int = 1024,
        audio_features: int = 256,
        text_features: int = 256,
        use_encodec: bool = False,  # NEW: Use EnCodec codes instead of mel spectrograms
        gradient_checkpointing: bool = False,  # NEW (Issue #30): Enable gradient checkpointing for memory optimization
        video_frames_per_tr: int = 19,  # NEW (Issue #30): Video frames per TR after frame skipping
    ):
        super().__init__()

        self.video_height = video_height
        self.video_width = video_width
        self.audio_mels = audio_mels
        self.audio_codebooks = audio_codebooks
        self.audio_frames_per_tr = audio_frames_per_tr
        self.text_dim = text_dim
        self.n_voxels = n_voxels
        self.bottleneck_dim = bottleneck_dim
        self.use_encodec = use_encodec
        self.gradient_checkpointing = gradient_checkpointing
        self.video_frames_per_tr = video_frames_per_tr

        # Layer 2A: Video encoder
        # Calculate input dimension for temporal concatenation
        # Example: 19 frames/TR (with frame_skip=2) × 160×90×3 = 820,800
        video_input_dim = video_frames_per_tr * video_height * video_width * 3

        self.video_encoder = VideoEncoder(
            input_dim=video_input_dim,  # Flattened temporal concatenation
            output_features=video_features,
            gradient_checkpointing=gradient_checkpointing,  # Pass gradient checkpointing flag
        )

        # Layer 2B: Audio encoder (supports both mel spectrograms and EnCodec codes)
        self.audio_encoder = AudioEncoder(
            input_mels=audio_mels,
            input_codebooks=audio_codebooks,
            frames_per_tr=audio_frames_per_tr,
            output_features=audio_features,
            use_encodec=use_encodec,
        )

        # Layer 2C: Text encoder
        self.text_encoder = TextEncoder(
            input_dim=text_dim, output_features=text_features
        )

        # Layer 3: Pooled features dimension
        self.pooled_dim = video_features + audio_features + text_features

        # Layer 4: Feature space convolution (1D conv over concatenated features)
        # This acts as a learned weighted combination of modalities
        self.feature_conv = nn.Sequential(
            nn.Linear(self.pooled_dim, self.pooled_dim),
            nn.BatchNorm1d(self.pooled_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Layer 5: First expansion (1536 → 4096)
        self.layer5 = nn.Sequential(
            nn.Linear(self.pooled_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Layer 6: Second expansion (4096 → 8000)
        self.layer6 = nn.Sequential(
            nn.Linear(4096, 8000), nn.BatchNorm1d(8000), nn.ReLU(), nn.Dropout(0.3)
        )

        # Layer 7: BOTTLENECK - compression to smallest dimension (8000 → bottleneck_dim)
        # No ReLU after bottleneck to allow negative values in latent space
        self.layer7_bottleneck = nn.Sequential(
            nn.Linear(8000, bottleneck_dim),  # Layer 7: BOTTLENECK (smallest layer)
            nn.BatchNorm1d(bottleneck_dim),
        )

        # Optional: Expand from bottleneck to voxel space (for direct voxel prediction)
        # This is NOT part of the main 13-layer architecture but useful for auxiliary loss
        self.bottleneck_to_voxels = nn.Sequential(
            nn.Linear(bottleneck_dim, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16384, n_voxels),
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        return_voxels: bool = False,
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
        # Convert audio to float if it's integer (EnCodec codes)
        if audio.dtype in [torch.int32, torch.int64, torch.long]:
            audio = audio.float()

        video_feat = self.video_encoder(video)  # (B, video_features)
        audio_feat = self.audio_encoder(audio)  # (B, audio_features)
        text_feat = self.text_encoder(text)  # (B, text_features)

        # Layer 3: Pool/concatenate features
        pooled = torch.cat(
            [video_feat, audio_feat, text_feat], dim=1
        )  # (B, pooled_dim)

        # Layer 4: Feature space convolution + ReLU
        conv_feat = self.feature_conv(pooled)  # (B, pooled_dim=1536)

        # Layer 5: First expansion (1536 → 4096)
        layer5_out = self.layer5(conv_feat)  # (B, 4096)

        # Layer 6: Second expansion (4096 → 8000)
        layer6_out = self.layer6(layer5_out)  # (B, 8000)

        # Layer 7: BOTTLENECK - compress to smallest dimension (8000 → 2048)
        bottleneck = self.layer7_bottleneck(layer6_out)  # (B, bottleneck_dim=2048)

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
            "video_encoder": count_params(self.video_encoder),
            "audio_encoder": count_params(self.audio_encoder),
            "text_encoder": count_params(self.text_encoder),
            "feature_conv": count_params(self.feature_conv),
            "layer5": count_params(self.layer5),
            "layer6": count_params(self.layer6),
            "layer7_bottleneck": count_params(self.layer7_bottleneck),
            "bottleneck_to_voxels": count_params(self.bottleneck_to_voxels),
            "total": count_params(self),
        }


def create_encoder(
    video_height: int = 90,
    video_width: int = 160,
    audio_mels: int = 2048,
    text_dim: int = 1024,
    n_voxels: int = 85810,
    bottleneck_dim: int = 2048,
) -> MultimodalEncoder:
    """
    Factory function to create MultimodalEncoder with default parameters.

    Parameters
    ----------
    video_height : int, default=90
        Video frame height
    video_width : int, default=160
        Video frame width
    audio_mels : int, default=2048
        Number of mel frequency bins
    text_dim : int, default=1024
        Dimensionality of text embeddings
    n_voxels : int, default=85810
        Number of brain voxels
    bottleneck_dim : int, default=2048
        Dimensionality of bottleneck layer (Layer 7, smallest in autoencoder)

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
        bottleneck_dim=bottleneck_dim,
    )

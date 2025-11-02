"""
Full autoencoder combining encoder and decoder for multimodal fMRI data.

Implements the complete architecture that:
1. Encodes video/audio/text → bottleneck (compressed brain activity)
2. Decodes bottleneck → reconstructed video/audio/text

Supports dual loss function:
- Reconstruction loss: Compare reconstructed stimuli to originals
- fMRI matching loss: Compare predicted fMRI to actual brain activity

Supports multi-GPU training via DistributedDataParallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path

from .encoder import MultimodalEncoder
from .decoder import MultimodalDecoder


class MultimodalAutoencoder(nn.Module):
    """
    Full autoencoder for multimodal fMRI prediction and reconstruction.

    Architecture:
    - Encoder: video/audio/text → bottleneck (8000-dim) → fMRI voxels (85,810)
    - Decoder: bottleneck → reconstructed video/audio/text

    The bottleneck serves as the compressed "brain activity" representation,
    which is the middle layer connecting encoder and decoder.

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
    bottleneck_dim : int, default=8000
        Dimensionality of bottleneck (middle layer)
    video_features : int, default=1024
        Video encoder output features
    audio_features : int, default=256
        Audio encoder output features
    text_features : int, default=256
        Text encoder output features
    decoder_hidden_dim : int, default=2048
        Decoder hidden layer dimension
    decoder_dropout : float, default=0.3
        Decoder dropout rate
    reconstruction_weight : float, default=1.0
        Weight for reconstruction loss
    fmri_weight : float, default=1.0
        Weight for fMRI matching loss
    """

    def __init__(
        self,
        video_height: int = 90,
        video_width: int = 160,
        audio_mels: int = 2048,
        text_dim: int = 1024,
        n_voxels: int = 85810,
        bottleneck_dim: int = 2048,
        video_features: int = 1024,
        audio_features: int = 256,
        text_features: int = 256,
        decoder_hidden_dim: int = 2048,
        decoder_dropout: float = 0.3,
        reconstruction_weight: float = 1.0,
        fmri_weight: float = 1.0,
        use_encodec: bool = False,  # NEW: Use EnCodec discrete codes
        audio_frames_per_tr: int = 65  # NEW: Frames per TR (65 mel, 112 EnCodec)
    ):
        super().__init__()

        self.video_height = video_height
        self.video_width = video_width
        self.audio_mels = audio_mels
        self.text_dim = text_dim
        self.n_voxels = n_voxels
        self.bottleneck_dim = bottleneck_dim
        self.reconstruction_weight = reconstruction_weight
        self.fmri_weight = fmri_weight
        self.use_encodec = use_encodec
        self.audio_frames_per_tr = audio_frames_per_tr

        # Encoder: stimulus → bottleneck → fMRI voxels
        self.encoder = MultimodalEncoder(
            video_height=video_height,
            video_width=video_width,
            audio_mels=audio_mels,
            audio_codebooks=8,  # EnCodec: 8 codebooks for 3.0 kbps
            audio_frames_per_tr=audio_frames_per_tr,  # Pass through from autoencoder
            text_dim=text_dim,
            n_voxels=n_voxels,
            bottleneck_dim=bottleneck_dim,
            video_features=video_features,
            audio_features=audio_features,
            text_features=text_features,
            use_encodec=use_encodec  # Pass through from autoencoder
        )

        # Decoder: bottleneck → reconstructed stimulus
        # Video output is flattened: 160×90×3 = 43,200
        video_dim = video_width * video_height * 3
        self.decoder = MultimodalDecoder(
            bottleneck_dim=bottleneck_dim,
            video_dim=video_dim,
            audio_dim=audio_mels,
            audio_frames_per_tr=audio_frames_per_tr,
            text_dim=text_dim,
            dropout=decoder_dropout,
            use_encodec=use_encodec
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        fmri_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full autoencoder.

        Parameters
        ----------
        video : torch.Tensor
            Shape (batch_size, 3, height, width) video frames
        audio : torch.Tensor
            Shape (batch_size, n_mels) mel spectrograms
        text : torch.Tensor
            Shape (batch_size, text_dim) text embeddings
        fmri_target : torch.Tensor, optional
            Shape (batch_size, n_voxels) target fMRI voxels for training

        Returns
        -------
        outputs : dict
            Dictionary containing:
            - 'bottleneck': Compressed brain representation (batch_size, bottleneck_dim)
            - 'predicted_fmri': Predicted fMRI voxels (batch_size, n_voxels)
            - 'video_recon': Reconstructed video (batch_size, video_dim)
            - 'audio_recon': Reconstructed audio (batch_size, audio_dim)
            - 'text_recon': Reconstructed text (batch_size, text_dim)
            - 'total_loss': Total loss (if training with fmri_target)
            - 'reconstruction_loss': Reconstruction loss (if training)
            - 'fmri_loss': fMRI matching loss (if training with fmri_target)
        """
        batch_size = video.size(0)

        # Encode: stimulus → bottleneck → fMRI voxels
        bottleneck, predicted_fmri = self.encoder(
            video, audio, text, return_voxels=True
        )

        # Decode: bottleneck → reconstructed stimulus
        video_recon, audio_recon, text_recon = self.decoder(bottleneck)

        outputs = {
            'bottleneck': bottleneck,
            'predicted_fmri': predicted_fmri,
            'video_recon': video_recon,
            'audio_recon': audio_recon,
            'text_recon': text_recon
        }

        # Compute losses if in training mode
        if self.training:
            # Flatten original video for comparison
            # video shape: (B, 3, H, W) → (B, 3*H*W)
            video_flat = video.view(batch_size, -1)

            # Reconstruction loss
            video_loss = F.mse_loss(video_recon, video_flat)

            # Audio loss: Different handling for EnCodec vs mel spectrograms
            if self.use_encodec:
                # EnCodec: Predict discrete codes
                # audio: (batch, n_codebooks, frames_per_tr) integer codes
                # audio_recon: (batch, n_codebooks, frames_per_tr) continuous predictions
                # Convert target to float for MSE loss
                audio_float = audio.float()
                audio_loss = F.mse_loss(audio_recon, audio_float)
            else:
                # Mel spectrograms: Continuous features
                # audio: (batch, n_mels, frames_per_tr) continuous
                # audio_recon: (batch, n_mels, frames_per_tr) continuous
                audio_loss = F.mse_loss(audio_recon, audio)

            text_loss = F.mse_loss(text_recon, text)
            reconstruction_loss = video_loss + audio_loss + text_loss

            outputs['reconstruction_loss'] = reconstruction_loss
            outputs['video_loss'] = video_loss
            outputs['audio_loss'] = audio_loss
            outputs['text_loss'] = text_loss

            # fMRI matching loss (if target provided)
            if fmri_target is not None:
                fmri_loss = F.mse_loss(predicted_fmri, fmri_target)
                outputs['fmri_loss'] = fmri_loss

                # Total loss (weighted combination)
                total_loss = (
                    self.reconstruction_weight * reconstruction_loss +
                    self.fmri_weight * fmri_loss
                )
                outputs['total_loss'] = total_loss
            else:
                # Only reconstruction loss if no fMRI target
                outputs['total_loss'] = self.reconstruction_weight * reconstruction_loss

        return outputs

    def encode_only(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        return_voxels: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode stimulus to bottleneck and optionally fMRI voxels.

        Parameters
        ----------
        video : torch.Tensor
            Shape (batch_size, 3, height, width) video frames
        audio : torch.Tensor
            Shape (batch_size, n_mels) mel spectrograms
        text : torch.Tensor
            Shape (batch_size, text_dim) text embeddings
        return_voxels : bool, default=False
            If True, also return predicted fMRI voxels

        Returns
        -------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim)
        predicted_fmri : torch.Tensor, optional
            Shape (batch_size, n_voxels) if return_voxels=True
        """
        return self.encoder(video, audio, text, return_voxels=return_voxels)

    def decode_only(
        self,
        bottleneck: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode bottleneck to reconstructed stimulus.

        Parameters
        ----------
        bottleneck : torch.Tensor
            Shape (batch_size, bottleneck_dim)

        Returns
        -------
        video_recon : torch.Tensor
            Shape (batch_size, video_dim) reconstructed video
        audio_recon : torch.Tensor
            Shape (batch_size, audio_dim) reconstructed audio
        text_recon : torch.Tensor
            Shape (batch_size, text_dim) reconstructed text
        """
        return self.decoder(bottleneck)

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate number of parameters in each component.

        Returns
        -------
        param_dict : dict
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        encoder_params = self.encoder.get_parameter_count()
        decoder_params = self.decoder.count_parameters()

        return {
            'encoder': encoder_params['total'],
            'encoder_breakdown': encoder_params,
            'decoder': decoder_params['total'],
            'decoder_breakdown': decoder_params,
            'total': count_params(self)
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        loss: Optional[float] = None,
        metadata: Optional[dict] = None
    ):
        """
        Save model checkpoint.

        Parameters
        ----------
        path : str
            Path to save checkpoint
        epoch : int
            Training epoch
        optimizer_state : dict, optional
            Optimizer state dict
        loss : float, optional
            Training loss
        metadata : dict, optional
            Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'architecture': {
                'video_height': self.video_height,
                'video_width': self.video_width,
                'audio_mels': self.audio_mels,
                'text_dim': self.text_dim,
                'n_voxels': self.n_voxels,
                'bottleneck_dim': self.bottleneck_dim,
                'reconstruction_weight': self.reconstruction_weight,
                'fmri_weight': self.fmri_weight,
                'use_encodec': self.use_encodec,
                'audio_frames_per_tr': self.audio_frames_per_tr
            }
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if loss is not None:
            checkpoint['loss'] = loss

        if metadata is not None:
            checkpoint['metadata'] = metadata

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: Optional[torch.device] = None
    ) -> Tuple['MultimodalAutoencoder', dict]:
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint file
        device : torch.device, optional
            Device to load model to

        Returns
        -------
        model : SherlockAutoencoder
            Loaded model
        checkpoint : dict
            Checkpoint dictionary with epoch, loss, etc.
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model with saved architecture
        arch = checkpoint['architecture']
        model = cls(
            video_height=arch['video_height'],
            video_width=arch['video_width'],
            audio_mels=arch['audio_mels'],
            text_dim=arch['text_dim'],
            n_voxels=arch['n_voxels'],
            bottleneck_dim=arch['bottleneck_dim'],
            reconstruction_weight=arch['reconstruction_weight'],
            fmri_weight=arch['fmri_weight'],
            use_encodec=arch.get('use_encodec', False),  # Default to False for old checkpoints
            audio_frames_per_tr=arch.get('audio_frames_per_tr', 65)  # Default to mel value
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model = model.to(device)

        return model, checkpoint


def create_autoencoder(
    video_height: int = 90,
    video_width: int = 160,
    audio_mels: int = 2048,
    text_dim: int = 1024,
    n_voxels: int = 85810,
    bottleneck_dim: int = 8000,
    reconstruction_weight: float = 1.0,
    fmri_weight: float = 1.0,
    use_encodec: bool = False,
    audio_frames_per_tr: int = 65
) -> MultimodalAutoencoder:
    """
    Factory function to create MultimodalAutoencoder with default parameters.

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
    reconstruction_weight : float, default=1.0
        Weight for reconstruction loss
    fmri_weight : float, default=1.0
        Weight for fMRI matching loss
    use_encodec : bool, default=False
        If True, use EnCodec discrete codes; if False, use mel spectrograms
    audio_frames_per_tr : int, default=65
        Number of audio frames per TR (65 for mel @ 44Hz, 112 for EnCodec @ 75Hz)

    Returns
    -------
    autoencoder : MultimodalAutoencoder
        Initialized autoencoder model
    """
    return MultimodalAutoencoder(
        video_height=video_height,
        video_width=video_width,
        audio_mels=audio_mels,
        text_dim=text_dim,
        n_voxels=n_voxels,
        bottleneck_dim=bottleneck_dim,
        reconstruction_weight=reconstruction_weight,
        fmri_weight=fmri_weight,
        use_encodec=use_encodec,
        audio_frames_per_tr=audio_frames_per_tr
    )


def prepare_for_distributed(
    model: MultimodalAutoencoder,
    device_ids: Optional[list] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False
) -> nn.parallel.DistributedDataParallel:
    """
    Wrap model in DistributedDataParallel for multi-GPU training.

    Parameters
    ----------
    model : SherlockAutoencoder
        Model to wrap
    device_ids : list, optional
        List of GPU device IDs to use
    output_device : int, optional
        Primary GPU for output
    find_unused_parameters : bool, default=False
        Find unused parameters (needed for some architectures)

    Returns
    -------
    ddp_model : DistributedDataParallel
        Wrapped model for distributed training

    Notes
    -----
    Before calling this function, you must initialize the process group:
    >>> torch.distributed.init_process_group(backend='nccl')
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed process group not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    # Move model to GPU
    if device_ids is not None and len(device_ids) > 0:
        model = model.to(f'cuda:{device_ids[0]}')
    else:
        model = model.cuda()

    # Wrap in DDP
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters
    )

    return ddp_model

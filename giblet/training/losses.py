"""
Loss functions for training the multimodal autoencoder.

This module provides various loss functions used for training:
1. Reconstruction losses (MSE, Cosine similarity)
2. fMRI matching losses
3. Perceptual losses for video/audio
4. Combined loss functions with weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ReconstructionLoss(nn.Module):
    """
    Multi-modal reconstruction loss.

    Computes MSE loss between reconstructed and original stimuli
    for all modalities (video, audio, text).

    Parameters
    ----------
    video_weight : float, default=1.0
        Weight for video reconstruction loss
    audio_weight : float, default=1.0
        Weight for audio reconstruction loss
    text_weight : float, default=1.0
        Weight for text reconstruction loss
    reduction : str, default='mean'
        Loss reduction method: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        text_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        self.reduction = reduction

    def forward(
        self,
        video_recon: torch.Tensor,
        video_target: torch.Tensor,
        audio_recon: torch.Tensor,
        audio_target: torch.Tensor,
        text_recon: torch.Tensor,
        text_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction loss.

        Parameters
        ----------
        video_recon : torch.Tensor
            Reconstructed video (batch_size, video_dim)
        video_target : torch.Tensor
            Target video (batch_size, video_dim)
        audio_recon : torch.Tensor
            Reconstructed audio (batch_size, audio_dim)
        audio_target : torch.Tensor
            Target audio (batch_size, audio_dim)
        text_recon : torch.Tensor
            Reconstructed text (batch_size, text_dim)
        text_target : torch.Tensor
            Target text (batch_size, text_dim)

        Returns
        -------
        total_loss : torch.Tensor
            Total weighted reconstruction loss
        loss_dict : dict
            Dictionary with individual modality losses
        """
        # Compute MSE for each modality
        # TEMPORARY: Handle dimension mismatch for temporal concatenation (Issue #29)
        # video_recon: (B, 43200) single frame from decoder
        # video_target: (B, 1641600) temporal concatenation from encoder
        # TODO: Update decoder for temporal concatenation (separate from Issue #29)
        if video_target.size(1) != video_recon.size(1):
            # Truncate target to match decoder output
            video_target = video_target[:, :video_recon.size(1)]

        # Similar handling for audio if needed
        if audio_target.dim() == 2 and audio_recon.dim() == 3:
            # Flatten decoder output to match flattened input
            audio_recon = audio_recon.view(audio_recon.size(0), -1)
        elif audio_target.dim() == 3 and audio_recon.dim() == 2:
            # Shouldn't happen, but handle just in case
            audio_target = audio_target.view(audio_target.size(0), -1)

        video_loss = F.mse_loss(video_recon, video_target, reduction=self.reduction)
        audio_loss = F.mse_loss(audio_recon, audio_target, reduction=self.reduction)
        text_loss = F.mse_loss(text_recon, text_target, reduction=self.reduction)

        # Weighted sum
        total_loss = (
            self.video_weight * video_loss +
            self.audio_weight * audio_loss +
            self.text_weight * text_loss
        )

        loss_dict = {
            'video_loss': video_loss,
            'audio_loss': audio_loss,
            'text_loss': text_loss,
            'reconstruction_loss': total_loss
        }

        return total_loss, loss_dict


class FMRIMatchingLoss(nn.Module):
    """
    fMRI matching loss.

    Computes loss between predicted and actual fMRI voxel activations.
    Supports MSE and correlation-based losses.

    Parameters
    ----------
    loss_type : str, default='mse'
        Type of loss: 'mse', 'mae', or 'correlation'
    reduction : str, default='mean'
        Loss reduction method: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        predicted_fmri: torch.Tensor,
        target_fmri: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fMRI matching loss.

        Parameters
        ----------
        predicted_fmri : torch.Tensor
            Predicted fMRI voxels (batch_size, n_voxels)
        target_fmri : torch.Tensor
            Target fMRI voxels (batch_size, n_voxels)

        Returns
        -------
        loss : torch.Tensor
            fMRI matching loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_fmri, target_fmri, reduction=self.reduction)

        elif self.loss_type == 'mae':
            loss = F.l1_loss(predicted_fmri, target_fmri, reduction=self.reduction)

        elif self.loss_type == 'correlation':
            # Compute correlation-based loss (1 - correlation)
            # Normalize along voxel dimension
            pred_norm = F.normalize(predicted_fmri, dim=1)
            target_norm = F.normalize(target_fmri, dim=1)

            # Cosine similarity (equivalent to correlation for normalized data)
            similarity = (pred_norm * target_norm).sum(dim=1)

            # Convert to loss (1 - similarity)
            loss = 1.0 - similarity

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class CombinedAutoEncoderLoss(nn.Module):
    """
    Combined loss function for autoencoder training.

    Combines reconstruction loss and fMRI matching loss with configurable weights.

    Parameters
    ----------
    reconstruction_weight : float, default=1.0
        Weight for reconstruction loss
    fmri_weight : float, default=1.0
        Weight for fMRI matching loss
    video_weight : float, default=1.0
        Weight for video in reconstruction loss
    audio_weight : float, default=1.0
        Weight for audio in reconstruction loss
    text_weight : float, default=1.0
        Weight for text in reconstruction loss
    fmri_loss_type : str, default='mse'
        Type of fMRI loss: 'mse', 'mae', or 'correlation'
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        fmri_weight: float = 1.0,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        text_weight: float = 1.0,
        fmri_loss_type: str = 'mse'
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.fmri_weight = fmri_weight

        self.reconstruction_loss = ReconstructionLoss(
            video_weight=video_weight,
            audio_weight=audio_weight,
            text_weight=text_weight
        )

        self.fmri_loss = FMRIMatchingLoss(loss_type=fmri_loss_type)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        text_target: torch.Tensor,
        fmri_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Parameters
        ----------
        outputs : dict
            Model outputs containing:
            - 'video_recon': Reconstructed video
            - 'audio_recon': Reconstructed audio
            - 'text_recon': Reconstructed text
            - 'predicted_fmri': Predicted fMRI voxels
        video_target : torch.Tensor
            Target video (flattened)
        audio_target : torch.Tensor
            Target audio
        text_target : torch.Tensor
            Target text
        fmri_target : torch.Tensor, optional
            Target fMRI voxels

        Returns
        -------
        total_loss : torch.Tensor
            Combined weighted loss
        loss_dict : dict
            Dictionary with all individual losses
        """
        # Reconstruction loss
        recon_loss, recon_dict = self.reconstruction_loss(
            video_recon=outputs['video_recon'],
            video_target=video_target,
            audio_recon=outputs['audio_recon'],
            audio_target=audio_target,
            text_recon=outputs['text_recon'],
            text_target=text_target
        )

        loss_dict = recon_dict.copy()

        # fMRI matching loss (if target provided)
        if fmri_target is not None:
            fmri_loss_val = self.fmri_loss(
                predicted_fmri=outputs['predicted_fmri'],
                target_fmri=fmri_target
            )
            loss_dict['fmri_loss'] = fmri_loss_val

            # Combined loss
            total_loss = (
                self.reconstruction_weight * recon_loss +
                self.fmri_weight * fmri_loss_val
            )
        else:
            # Only reconstruction loss
            total_loss = self.reconstruction_weight * recon_loss

        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for video reconstruction.

    Uses a pretrained network to compute feature-space loss
    rather than pixel-space loss.

    Parameters
    ----------
    layer_weights : dict, optional
        Weights for different feature layers
    """

    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        # Could implement VGG-based perceptual loss here
        # For now, placeholder for future enhancement
        raise NotImplementedError(
            "Perceptual loss not yet implemented. "
            "Use MSE-based reconstruction loss instead."
        )


def compute_correlation_metric(
    predicted: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute correlation between predicted and target tensors.

    This is useful as an evaluation metric (not for training loss).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted values (batch_size, n_features)
    target : torch.Tensor
        Target values (batch_size, n_features)
    dim : int, default=1
        Dimension along which to compute correlation

    Returns
    -------
    correlation : torch.Tensor
        Correlation values (batch_size,)
    """
    # Center the data
    pred_centered = predicted - predicted.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)

    # Compute correlation
    numerator = (pred_centered * target_centered).sum(dim=dim)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=dim))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=dim))
    denominator = pred_std * target_std

    # Avoid division by zero
    correlation = numerator / (denominator + 1e-8)

    return correlation


def compute_r2_score(
    predicted: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute R^2 score (coefficient of determination).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted values (batch_size, n_features)
    target : torch.Tensor
        Target values (batch_size, n_features)
    dim : int, default=1
        Dimension along which to compute R^2

    Returns
    -------
    r2 : torch.Tensor
        R^2 scores (batch_size,)
    """
    # Compute residual sum of squares
    ss_res = ((target - predicted) ** 2).sum(dim=dim)

    # Compute total sum of squares
    target_mean = target.mean(dim=dim, keepdim=True)
    ss_tot = ((target - target_mean) ** 2).sum(dim=dim)

    # R^2 = 1 - (SS_res / SS_tot)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return r2

"""
Loss functions for training the multimodal autoencoder with modality normalization.

This module provides modality-normalized loss functions where each modality's loss
is divided by its standard deviation to make losses comparable across modalities.
All losses are in "standard deviation units" relative to their modality.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedReconstructionLoss(nn.Module):
    """
    Multi-modal reconstruction loss with modality-specific normalization.

    Computes MSE loss between reconstructed and original stimuli for all modalities
    (video, audio, text), normalized by the standard deviation of each modality's
    target to make losses comparable across modalities.

    Each loss is computed as: (MSE / std(target))
    This puts all losses in standard deviation units.

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
    normalize_by_std : bool, default=True
        If True, normalize each modality's loss by its target std
    """

    def __init__(
        self,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        text_weight: float = 1.0,
        reduction: str = "mean",
        normalize_by_std: bool = True,
    ):
        super().__init__()
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        self.reduction = reduction
        self.normalize_by_std = normalize_by_std

    def forward(
        self,
        video_recon: torch.Tensor,
        video_target: torch.Tensor,
        audio_recon: torch.Tensor,
        audio_target: torch.Tensor,
        text_recon: torch.Tensor,
        text_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute normalized reconstruction loss.

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
            Dictionary with individual modality losses (normalized and unnormalized)
        """
        # Handle dimension mismatches (from existing code)
        if video_target.size(1) != video_recon.size(1):
            video_target = video_target[:, : video_recon.size(1)]

        if audio_target.dim() == 2 and audio_recon.dim() == 3:
            audio_recon = audio_recon.view(audio_recon.size(0), -1)
        elif audio_target.dim() == 3 and audio_recon.dim() == 2:
            audio_target = audio_target.view(audio_target.size(0), -1)

        # Convert audio_target to float if it's integer (EnCodec codes)
        if audio_target.dtype in [torch.int32, torch.int64, torch.long]:
            audio_target = audio_target.float()

        # Compute raw MSE for each modality
        video_mse = F.mse_loss(video_recon, video_target, reduction=self.reduction)
        audio_mse = F.mse_loss(audio_recon, audio_target, reduction=self.reduction)
        text_mse = F.mse_loss(text_recon, text_target, reduction=self.reduction)

        loss_dict = {
            "video_loss_raw": video_mse,
            "audio_loss_raw": audio_mse,
            "text_loss_raw": text_mse,
        }

        # Normalize by standard deviation if requested
        if self.normalize_by_std:
            # Compute std for each modality's target
            # Use std across all dimensions for a single normalizing factor
            video_std = torch.std(video_target) + 1e-8  # avoid division by zero
            audio_std = torch.std(audio_target) + 1e-8
            text_std = torch.std(text_target) + 1e-8

            # Normalize losses: MSE / std
            # This puts losses in "standard deviation units"
            video_loss = video_mse / video_std
            audio_loss = audio_mse / audio_std
            text_loss = text_mse / text_std

            loss_dict.update({
                "video_std": video_std,
                "audio_std": audio_std,
                "text_std": text_std,
                "video_loss": video_loss,
                "audio_loss": audio_loss,
                "text_loss": text_loss,
            })
        else:
            # Use raw MSE
            video_loss = video_mse
            audio_loss = audio_mse
            text_loss = text_mse

            loss_dict.update({
                "video_loss": video_loss,
                "audio_loss": audio_loss,
                "text_loss": text_loss,
            })

        # Weighted sum
        total_loss = (
            self.video_weight * video_loss
            + self.audio_weight * audio_loss
            + self.text_weight * text_loss
        )

        loss_dict["reconstruction_loss"] = total_loss

        return total_loss, loss_dict


class NormalizedFMRIMatchingLoss(nn.Module):
    """
    fMRI matching loss with normalization.

    Computes loss between predicted and actual fMRI voxel activations,
    normalized by the target standard deviation.

    Parameters
    ----------
    loss_type : str, default='mse'
        Type of loss: 'mse', 'mae', or 'correlation'
    reduction : str, default='mean'
        Loss reduction method: 'mean', 'sum', or 'none'
    normalize_by_std : bool, default=True
        If True, normalize loss by target std
    """

    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
        normalize_by_std: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize_by_std = normalize_by_std

    def forward(
        self, predicted_fmri: torch.Tensor, target_fmri: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute normalized fMRI matching loss.

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
        loss_dict : dict
            Dictionary with raw and normalized losses
        """
        if self.loss_type == "mse":
            raw_loss = F.mse_loss(predicted_fmri, target_fmri, reduction=self.reduction)
        elif self.loss_type == "mae":
            raw_loss = F.l1_loss(predicted_fmri, target_fmri, reduction=self.reduction)
        elif self.loss_type == "correlation":
            # Compute correlation-based loss (1 - correlation)
            pred_norm = F.normalize(predicted_fmri, dim=1)
            target_norm = F.normalize(target_fmri, dim=1)
            similarity = (pred_norm * target_norm).sum(dim=1)
            raw_loss = 1.0 - similarity
            if self.reduction == "mean":
                raw_loss = raw_loss.mean()
            elif self.reduction == "sum":
                raw_loss = raw_loss.sum()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss_dict = {"fmri_loss_raw": raw_loss}

        # Normalize by std if requested
        if self.normalize_by_std and self.loss_type in ["mse", "mae"]:
            fmri_std = torch.std(target_fmri) + 1e-8
            normalized_loss = raw_loss / fmri_std
            loss_dict["fmri_std"] = fmri_std
            loss_dict["fmri_loss"] = normalized_loss
            return normalized_loss, loss_dict
        else:
            loss_dict["fmri_loss"] = raw_loss
            return raw_loss, loss_dict


class NormalizedCombinedAutoEncoderLoss(nn.Module):
    """
    Combined loss function with modality-specific normalization.

    All modality losses are normalized by their respective standard deviations
    to make them comparable in "standard deviation units".

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
    normalize_by_std : bool, default=True
        If True, normalize all losses by their modality std
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        fmri_weight: float = 1.0,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        text_weight: float = 1.0,
        fmri_loss_type: str = "mse",
        normalize_by_std: bool = True,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.fmri_weight = fmri_weight

        self.reconstruction_loss = NormalizedReconstructionLoss(
            video_weight=video_weight,
            audio_weight=audio_weight,
            text_weight=text_weight,
            normalize_by_std=normalize_by_std,
        )

        self.fmri_loss = NormalizedFMRIMatchingLoss(
            loss_type=fmri_loss_type,
            normalize_by_std=normalize_by_std,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        text_target: torch.Tensor,
        fmri_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined normalized loss.

        Parameters
        ----------
        outputs : dict
            Model outputs containing reconstructions and predictions
        video_target, audio_target, text_target, fmri_target : torch.Tensor
            Target values for each modality

        Returns
        -------
        total_loss : torch.Tensor
            Combined weighted loss
        loss_dict : dict
            Dictionary with all individual losses (raw and normalized)
        """
        # Reconstruction loss
        recon_loss, recon_dict = self.reconstruction_loss(
            video_recon=outputs["video_recon"],
            video_target=video_target,
            audio_recon=outputs["audio_recon"],
            audio_target=audio_target,
            text_recon=outputs["text_recon"],
            text_target=text_target,
        )

        loss_dict = recon_dict.copy()

        # fMRI matching loss (if target provided)
        if fmri_target is not None:
            fmri_loss_val, fmri_dict = self.fmri_loss(
                predicted_fmri=outputs["predicted_fmri"], target_fmri=fmri_target
            )
            loss_dict.update(fmri_dict)

            # Combined loss
            total_loss = (
                self.reconstruction_weight * recon_loss
                + self.fmri_weight * fmri_loss_val
            )
        else:
            # Only reconstruction loss
            total_loss = self.reconstruction_weight * recon_loss

        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict

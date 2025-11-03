"""
Trainer for multimodal autoencoder with multi-GPU support.

This module provides a comprehensive training system supporting:
- Single-GPU and multi-GPU (DistributedDataParallel) training
- Training and validation loops
- Checkpointing (save/resume)
- Learning rate scheduling
- Early stopping
- Mixed precision training (FP16)
- Logging and metrics tracking
- Gradient clipping

Example
-------
>>> from giblet.models.autoencoder import create_autoencoder
>>> from giblet.data.dataset import MultimodalDataset
>>> from giblet.training.trainer import Trainer, TrainingConfig
>>>
>>> # Create model and datasets
>>> model = create_autoencoder()
>>> train_dataset = MultimodalDataset('data/', split='train')
>>> val_dataset = MultimodalDataset('data/', split='val')
>>>
>>> # Configure training
>>> config = TrainingConfig(
...     learning_rate=1e-4,
...     batch_size=64,
...     num_epochs=100,
...     use_mixed_precision=True
... )
>>>
>>> # Train
>>> trainer = Trainer(model, train_dataset, val_dataset, config)
>>> trainer.train()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

from ..models.autoencoder import MultimodalAutoencoder
# Keep backwards compatibility
SherlockAutoencoder = MultimodalAutoencoder
from .losses import CombinedAutoEncoderLoss, compute_correlation_metric, compute_r2_score


@dataclass
class TrainingConfig:
    """
    Configuration for training the autoencoder.

    Parameters
    ----------
    learning_rate : float, default=1e-4
        Initial learning rate
    batch_size : int, default=64
        Batch size per GPU
    num_epochs : int, default=100
        Number of training epochs
    weight_decay : float, default=1e-5
        Weight decay for AdamW optimizer
    reconstruction_weight : float, default=1.0
        Weight for reconstruction loss
    fmri_weight : float, default=1.0
        Weight for fMRI matching loss
    video_weight : float, default=1.0
        Weight for video in reconstruction
    audio_weight : float, default=1.0
        Weight for audio in reconstruction
    text_weight : float, default=1.0
        Weight for text in reconstruction
    fmri_loss_type : str, default='mse'
        Type of fMRI loss: 'mse', 'mae', or 'correlation'
    scheduler_type : str, default='cosine'
        Learning rate scheduler: 'cosine', 'step', or 'none'
    warmup_epochs : int, default=5
        Number of warmup epochs for learning rate
    min_lr : float, default=1e-6
        Minimum learning rate for scheduler
    gradient_clip_val : float, default=1.0
        Gradient clipping value (max norm)
    use_mixed_precision : bool, default=True
        Use mixed precision (FP16) training
    num_workers : int, default=4
        Number of dataloader workers
    pin_memory : bool, default=True
        Pin memory for faster GPU transfer
    checkpoint_dir : str, default='checkpoints'
        Directory for saving checkpoints
    log_dir : str, default='logs'
        Directory for saving logs
    save_every : int, default=5
        Save checkpoint every N epochs
    validate_every : int, default=1
        Run validation every N epochs
    early_stopping_patience : int, default=10
        Early stopping patience (epochs)
    early_stopping_delta : float, default=1e-4
        Minimum improvement for early stopping
    resume_from : str, optional
        Path to checkpoint to resume from
    """

    # Optimizer parameters
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    weight_decay: float = 1e-5

    # Loss weights
    reconstruction_weight: float = 1.0
    fmri_weight: float = 1.0
    video_weight: float = 1.0
    audio_weight: float = 1.0
    text_weight: float = 1.0
    fmri_loss_type: str = 'mse'

    # Learning rate scheduling
    scheduler_type: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training settings
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing and logging
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    save_every: int = 5
    validate_every: int = 1

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4

    # Resume training
    resume_from: Optional[str] = None


class Trainer:
    """
    Trainer for Sherlock autoencoder with multi-GPU support.

    Parameters
    ----------
    model : MultimodalAutoencoder
        Model to train
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    config : TrainingConfig
        Training configuration
    distributed : bool, default=False
        Whether to use distributed training
    local_rank : int, default=0
        Local rank for distributed training
    world_size : int, default=1
        World size for distributed training
    """

    def __init__(
        self,
        model: MultimodalAutoencoder,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Distributed training settings
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (local_rank == 0)

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap model in DDP if distributed
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        # Create dataloaders
        self._create_dataloaders()

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create loss function
        self.criterion = CombinedAutoEncoderLoss(
            reconstruction_weight=config.reconstruction_weight,
            fmri_weight=config.fmri_weight,
            video_weight=config.video_weight,
            audio_weight=config.audio_weight,
            text_weight=config.text_weight,
            fmri_loss_type=config.fmri_loss_type
        ).to(self.device)

        # Create learning rate scheduler
        self._create_scheduler()

        # Mixed precision scaler
        # IMPORTANT (Issue #30): GradScaler is NOT compatible with bfloat16!
        # bfloat16 has same exponent range as float32, so no scaling needed.
        # Only use GradScaler for float16 mixed precision.
        model_dtype = next(model.parameters()).dtype
        use_grad_scaler = config.use_mixed_precision and model_dtype != torch.bfloat16
        self.scaler = GradScaler() if use_grad_scaler else None

        if config.use_mixed_precision and model_dtype == torch.bfloat16:
            if self.is_main_process:
                print("  Using bfloat16 mixed precision WITHOUT GradScaler (bfloat16 doesn't need scaling)")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # History
        self.train_history = []
        self.val_history = []

        # Create directories
        if self.is_main_process:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if specified
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        # Create samplers for distributed training
        if self.distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            # Cosine annealing with warmup
            total_steps = len(self.train_loader) * self.config.num_epochs
            warmup_steps = len(self.train_loader) * self.config.warmup_epochs

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.min_lr
            )
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )

        elif self.config.scheduler_type == 'step':
            # Step decay
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            self.warmup_scheduler = None

        else:  # 'none'
            self.scheduler = None
            self.warmup_scheduler = None

    def _update_lr(self):
        """Update learning rate based on scheduler."""
        if self.scheduler is None:
            return

        # Use warmup scheduler for first few epochs
        if (self.warmup_scheduler is not None and
            self.current_epoch < self.config.warmup_epochs):
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

    def train(self):
        """
        Run full training loop.

        Returns
        -------
        history : dict
            Training history with losses and metrics
        """
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Starting training")
            print("=" * 80)
            print(f"Device: {self.device}")
            print(f"Distributed: {self.distributed}")
            if self.distributed:
                print(f"World size: {self.world_size}")
                print(f"Local rank: {self.local_rank}")
            print(f"Model parameters: {self._count_parameters():,}")
            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Val samples: {len(self.val_dataset)}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Number of epochs: {self.config.num_epochs}")
            print(f"Mixed precision: {self.config.use_mixed_precision}")
            print("=" * 80 + "\n")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_metrics = self._train_epoch()

            # Validate
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate()
            else:
                val_metrics = None

            # Update learning rate
            self._update_lr()

            # Log metrics
            if self.is_main_process:
                self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if self.is_main_process and epoch % self.config.save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_metrics['total_loss'] if val_metrics else None
                )

            # Check early stopping
            if val_metrics is not None:
                self._check_early_stopping(val_metrics['total_loss'])

                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    if self.is_main_process:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Save final checkpoint
        if self.is_main_process:
            self.save_checkpoint(
                epoch=self.config.num_epochs,
                val_loss=self.best_val_loss,
                is_final=True
            )

            print("\n" + "=" * 80)
            print("Training complete!")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            print("=" * 80 + "\n")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss
        }

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns
        -------
        metrics : dict
            Average metrics for the epoch
        """
        self.model.train()

        losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'video_loss': 0.0,
            'audio_loss': 0.0,
            'text_loss': 0.0,
            'fmri_loss': 0.0
        }

        num_batches = len(self.train_loader)

        if self.is_main_process:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
                leave=False
            )
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            text = batch['text'].to(self.device)
            fmri = batch['fmri'].to(self.device)

            # Prepare targets
            batch_size = video.size(0)
            video_flat = video.view(batch_size, -1)  # Flatten if needed

            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                # MEMORY OPTIMIZATION (Issue #30): Use autocast with bfloat16 dtype
                # Model and inputs are already in bfloat16, autocast ensures ops use appropriate precision
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(video, audio, text, fmri_target=fmri)

                    # NUMERICAL STABILITY: Cast to float32 for loss computation
                    # Loss functions need higher precision to avoid numerical issues
                    outputs_fp32 = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v
                                    for k, v in outputs.items()}
                    video_flat_fp32 = video_flat.float()
                    audio_fp32 = audio.float() if torch.is_floating_point(audio) else audio
                    text_fp32 = text.float()
                    fmri_fp32 = fmri.float()

                    loss, loss_dict = self.criterion(
                        outputs_fp32, video_flat_fp32, audio_fp32, text_fp32, fmri_fp32
                    )
            else:
                outputs = self.model(video, audio, text, fmri_target=fmri)
                loss, loss_dict = self.criterion(
                    outputs, video_flat, audio, text, fmri
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
                self.optimizer.step()

            # Accumulate losses
            for key in losses.keys():
                if key in loss_dict:
                    losses[key] += loss_dict[key].item()

            self.global_step += 1

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

        # Average losses
        metrics = {k: v / num_batches for k, v in losses.items()}
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        self.train_history.append(metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns
        -------
        metrics : dict
            Validation metrics
        """
        self.model.eval()

        losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'video_loss': 0.0,
            'audio_loss': 0.0,
            'text_loss': 0.0,
            'fmri_loss': 0.0
        }

        correlations = []
        r2_scores = []

        num_batches = len(self.val_loader)

        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        else:
            pbar = self.val_loader

        for batch in pbar:
            # Move batch to device
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            text = batch['text'].to(self.device)
            fmri = batch['fmri'].to(self.device)

            # Prepare targets
            batch_size = video.size(0)
            video_flat = video.view(batch_size, -1)

            # Forward pass (no gradient computation in validation)
            # MEMORY OPTIMIZATION (Issue #30): Use autocast with bfloat16 for validation too
            if self.config.use_mixed_precision:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(video, audio, text, fmri_target=fmri)

                    # NUMERICAL STABILITY: Cast to float32 for loss computation
                    outputs_fp32 = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v
                                    for k, v in outputs.items()}
                    video_flat_fp32 = video_flat.float()
                    audio_fp32 = audio.float() if torch.is_floating_point(audio) else audio
                    text_fp32 = text.float()
                    fmri_fp32 = fmri.float()

                    loss, loss_dict = self.criterion(
                        outputs_fp32, video_flat_fp32, audio_fp32, text_fp32, fmri_fp32
                    )
            else:
                outputs = self.model(video, audio, text, fmri_target=fmri)
                loss, loss_dict = self.criterion(
                    outputs, video_flat, audio, text, fmri
                )

            # Accumulate losses
            for key in losses.keys():
                if key in loss_dict:
                    losses[key] += loss_dict[key].item()

            # Compute metrics for fMRI prediction
            if 'predicted_fmri' in outputs:
                corr = compute_correlation_metric(
                    outputs['predicted_fmri'], fmri, dim=1
                )
                r2 = compute_r2_score(
                    outputs['predicted_fmri'], fmri, dim=1
                )
                correlations.append(corr.cpu())
                r2_scores.append(r2.cpu())

        # Average losses
        metrics = {k: v / num_batches for k, v in losses.items()}

        # Average metrics
        if correlations:
            correlations = torch.cat(correlations)
            r2_scores = torch.cat(r2_scores)
            metrics['fmri_correlation'] = correlations.mean().item()
            metrics['fmri_r2'] = r2_scores.mean().item()

        self.val_history.append(metrics)

        return metrics

    def _check_early_stopping(self, val_loss: float):
        """Check early stopping criteria."""
        improvement = self.best_val_loss - val_loss

        if improvement > self.config.early_stopping_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics to console and file."""
        log_str = f"Epoch {epoch + 1}/{self.config.num_epochs} | "
        log_str += f"Train Loss: {train_metrics['total_loss']:.4f} | "

        if val_metrics is not None:
            log_str += f"Val Loss: {val_metrics['total_loss']:.4f} | "
            if 'fmri_correlation' in val_metrics:
                log_str += f"fMRI Corr: {val_metrics['fmri_correlation']:.4f} | "

        log_str += f"LR: {train_metrics['lr']:.2e}"

        print(log_str)

        # Save to file
        log_file = Path(self.config.log_dir) / 'training.log'
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        is_final: bool = False
    ):
        """
        Save training checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        val_loss : float, optional
            Validation loss
        is_final : bool, default=False
            Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if val_loss is not None:
            checkpoint['val_loss'] = val_loss

        # Save checkpoint
        if is_final:
            checkpoint_path = Path(self.config.checkpoint_dir) / 'final_checkpoint.pt'
        else:
            checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if val_loss is not None and val_loss <= self.best_val_loss:
            best_path = Path(self.config.checkpoint_dir) / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        """
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model
        self.model_module.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        if self.is_main_process:
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training.

    Parameters
    ----------
    rank : int
        Rank of the current process
    world_size : int
        Total number of processes
    backend : str, default='nccl'
        Distributed backend ('nccl' or 'gloo')
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

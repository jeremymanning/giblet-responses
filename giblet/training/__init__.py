"""
Training module for Sherlock autoencoder.

This module provides comprehensive training functionality including:
- Trainer class with single-GPU and multi-GPU support
- Loss functions for reconstruction and fMRI matching
- Training configuration
- Checkpointing and logging

Example
-------
>>> from giblet.models.autoencoder import create_autoencoder
>>> from giblet.data.dataset import MultimodalDataset
>>> from giblet.training import Trainer, TrainingConfig
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
...     num_epochs=100
... )
>>>
>>> # Train
>>> trainer = Trainer(model, train_dataset, val_dataset, config)
>>> history = trainer.train()
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    setup_distributed,
    cleanup_distributed
)

from .losses import (
    ReconstructionLoss,
    FMRIMatchingLoss,
    CombinedAutoEncoderLoss,
    compute_correlation_metric,
    compute_r2_score
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'setup_distributed',
    'cleanup_distributed',
    'ReconstructionLoss',
    'FMRIMatchingLoss',
    'CombinedAutoEncoderLoss',
    'compute_correlation_metric',
    'compute_r2_score',
]

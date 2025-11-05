#!/usr/bin/env python3
"""
Simple training example for multimodal autoencoder.

This script demonstrates how to use the training system with minimal configuration.
Example uses the Sherlock dataset.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.dataset import MultimodalDataset
from giblet.models.autoencoder import create_autoencoder
from giblet.training import Trainer, TrainingConfig


def main():
    """Run a simple training example."""

    print("\n" + "=" * 80)
    print("Multimodal Autoencoder - Training Example")
    print("=" * 80 + "\n")

    # 1. Create model
    print("Creating model...")
    model = create_autoencoder(
        video_height=90,
        video_width=160,
        audio_mels=128,
        text_dim=1024,
        n_voxels=85810,
        bottleneck_dim=8000,
        reconstruction_weight=1.0,
        fmri_weight=1.0,
    )

    param_count = model.get_parameter_count()
    print(f"Model created with {param_count['total']:,} parameters")
    print(f"  Encoder: {param_count['encoder']:,}")
    print(f"  Decoder: {param_count['decoder']:,}")

    # 2. Load datasets
    print("\nLoading datasets...")
    data_dir = Path("data/")

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please download data first or specify correct path.")
        return

    try:
        # Load training data
        train_dataset = MultimodalDataset(
            data_dir=data_dir,
            subjects="all",  # All 17 subjects
            split="train",  # 80% of data
            apply_hrf=True,  # Convolve stimuli with HRF
            mode="per_subject",
            preprocess=True,
        )

        # Load validation data
        val_dataset = MultimodalDataset(
            data_dir=data_dir,
            subjects="all",
            split="val",  # 20% of data
            apply_hrf=True,
            mode="per_subject",
            preprocess=True,
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Using dummy datasets for demonstration...")

        # Create dummy datasets for testing
        import torch
        from torch.utils.data import Dataset

        class DummyDataset(Dataset):
            def __init__(self, n_samples):
                self.n_samples = n_samples

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return {
                    "video": torch.randn(3, 90, 160),
                    "audio": torch.randn(128),
                    "text": torch.randn(1024),
                    "fmri": torch.randn(85810),
                    "subject_id": 1,
                    "tr_index": idx,
                }

        train_dataset = DummyDataset(n_samples=100)
        val_dataset = DummyDataset(n_samples=20)
        print(
            f"Created dummy datasets: train={len(train_dataset)}, val={len(val_dataset)}"
        )

    # 3. Configure training
    print("\nConfiguring training...")
    config = TrainingConfig(
        # Optimizer
        learning_rate=1e-4,
        batch_size=32,  # Adjust based on GPU memory
        num_epochs=10,  # Small number for demo
        weight_decay=1e-5,
        # Loss weights
        reconstruction_weight=1.0,
        fmri_weight=1.0,
        video_weight=1.0,
        audio_weight=1.0,
        text_weight=1.0,
        fmri_loss_type="mse",
        # Scheduling
        scheduler_type="cosine",
        warmup_epochs=2,
        min_lr=1e-6,
        # Training settings
        gradient_clip_val=1.0,
        use_mixed_precision=True,  # Use FP16 for efficiency
        num_workers=4,
        pin_memory=True,
        # Checkpointing
        checkpoint_dir="checkpoints",
        log_dir="logs",
        save_every=2,
        validate_every=1,
        # Early stopping
        early_stopping_patience=5,
        early_stopping_delta=1e-4,
    )

    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num epochs: {config.num_epochs}")
    print(f"  Mixed precision: {config.use_mixed_precision}")

    # 4. Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        distributed=False,  # Set True for multi-GPU
        local_rank=0,
        world_size=1,
    )

    print(f"Trainer initialized on device: {trainer.device}")

    # 5. Train
    print("\nStarting training...")
    print("=" * 80)

    try:
        history = trainer.train()

        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {history['best_val_loss']:.6f}")
        print("=" * 80 + "\n")

        # Print training summary
        print("Training Summary:")
        print(f"  Total epochs: {len(history['train_history'])}")
        print(f"  Final train loss: {history['train_history'][-1]['total_loss']:.6f}")
        print(f"  Final val loss: {history['val_history'][-1]['total_loss']:.6f}")

        if "fmri_correlation" in history["val_history"][-1]:
            print(
                f"  Final fMRI correlation: {history['val_history'][-1]['fmri_correlation']:.4f}"
            )

        print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
        print(f"Logs saved to: {config.log_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Progress has been saved. You can resume with:")
        print(f"  config.resume_from = '{config.checkpoint_dir}/checkpoint_epoch_N.pt'")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

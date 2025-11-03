#!/usr/bin/env python3
"""
Training script for multimodal autoencoder.

Supports both single-GPU and multi-GPU distributed training.

Usage
-----
Single GPU:
    python scripts/train.py --config examples/train_config.yaml

Multi-GPU (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train.py --config examples/train_config.yaml --distributed

Multi-node multi-GPU:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<addr> --master_port=<port> \\
        scripts/train.py --config examples/train_config.yaml --distributed
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.distributed as dist

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from giblet.models.autoencoder import create_autoencoder
from giblet.data.dataset import MultimodalDataset
from giblet.training.trainer import Trainer, TrainingConfig, setup_distributed, cleanup_distributed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_training_config(config: dict) -> TrainingConfig:
    """Create TrainingConfig from dictionary."""
    # Extract training-specific parameters
    training_params = {
        'learning_rate': config.get('learning_rate', 1e-4),
        'batch_size': config.get('batch_size', 64),
        'num_epochs': config.get('num_epochs', 100),
        'weight_decay': config.get('weight_decay', 1e-5),
        'reconstruction_weight': config.get('reconstruction_weight', 1.0),
        'fmri_weight': config.get('fmri_weight', 1.0),
        'video_weight': config.get('video_weight', 1.0),
        'audio_weight': config.get('audio_weight', 1.0),
        'text_weight': config.get('text_weight', 1.0),
        'fmri_loss_type': config.get('fmri_loss_type', 'mse'),
        'scheduler_type': config.get('scheduler_type', 'cosine'),
        'warmup_epochs': config.get('warmup_epochs', 5),
        'min_lr': config.get('min_lr', 1e-6),
        'gradient_clip_val': config.get('gradient_clip_val', 1.0),
        'use_mixed_precision': config.get('use_mixed_precision', True),
        'num_workers': config.get('num_workers', 4),
        'pin_memory': config.get('pin_memory', True),
        'checkpoint_dir': config.get('checkpoint_dir', 'checkpoints'),
        'log_dir': config.get('log_dir', 'logs'),
        'save_every': config.get('save_every', 5),
        'validate_every': config.get('validate_every', 1),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'early_stopping_delta': config.get('early_stopping_delta', 1e-4),
        'resume_from': config.get('resume_from', None)
    }

    return TrainingConfig(**training_params)


def main():
    parser = argparse.ArgumentParser(description='Train Sherlock autoencoder')
    parser.add_argument(
        '--config',
        type=str,
        default='examples/train_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed training'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (set automatically by torchrun)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Override data directory from config'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    training_config = create_training_config(config)

    # Setup distributed training
    if args.distributed:
        # Configure NCCL for tensor01/tensor02 clusters (Issue #30 fix)
        # Disable shared memory transport to avoid /dev/shm communication errors
        # This forces NCCL to use socket transport which is more reliable
        os.environ['NCCL_SHM_DISABLE'] = '1'  # Disable shared memory
        os.environ['NCCL_P2P_DISABLE'] = '0'  # Keep P2P enabled (NVLink works)
        os.environ['NCCL_DEBUG'] = 'INFO'  # Enable debug logging
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30 min timeout for large model init

        # Get rank from environment (set by torchrun)
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Initialize process group
        setup_distributed(local_rank, world_size)

        is_main_process = (local_rank == 0)
    else:
        local_rank = 0
        world_size = 1
        is_main_process = True

    if is_main_process:
        print("\n" + "=" * 80)
        print("Multimodal Autoencoder Training")
        print("=" * 80)
        print(f"Config file: {args.config}")
        print(f"Distributed: {args.distributed}")
        if args.distributed:
            print(f"World size: {world_size}")
            print(f"Local rank: {local_rank}")
        print("=" * 80 + "\n")

    # Get data directory
    data_dir = args.data_dir or config.get('data', {}).get('data_dir', 'data/')
    data_config = config.get('data', {})

    # Create datasets
    if is_main_process:
        print("Loading datasets...")

    train_dataset = MultimodalDataset(
        data_dir=data_dir,
        subjects=data_config.get('subjects', 'all'),
        split='train',
        apply_hrf=data_config.get('apply_hrf', True),
        mode=data_config.get('mode', 'per_subject'),
        preprocess=True
    )

    val_dataset = MultimodalDataset(
        data_dir=data_dir,
        subjects=data_config.get('subjects', 'all'),
        split='val',
        apply_hrf=data_config.get('apply_hrf', True),
        mode=data_config.get('mode', 'per_subject'),
        preprocess=True
    )

    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    # Create model
    if is_main_process:
        print("\nCreating model...")

    model_config = config.get('model', {})
    model = create_autoencoder(
        video_height=model_config.get('video_height', 90),
        video_width=model_config.get('video_width', 160),
        audio_mels=model_config.get('audio_mels', 128),
        text_dim=model_config.get('text_dim', 1024),
        n_voxels=model_config.get('n_voxels', 85810),
        bottleneck_dim=model_config.get('bottleneck_dim', 8000),
        reconstruction_weight=training_config.reconstruction_weight,
        fmri_weight=training_config.fmri_weight,
        use_encodec=model_config.get('use_encodec', False),
        audio_frames_per_tr=model_config.get('audio_frames_per_tr', 65)
    )

    if is_main_process:
        param_count = model.get_parameter_count()
        print(f"Model parameters: {param_count['total']:,}")
        print(f"  Encoder: {param_count['encoder']:,}")
        print(f"  Decoder: {param_count['decoder']:,}")

    # Create trainer
    if is_main_process:
        print("\nInitializing trainer...")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        distributed=args.distributed,
        local_rank=local_rank,
        world_size=world_size
    )

    # Train
    try:
        history = trainer.train()

        if is_main_process:
            print("\nTraining completed successfully!")
            print(f"Best validation loss: {history['best_val_loss']:.6f}")

    except KeyboardInterrupt:
        if is_main_process:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            trainer.save_checkpoint(
                epoch=trainer.current_epoch,
                val_loss=trainer.best_val_loss,
                is_final=False
            )
            print("Checkpoint saved. You can resume training with --resume_from")

    except Exception as e:
        if is_main_process:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()

        raise

    finally:
        # Cleanup distributed training
        if args.distributed:
            cleanup_distributed()


if __name__ == '__main__':
    main()

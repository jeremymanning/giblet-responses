# Training System for Sherlock Autoencoder

This directory contains the complete training system for the Sherlock autoencoder, supporting both single-GPU and multi-GPU distributed training.

## Overview

The training system provides:

- **Trainer class** with comprehensive training loop
- **Loss functions** for reconstruction and fMRI matching
- **Multi-GPU support** via PyTorch DistributedDataParallel
- **Mixed precision training** (FP16) for efficiency
- **Checkpointing** with save/resume functionality
- **Learning rate scheduling** (cosine annealing with warmup)
- **Early stopping** based on validation loss
- **Logging** of losses and metrics

## Quick Start

### Single GPU Training

```python
from giblet.models.autoencoder import create_autoencoder
from giblet.data.dataset import SherlockDataset
from giblet.training import Trainer, TrainingConfig

# Create model
model = create_autoencoder()

# Load datasets
train_dataset = SherlockDataset('data/', split='train')
val_dataset = SherlockDataset('data/', split='val')

# Configure training
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    use_mixed_precision=True
)

# Train
trainer = Trainer(model, train_dataset, val_dataset, config)
history = trainer.train()
```

### Multi-GPU Training (8 GPUs)

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=8 scripts/train.py --config examples/train_config.yaml --distributed

# Or using the training script directly
python scripts/train.py --config examples/train_config.yaml --distributed
```

## File Structure

```
giblet/training/
├── __init__.py          # Module exports
├── trainer.py           # Trainer class with distributed support
├── losses.py            # Loss functions and metrics
└── README.md            # This file

examples/
└── train_config.yaml    # Example training configuration

scripts/
└── train.py             # Training script with CLI interface

tests/
└── test_training.py     # Comprehensive tests
```

## Configuration

Training is configured via a YAML file or `TrainingConfig` dataclass:

```yaml
# Optimizer parameters
learning_rate: 1.0e-4
batch_size: 64
num_epochs: 100
weight_decay: 1.0e-5

# Loss weights
reconstruction_weight: 1.0
fmri_weight: 1.0
video_weight: 1.0
audio_weight: 1.0
text_weight: 1.0

# Learning rate scheduling
scheduler_type: 'cosine'  # 'cosine', 'step', or 'none'
warmup_epochs: 5
min_lr: 1.0e-6

# Training settings
gradient_clip_val: 1.0
use_mixed_precision: true
num_workers: 4

# Checkpointing
checkpoint_dir: 'checkpoints'
save_every: 5

# Early stopping
early_stopping_patience: 10
```

## Loss Functions

### ReconstructionLoss

Computes MSE loss between reconstructed and original stimuli (video, audio, text):

```python
from giblet.training.losses import ReconstructionLoss

loss_fn = ReconstructionLoss(
    video_weight=1.0,
    audio_weight=1.0,
    text_weight=1.0
)

loss, loss_dict = loss_fn(
    video_recon, video_target,
    audio_recon, audio_target,
    text_recon, text_target
)
```

### FMRIMatchingLoss

Computes loss between predicted and actual fMRI voxel activations:

```python
from giblet.training.losses import FMRIMatchingLoss

# MSE-based loss
loss_fn = FMRIMatchingLoss(loss_type='mse')

# Or correlation-based loss
loss_fn = FMRIMatchingLoss(loss_type='correlation')

loss = loss_fn(predicted_fmri, target_fmri)
```

### CombinedAutoEncoderLoss

Combines reconstruction and fMRI losses with configurable weights:

```python
from giblet.training.losses import CombinedAutoEncoderLoss

loss_fn = CombinedAutoEncoderLoss(
    reconstruction_weight=1.0,
    fmri_weight=1.0
)

loss, loss_dict = loss_fn(outputs, video_target, audio_target, text_target, fmri_target)
```

## Trainer Class

The `Trainer` class handles the complete training loop:

```python
from giblet.training import Trainer, TrainingConfig

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    distributed=False,  # Set True for multi-GPU
    local_rank=0,
    world_size=1
)

# Train
history = trainer.train()

# Access training history
print(f"Best validation loss: {history['best_val_loss']}")
print(f"Train losses: {history['train_history']}")
print(f"Val losses: {history['val_history']}")
```

### Key Features

1. **Automatic Device Management**: Moves model and data to GPU automatically
2. **Mixed Precision**: Uses FP16 when enabled for 2x speedup and lower memory
3. **Gradient Clipping**: Prevents exploding gradients
4. **Learning Rate Scheduling**: Cosine annealing with linear warmup
5. **Checkpointing**: Saves checkpoints periodically and on best validation loss
6. **Early Stopping**: Stops training if validation loss doesn't improve
7. **Progress Bars**: Uses tqdm for visual progress tracking

## Multi-GPU Training

### Setup

For multi-GPU training, you need to:

1. Initialize the distributed process group
2. Use DistributedSampler for data loading
3. Wrap model in DistributedDataParallel

The `Trainer` class handles all of this automatically when `distributed=True`.

### Using torchrun

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 scripts/train.py --config examples/train_config.yaml --distributed

# Multi-node (2 nodes, 8 GPUs each)
# On node 0:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=12355 \
    scripts/train.py --config examples/train_config.yaml --distributed

# On node 1:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=12355 \
    scripts/train.py --config examples/train_config.yaml --distributed
```

### Manual Distributed Setup

```python
import torch.distributed as dist
from giblet.training import setup_distributed, cleanup_distributed

# Initialize process group
setup_distributed(rank=local_rank, world_size=world_size)

# Create trainer with distributed=True
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    distributed=True,
    local_rank=local_rank,
    world_size=world_size
)

# Train
trainer.train()

# Cleanup
cleanup_distributed()
```

## Checkpointing

Checkpoints are saved automatically and include:

- Model state dict
- Optimizer state dict
- Scheduler state dict (if used)
- Mixed precision scaler state dict (if used)
- Training history
- Configuration

### Loading Checkpoints

```python
# Resume training from checkpoint
config = TrainingConfig(
    resume_from='checkpoints/checkpoint_epoch_50.pt',
    # ... other settings
)

trainer = Trainer(model, train_dataset, val_dataset, config)
trainer.train()  # Continues from epoch 50
```

### Checkpoint Files

- `checkpoint_epoch_N.pt`: Saved every `save_every` epochs
- `best_checkpoint.pt`: Saved when validation loss improves
- `final_checkpoint.pt`: Saved at the end of training

## Monitoring Training

### Console Output

The trainer prints progress to console:

```
================================================================================
Starting training
================================================================================
Device: cuda:0
Distributed: True
World size: 8
Model parameters: 1,974,067,762
Train samples: 12512
Val samples: 3128
Batch size: 64
Learning rate: 0.0001
Number of epochs: 100
Mixed precision: True
================================================================================

Epoch 1/100 | Train Loss: 15.3421 | Val Loss: 12.4567 | fMRI Corr: 0.2345 | LR: 1.00e-04
Saved checkpoint: checkpoints/checkpoint_epoch_1.pt
```

### TensorBoard (Optional)

To add TensorBoard logging, install tensorboard:

```bash
pip install tensorboard
```

Then view logs:

```bash
tensorboard --logdir logs/
```

### Weights & Biases (Optional)

To use Weights & Biases, uncomment in requirements.txt and add to trainer:

```python
import wandb

wandb.init(project='sherlock-autoencoder', config=asdict(config))

# In training loop
wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
```

## Hardware Requirements

### Single GPU

- **Minimum**: NVIDIA GPU with 24GB VRAM (RTX 3090, A5000)
- **Recommended**: NVIDIA GPU with 40-48GB VRAM (A6000, A100)

With mixed precision and batch size 32-64:
- Memory usage: ~20-30GB
- Training time: ~2-3 days for 100 epochs

### Multi-GPU (8× A6000)

With 8 GPUs and batch size 64 per GPU (effective batch size 512):
- Total memory: 384GB across GPUs
- Training time: ~6-8 hours for 100 epochs
- Throughput: ~15-20 samples/second

## Performance Tips

1. **Use mixed precision**: Enables 2x speedup with minimal accuracy loss
2. **Optimize batch size**: Larger batches train faster but need more memory
3. **Tune num_workers**: 4-8 workers usually optimal for data loading
4. **Use gradient accumulation**: Simulate larger batches on limited memory
5. **Profile bottlenecks**: Use PyTorch profiler to identify slow operations

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size`
- Enable `use_mixed_precision`
- Reduce `num_workers`
- Use gradient accumulation

### Slow Training

- Increase `batch_size` (if memory allows)
- Enable `use_mixed_precision`
- Use `pin_memory=True`
- Check data loading isn't bottleneck (increase `num_workers`)

### Poor Convergence

- Adjust `learning_rate` (try 1e-5 to 1e-3)
- Tune loss weights (`reconstruction_weight`, `fmri_weight`)
- Increase `warmup_epochs`
- Check gradient norms (may need different `gradient_clip_val`)

### Distributed Training Issues

- Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
- Check network connectivity between nodes
- Verify same Python environment on all nodes
- Use `NCCL_DEBUG=INFO` for detailed logs

## Testing

Run tests to verify the training system:

```bash
# Test loss functions
python -m pytest tests/test_training.py::TestLossFunctions -v

# Test trainer
python -m pytest tests/test_training.py::TestTrainer -v

# Run all tests
python -m pytest tests/test_training.py -v

# Or run directly
python tests/test_training.py
```

## Example Training Session

```bash
# 1. Prepare data
python -c "
from giblet.data.dataset import SherlockDataset
train_ds = SherlockDataset('data/', split='train', preprocess=True)
val_ds = SherlockDataset('data/', split='val', preprocess=True)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')
"

# 2. Edit configuration
vim examples/train_config.yaml

# 3. Start training (single GPU)
python scripts/train.py --config examples/train_config.yaml

# 4. Or multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 scripts/train.py --config examples/train_config.yaml --distributed

# 5. Monitor progress
tail -f logs/training.log

# 6. Resume if interrupted
# Edit train_config.yaml to set resume_from: 'checkpoints/checkpoint_epoch_N.pt'
python scripts/train.py --config examples/train_config.yaml
```

## API Reference

See docstrings in:
- `trainer.py`: Complete Trainer API
- `losses.py`: Loss functions and metrics

## Citation

If you use this training system in your research, please cite:

```bibtex
@software{giblet_training,
  title={Giblet Training System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/giblet-responses}
}
```

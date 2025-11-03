# Giblet-Responses Setup Guide

**Complete setup guide for new users** - from zero to training on cluster in minutes.

Last updated: 2025-10-30

---

## Overview

This guide walks you through setting up the giblet-responses environment from scratch and covers:

- **Local Development**: Setting up on your laptop (macOS/Linux)
- **Quick Testing**: Verifying your installation works
- **Cluster Training**: Running large-scale 8-GPU training on tensor01/tensor02
- **Monitoring & Retrieval**: Checking progress and getting results

---

## Prerequisites

Before starting, ensure you have:

- **Git** installed
- **macOS or Linux** (Windows may work but is untested)
- **For cluster training**: Access credentials for tensor01 or tensor02

---

## Quick Start (Local Setup)

### Automated Setup (Recommended)

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/jeremymanning/giblet-responses.git
cd giblet-responses

# Run automated setup (handles everything)
./setup_environment.sh
```

**What this does:**
- Detects your OS and architecture
- Installs miniconda if needed
- Creates `giblet-py311` conda environment with Python 3.11
- Installs all dependencies (PyTorch, neuroimaging tools, transformers, etc.)
- Offers to download Sherlock dataset (~11GB)
- Verifies installation and runs quick tests

**Time required:** 5-15 minutes (depending on internet speed for dataset)

---

### Manual Setup

If the automated script fails or you prefer manual control:

#### 1. Install Miniconda (if needed)

```bash
# Download and install miniconda for your platform
# Visit: https://docs.conda.io/en/latest/miniconda.html

# For macOS (Intel):
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# For macOS (Apple Silicon):
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# For Linux:
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### 2. Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n giblet-py311 python=3.11 -y

# Activate it
conda activate giblet-py311
```

#### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements_conda.txt
```

#### 4. Download Dataset

```bash
# Download Sherlock dataset (11GB, ~5 min)
./download_data_from_dropbox.sh
```

#### 5. Verify Installation

```bash
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import giblet; print('Giblet module loaded')"

# Optional: Run tests
pytest tests/ -v
```

---

## Testing Your Installation

After setup, verify everything works:

### Quick Test

```bash
# Activate environment (if not already active)
conda activate giblet-py311

# Run quick validation
./run_giblet.sh --task test
```

This runs the test suite to verify all components work correctly.

### Test Local Training

Test training locally with a small config (2 epochs, 1 subject):

```bash
./run_giblet.sh --task train --config test_config.yaml --gpus 1
```

**Expected behavior:**
- Uses 1 GPU (or CPU if no GPU available)
- Trains for 2 epochs on single subject
- Takes ~5-10 minutes
- Creates `test_checkpoints/` and `test_logs/`

**Success indicators:**
- No errors during data loading
- Training progress bars appear
- Loss values decrease
- Checkpoint saved after each epoch

---

## Local Development

### Directory Structure

```
giblet-responses/
├── data/                    # Dataset (gitignored)
│   ├── stimuli_Sherlock.m4v
│   ├── annotations.xlsx
│   └── sherlock_nii/
├── giblet/                  # Main code package
│   ├── models/             # Model architectures
│   ├── data/               # Data loaders
│   ├── alignment/          # Modality synchronization
│   └── training/           # Training utilities
├── scripts/                # Executable scripts
│   └── train.py
├── examples/               # Example configs
│   └── train_config.yaml
├── tests/                  # Test suite
├── checkpoints/            # Saved models (gitignored)
└── logs/                   # Training logs (gitignored)
```

### Running Local Training

#### Single GPU Training

```bash
# Activate environment
conda activate giblet-py311

# Train with example config
./run_giblet.sh --task train --config examples/train_config.yaml --gpus 1
```

#### CPU Training (No GPU)

```bash
./run_giblet.sh --task train --config examples/train_config.yaml --gpus 0
```

**Note:** CPU training is very slow and only recommended for testing.

#### Custom Configuration

Create your own config file:

```bash
# Copy example config
cp examples/train_config.yaml my_config.yaml

# Edit parameters
nano my_config.yaml

# Train with custom config
./run_giblet.sh --task train --config my_config.yaml --gpus 1
```

---

## Cluster Training

For serious training runs, use the tensor01 or tensor02 GPU clusters. Each has 8 NVIDIA GPUs.

### Initial Setup on Cluster

**Good news:** The `remote_train.sh` script handles **everything automatically**:
- SSH connection via sshpass
- Code synchronization via rsync
- Remote environment setup
- Training job launch in screen session

You don't need to manually SSH or set up the cluster!

### Cluster Credentials

**First time only:** Create credentials file for your cluster.

#### 1. Create credentials directory

```bash
mkdir -p cluster_config
```

#### 2. Create credentials file

For **tensor01**:

```bash
cat > cluster_config/tensor01_credentials.json << 'EOF'
{
  "server": "tensor01.dartmouth.edu",
  "username": "YOUR_USERNAME",
  "password": "YOUR_PASSWORD",
  "base_path": "/home/YOUR_USERNAME/giblet-responses"
}
EOF
```

For **tensor02**:

```bash
cat > cluster_config/tensor02_credentials.json << 'EOF'
{
  "server": "tensor02.dartmouth.edu",
  "username": "YOUR_USERNAME",
  "password": "YOUR_PASSWORD",
  "base_path": "/home/YOUR_USERNAME/giblet-responses"
}
EOF
```

**Replace:**
- `YOUR_USERNAME` with your cluster username
- `YOUR_PASSWORD` with your cluster password

**Security:** The `cluster_config/` directory is gitignored - your credentials stay private.

---

### Launching Training on Cluster

#### Basic 8-GPU Training

Train on all 8 GPUs with default config:

```bash
# On tensor01
./remote_train.sh --cluster tensor01 --config examples/train_config.yaml --gpus 8 --name my_training

# Or on tensor02
./remote_train.sh --cluster tensor02 --config examples/train_config.yaml --gpus 8 --name my_training
```

**What this does:**
1. Tests SSH connection
2. Syncs your code to cluster via rsync
3. Verifies/creates remote environment
4. Launches training in screen session named `my_training`
5. Training continues even if you disconnect

#### Custom Configuration

```bash
# Create cluster-optimized config
cp examples/train_config.yaml cluster_config.yaml

# Edit parameters (larger batch size, more epochs, etc.)
nano cluster_config.yaml

# Launch on cluster
./remote_train.sh --cluster tensor01 --config cluster_config.yaml --gpus 8 --name big_run
```

#### Using Fewer GPUs

```bash
# Use 4 GPUs instead of 8
./remote_train.sh --cluster tensor01 --config my_config.yaml --gpus 4 --name test_run
```

#### Dry Run (Test Without Executing)

```bash
# See what would happen without actually running
./remote_train.sh --cluster tensor01 --config my_config.yaml --gpus 8 --dry-run
```

---

### Monitoring Training

#### Check Training Status

From your local machine:

```bash
./check_remote_status.sh --cluster tensor01
```

This shows:
- Running screen sessions
- GPU utilization
- Recent log output

#### Attach to Training Session

```bash
# SSH to cluster
ssh YOUR_USERNAME@tensor01.dartmouth.edu

# List screen sessions
screen -ls

# Attach to your training session
screen -r my_training

# Watch training in real-time
# Press Ctrl+A then D to detach (training continues)
```

#### View Logs

```bash
# SSH to cluster
ssh YOUR_USERNAME@tensor01.dartmouth.edu

# View live training logs
cd ~/giblet-responses
tail -f logs/training_my_training_*.log

# Or view last 100 lines
tail -100 logs/training_my_training_*.log
```

#### Check GPU Utilization

```bash
# SSH to cluster
ssh YOUR_USERNAME@tensor01.dartmouth.edu

# Real-time GPU monitoring
nvidia-smi

# Or continuous updates every 2 seconds
watch -n 2 nvidia-smi
```

---

### Resuming Training

If training crashes or you need to continue from a checkpoint:

```bash
# Resume from last checkpoint
./remote_train.sh --cluster tensor01 --config my_config.yaml --resume --name my_training
```

**What this does:**
- Backs up existing checkpoints before starting
- Loads latest checkpoint from `checkpoints/` directory
- Continues training from that point

---

### Killing Training Session

If you need to stop training:

```bash
# Kill the training session
./remote_train.sh --cluster tensor01 --name my_training --kill

# Or manually via SSH
ssh YOUR_USERNAME@tensor01.dartmouth.edu
screen -X -S my_training quit
```

---

### Retrieving Results

After training completes, sync results back to your local machine:

#### Using sshpass (Recommended)

```bash
# Sync checkpoints
sshpass -p YOUR_PASSWORD rsync -avz --progress \
  YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/ \
  ./checkpoints/

# Sync logs
sshpass -p YOUR_PASSWORD rsync -avz --progress \
  YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/logs/ \
  ./logs/
```

#### Using scp

```bash
# Download checkpoints
scp -r YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/ ./checkpoints/

# Download logs
scp -r YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/logs/ ./logs/
```

#### Selective Download

```bash
# Download only specific checkpoint
scp YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/checkpoint_epoch_100.pt \
  ./checkpoints/

# Download only recent logs (last 3 days)
rsync -avz --progress --include="*/" --include="*.log" \
  --exclude="*" --prune-empty-dirs \
  --min-age=3d \
  YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/logs/ \
  ./logs/
```

---

## Configuration

### Training Configuration Files

Training is controlled by YAML config files. Here's the structure:

```yaml
# Optimizer parameters
learning_rate: 1.0e-4
batch_size: 64           # Per-GPU batch size
num_epochs: 100
weight_decay: 1.0e-5

# Loss weights
reconstruction_weight: 1.0
fmri_weight: 1.0
video_weight: 1.0
audio_weight: 1.0
text_weight: 1.0
fmri_loss_type: 'mse'    # 'mse', 'mae', or 'correlation'

# Learning rate scheduling
scheduler_type: 'cosine' # 'cosine', 'step', or 'none'
warmup_epochs: 5
min_lr: 1.0e-6

# Training settings
gradient_clip_val: 1.0
use_mixed_precision: true
num_workers: 4
pin_memory: true

# Checkpointing
checkpoint_dir: 'checkpoints'
log_dir: 'logs'
save_every: 5            # Save checkpoint every N epochs
validate_every: 1        # Validate every N epochs

# Early stopping
early_stopping_patience: 10
early_stopping_delta: 1.0e-4

# Data settings
data:
  data_dir: 'data/'
  subjects: 'all'        # 'all' or [1, 2, 3, ...]
  apply_hrf: true
  mode: 'per_subject'    # 'per_subject' or 'cross_subject'

# Model architecture
model:
  video_height: 90
  video_width: 160
  audio_mels: 2048
  text_dim: 1024
  n_voxels: 85810
  bottleneck_dim: 8000
  video_features: 1024
  audio_features: 256
  text_features: 256
  decoder_hidden_dim: 2048
  decoder_dropout: 0.3

# Distributed training (auto-configured by run_giblet.sh)
distributed:
  enabled: false         # Set by --gpus > 1
  backend: 'nccl'        # 'nccl' for GPU, 'gloo' for CPU
  world_size: 8          # Number of GPUs
```

### Key Parameters to Tune

**For testing:**
- `batch_size: 2` (small batches)
- `num_epochs: 2` (quick run)
- `subjects: [1]` (single subject)

**For small runs:**
- `batch_size: 32`
- `num_epochs: 50`
- `subjects: [1, 2, 3, 4]`

**For full training:**
- `batch_size: 64` (or higher with 8 GPUs)
- `num_epochs: 100-200`
- `subjects: 'all'` (all 17 subjects)

### Cluster Credentials

Stored in `cluster_config/CLUSTER_credentials.json`:

```json
{
  "server": "tensor01.dartmouth.edu",
  "username": "YOUR_USERNAME",
  "password": "YOUR_PASSWORD",
  "base_path": "/home/YOUR_USERNAME/giblet-responses"
}
```

**Important:**
- This directory is gitignored - credentials never go to GitHub
- Create separate files for tensor01 and tensor02
- Keep passwords secure (consider using SSH keys instead)

---

## Troubleshooting

### Common Issues

#### 1. Conda Environment Issues

**Problem:** `conda: command not found`

**Solution:**
```bash
# Add conda to PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Or reinitialize
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

#### 2. CUDA/GPU Issues

**Problem:** `No CUDA-capable device detected`

**Solution:**
```bash
# Check if PyTorch sees GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check GPU status
nvidia-smi

# If False, reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Data Download Failures

**Problem:** `Download failed or file is empty`

**Solution:**
```bash
# Delete partial download and retry
rm sherlock_dataset.zip
./download_data_from_dropbox.sh

# Or download manually from:
# https://www.dropbox.com/scl/fi/tlr6orirwc14gdq7yqwcl/sherlock_dataset.zip?rlkey=82h9hyrbv37xvtqff5mw6h6m5&dl=1
```

#### 4. Screen Session Issues

**Problem:** Screen session won't attach

**Solution:**
```bash
# List all sessions
screen -ls

# Kill zombie sessions
screen -wipe

# Force quit specific session
screen -X -S session_name quit

# Create new session
screen -S new_session
```

#### 5. SSH Connection Problems

**Problem:** `SSH connection failed`

**Solution:**
```bash
# Test SSH manually
ssh YOUR_USERNAME@tensor01.dartmouth.edu

# Check credentials file
cat cluster_config/tensor01_credentials.json

# Install sshpass (macOS)
brew install hudochenkov/sshpass/sshpass

# Install sshpass (Linux)
sudo apt-get install sshpass
```

#### 6. Out of Memory (OOM)

**Problem:** `CUDA out of memory`

**Solution:**
- Reduce `batch_size` in config (e.g., 64 → 32 → 16)
- Enable `use_mixed_precision: true`
- Reduce `num_workers` (e.g., 4 → 2)
- Use gradient checkpointing (advanced)

#### 7. Training Hangs/Freezes

**Problem:** Training stops progressing

**Solution:**
```bash
# Check if GPU is actually working
ssh YOUR_USERNAME@tensor01.dartmouth.edu
nvidia-smi

# Check logs for errors
tail -100 logs/training_*.log

# Kill and restart
screen -X -S my_training quit
./remote_train.sh --cluster tensor01 --resume --name my_training
```

---

## Advanced Usage

### Custom Training Configs

Create specialized configs for different scenarios:

#### Quick Testing Config

```yaml
# test_config.yaml
batch_size: 2
num_epochs: 2
data:
  subjects: [1]
checkpoint_dir: 'test_checkpoints'
log_dir: 'test_logs'
```

#### Small-Scale Config

```yaml
# small_config.yaml
batch_size: 32
num_epochs: 50
data:
  subjects: [1, 2, 3, 4]
```

#### Full-Scale Cluster Config

```yaml
# cluster_config.yaml
batch_size: 128         # 8 GPUs × 16 per GPU
num_epochs: 200
data:
  subjects: 'all'       # All 17 subjects
distributed:
  enabled: true
  world_size: 8
```

### Multi-Node Training (Future)

Currently not implemented, but planned:

```bash
# Future: Train across tensor01 + tensor02 (16 GPUs)
./remote_train.sh --clusters tensor01,tensor02 --gpus 16 --config large_config.yaml
```

### Checkpointing Strategy

**Automatic checkpointing:**
- Saves every N epochs (controlled by `save_every`)
- Keeps best model based on validation loss
- Saves optimizer state for resuming

**Checkpoint files:**
```
checkpoints/
├── checkpoint_latest.pt      # Most recent
├── checkpoint_best.pt        # Best validation loss
├── checkpoint_epoch_50.pt    # Epoch 50
└── checkpoint_epoch_100.pt   # Epoch 100
```

**Resume from specific checkpoint:**
```yaml
# In config.yaml
resume_from: 'checkpoints/checkpoint_epoch_50.pt'
```

---

## Development Workflow

Recommended workflow for efficient development:

### 1. Develop Locally

```bash
# Work on code locally
cd giblet-responses
conda activate giblet-py311

# Edit code in giblet/ directory
nano giblet/models/autoencoder.py

# Run tests
pytest tests/test_models.py -v
```

### 2. Test with Small Config

```bash
# Quick test on local machine
./run_giblet.sh --task train --config test_config.yaml --gpus 1
```

### 3. Commit and Push

```bash
# Commit changes
git add .
git commit -m "Update autoencoder architecture"
git push origin main
```

### 4. Train on Cluster

```bash
# Launch full training on cluster
./remote_train.sh --cluster tensor01 --config cluster_config.yaml --gpus 8 --name experiment_v2
```

### 5. Monitor and Retrieve

```bash
# Check status
./check_remote_status.sh --cluster tensor01

# Attach to session to see progress
ssh YOUR_USERNAME@tensor01.dartmouth.edu
screen -r experiment_v2
# Ctrl+A, D to detach

# Sync results when done
sshpass -p PASSWORD rsync -avz \
  YOUR_USERNAME@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/ \
  ./checkpoints/
```

---

## Additional Resources

### Documentation

- **README.md** - Project overview and research goals
- **CLAUDE.md** - Project instructions and development guide
- **ENVIRONMENT_SETUP.md** - Detailed dependency setup
- **STATUS.md** - Current project status

### GitHub Repository

- **Repository:** https://github.com/jeremymanning/giblet-responses
- **Issues:** https://github.com/jeremymanning/giblet-responses/issues
- **Issue #19:** Cluster deployment (this guide addresses it)

### Slack Communication

- **Channel:** https://context-lab.slack.com/archives/C020V4HJFT4
- For questions, discussions, and coordination

### Key References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Distributed Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Nilearn fMRI Analysis](https://nilearn.github.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Getting Help

If you encounter issues:

1. **Check troubleshooting section** above
2. **Search GitHub issues** for similar problems
3. **Ask in Slack channel** for quick help
4. **Create GitHub issue** for bugs or feature requests

---

## Summary

**Quick reference for common commands:**

```bash
# Local setup
./setup_environment.sh
conda activate giblet-py311

# Local testing
./run_giblet.sh --task train --config test_config.yaml --gpus 1

# Cluster training (8 GPUs)
./remote_train.sh --cluster tensor01 --config cluster_config.yaml --gpus 8 --name run_name

# Monitor
./check_remote_status.sh --cluster tensor01
ssh user@tensor01.dartmouth.edu && screen -r run_name

# Retrieve results
sshpass -p PASSWORD rsync -avz user@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/ ./checkpoints/

# Resume training
./remote_train.sh --cluster tensor01 --resume --name run_name

# Kill training
./remote_train.sh --cluster tensor01 --kill --name run_name
```

---

**Last updated:** 2025-10-30
**Related issue:** #19 (Cluster deployment)

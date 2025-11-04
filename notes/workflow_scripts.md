# Workflow Scripts - Always Use These!

**IMPORTANT**: Always use the existing scripts instead of making custom SSH/rsync calls.

## Script Usage

### 1. `run_giblet.sh` - Local and Remote Execution
**Purpose**: Main script for running all tasks (training, testing, validation)
**Used by**: Both locally and remotely (called by remote_train.sh on clusters)

**Usage**:
```bash
# Local training
./run_giblet.sh --task train --config configs/training/test_config.yaml --gpus 1

# Distributed training (8 GPUs)
./run_giblet.sh --task train --config configs/training/production_config.yaml --gpus 8

# Run tests
./run_giblet.sh --task test

# Validate data
./run_giblet.sh --task validate_data
```

### 2. `remote_train.sh` - Cluster Training
**Purpose**: Launch training on remote clusters (tensor01/tensor02)
**Features**:
- SSH authentication
- Code synchronization via rsync
- Screen session management
- Calls run_giblet.sh on the remote server

**Usage**:
```bash
# Launch on tensor01
./remote_train.sh --cluster tensor01 --config configs/training/production_config.yaml --gpus 8

# Kill existing session and restart
./remote_train.sh --cluster tensor01 --config configs/training/test_config.yaml --gpus 8 --kill

# Resume from checkpoint
./remote_train.sh --cluster tensor01 --config configs/training/production_config.yaml --gpus 8 --resume

# Custom session name
./remote_train.sh --cluster tensor01 --config configs/training/test_config.yaml --name my_test --gpus 4
```

### 3. `check_remote_status.sh` - Status Monitoring
**Purpose**: Check status of training jobs on remote clusters

**Usage**:
```bash
# Check status on tensor01
./check_remote_status.sh tensor01

# Check status on tensor02
./check_remote_status.sh tensor02
```

### 4. `sync_checkpoints.sh` - Download Trained Models
**Purpose**: Synchronize checkpoints and logs from remote clusters to local machine

**Usage**:
```bash
# Download from specific checkpoint directory
./sync_checkpoints.sh --cluster tensor01 --checkpoint-dir production_500epoch_checkpoints

# Download logs only
./sync_checkpoints.sh --cluster tensor01 --logs-only

# Download everything
./sync_checkpoints.sh --cluster tensor01 --checkpoint-dir test_8gpu_checkpoints --include-logs
```

## DO NOT Do These:

❌ Direct SSH commands for training
❌ Manual rsync calls for code sync
❌ Custom nohup/screen commands
❌ Ad-hoc checkpoint downloads

## DO Use These:

✅ `remote_train.sh` for launching cluster training
✅ `check_remote_status.sh` for monitoring
✅ `sync_checkpoints.sh` for downloading results
✅ `run_giblet.sh` for all local operations

## Workflow Example

```bash
# 1. Launch production training
./remote_train.sh --cluster tensor01 --config configs/training/production_500epoch_config.yaml --gpus 8 --name production_500epoch

# 2. Monitor progress
./check_remote_status.sh tensor01

# 3. Download checkpoints when complete
./sync_checkpoints.sh --cluster tensor01 --checkpoint-dir production_500epoch_checkpoints --include-logs

# 4. Verify checkpoint locally
python verify_checkpoint.py production_500epoch_checkpoints/best_checkpoint.pt
```

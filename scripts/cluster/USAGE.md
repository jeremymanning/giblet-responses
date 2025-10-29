# Cluster Deployment Scripts Usage Guide

This guide explains how to use the cluster management scripts for tensor01 and tensor02.

## Overview

The cluster scripts provide a complete workflow for:
1. Setting up the cluster environment
2. Submitting training jobs via SLURM
3. Monitoring job progress
4. Syncing results back to your local machine

## Prerequisites

- `sshpass` installed (for credential-based SSH)
- Credentials configured in `cluster_config/tensor01_credentials.json` and `cluster_config/tensor02_credentials.json`
- conda/anaconda installed on the clusters
- SLURM job scheduler available on the clusters

### Install sshpass (macOS)

```bash
brew install sshpass
```

## Quick Start

### 1. Initial Cluster Setup

Before running any jobs, set up the cluster environment:

```bash
./setup_cluster.sh tensor01
# or
./setup_cluster.sh tensor02
```

This script will:
- Test SSH connection
- Create project directories on the cluster
- Verify/create conda environment (`giblet-env`)
- Sync code from your local machine
- Install Python dependencies from `requirements.txt`
- Download the dataset via `download_data_from_dropbox.sh`

### 2. Submit a Training Job

```bash
./submit_job.sh tensor01 my_training demo_decoder.py --epochs 100
```

Parameters:
- `tensor01`: Target cluster (tensor01 or tensor02)
- `my_training`: Job name for SLURM
- `demo_decoder.py`: Script path relative to project root
- `--epochs 100`: Arguments passed to the Python script

The script will:
- Create a SLURM batch script requesting 8 A6000 GPUs
- Copy the script to the cluster
- Submit the job to SLURM
- Return the job ID

**Example with different scripts:**

```bash
# Submit training job
./submit_job.sh tensor01 training_v1 train.py --model bert-base --batch_size 32

# Submit decoding job
./submit_job.sh tensor02 decoder_v2 demo_decoder.py --input data.pkl

# Submit with multiple arguments
./submit_job.sh tensor01 experiment1 run_experiment.py --config config.yaml --gpu 0,1,2,3
```

### 3. Monitor Job Status

```bash
./monitor_job.sh tensor01 12345
```

Parameters:
- `tensor01`: Cluster name
- `12345`: Job ID from submit_job.sh

Options:
- `--tail`: Follow log output in real-time (like `tail -f`)
- `--error`: Show error log instead of output log

**Examples:**

```bash
# Check job status (shows last 50 lines of log)
./monitor_job.sh tensor01 12345

# Follow job in real-time
./monitor_job.sh tensor01 12345 --tail

# Watch error log
./monitor_job.sh tensor01 12345 --error --tail

# Check job on tensor02
./monitor_job.sh tensor02 67890
```

### 4. Retrieve Results

```bash
./sync_results.sh tensor01
```

This downloads:
- `checkpoints_tensor01/` - Saved model checkpoints
- `results_tensor01/` - Result files and metrics
- `logs_tensor01/` - SLURM log files
- `output_tensor01/` - Any output files

**Options:**

```bash
# Dry run (shows what would be synced)
./sync_results.sh tensor01 --dry-run

# Sync from tensor02
./sync_results.sh tensor02

# Sync with dry-run
./sync_results.sh tensor02 --dry-run
```

## Complete Workflow Example

Here's a typical workflow from start to finish:

```bash
# Navigate to scripts directory
cd scripts/cluster

# 1. Setup the cluster (one-time)
./setup_cluster.sh tensor01
echo "Setup complete! Waiting 30 seconds..."
sleep 30

# 2. Submit a job
JOB_OUTPUT=$(./submit_job.sh tensor01 exp1 demo_decoder.py --epochs 50)
# Extract job ID from output
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')

# 3. Monitor the job (check every 30 seconds)
for i in {1..10}; do
    echo "Check $i..."
    ./monitor_job.sh tensor01 $JOB_ID --tail
    sleep 30
done

# 4. Once job completes, sync results
./sync_results.sh tensor01

# 5. Review results
ls -la ../results_tensor01/
cat ../logs_tensor01/*.out
```

## Manual SSH Access

If you need to SSH into the cluster manually:

```bash
# Read password from credentials JSON
PASSWORD=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['password'])")
USERNAME=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['username'])")

# SSH with sshpass
sshpass -p "$PASSWORD" ssh $USERNAME@tensor01.dartmouth.edu
```

## Advanced Usage

### Submit Multiple Jobs

```bash
# Submit jobs in parallel
./submit_job.sh tensor01 exp1 train.py --seed 1 &
./submit_job.sh tensor01 exp2 train.py --seed 2 &
./submit_job.sh tensor02 exp3 train.py --seed 3 &

wait  # Wait for all to complete

# Or collect job IDs
JOB_1=$(./submit_job.sh tensor01 exp1 train.py --seed 1 | grep -oP '(?<=Job ID: )\d+')
JOB_2=$(./submit_job.sh tensor01 exp2 train.py --seed 2 | grep -oP '(?<=Job ID: )\d+')

# Monitor both
./monitor_job.sh tensor01 $JOB_1
./monitor_job.sh tensor01 $JOB_2
```

### Check Cluster Resource Status

```bash
# SSH and check available GPUs
PASSWORD=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['password'])")
USERNAME=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['username'])")

sshpass -p "$PASSWORD" ssh $USERNAME@tensor01.dartmouth.edu "nvidia-smi"
sshpass -p "$PASSWORD" ssh $USERNAME@tensor01.dartmouth.edu "squeue"
```

### Sync Before Submitting Job

```bash
# Sync code changes to cluster before submitting
./setup_cluster.sh tensor01
# This will re-sync code and update dependencies

# Then submit job
./submit_job.sh tensor01 updated_job train.py
```

## SLURM Job Configuration

The scripts request the following resources by default:

```
- Partition: gpu
- Nodes: 1
- Tasks: 1
- CPUs per task: 16
- GPUs per node: 8 (A6000)
- Memory: 128GB
- Walltime: 24 hours
```

To modify these, edit the SLURM section in `submit_job.sh`:

```bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
```

## Troubleshooting

### SSH Connection Issues

```bash
# Test SSH connection manually
sshpass -p "PASSWORD" ssh -o StrictHostKeyChecking=no username@tensor01.dartmouth.edu "echo 'test'"

# Check if sshpass is installed
command -v sshpass

# If not installed (macOS):
brew install sshpass
```

### Job Submission Fails

1. Check credentials in `cluster_config/tensor01_credentials.json`
2. Verify script path exists relative to project root
3. Check SLURM availability: `sshpass -p "PASSWORD" ssh user@server "which sbatch"`

### Low Disk Space

```bash
# Check disk usage on cluster
sshpass -p "PASSWORD" ssh user@tensor01.dartmouth.edu "df -h ~"

# Clean up old log files
sshpass -p "PASSWORD" ssh user@tensor01.dartmouth.edu "rm -f ~/giblet-responses/slurm_logs/*.out"
```

### Missing Data on Cluster

If the data download failed during setup:

```bash
# SSH into cluster
sshpass -p "PASSWORD" ssh user@tensor01.dartmouth.edu

# Run download manually
cd ~/giblet-responses
bash download_data_from_dropbox.sh
```

## Security Notes

1. **Credentials**: Keep `cluster_config/*.json` files secure - they contain passwords
2. **Git**: Ensure credentials files are in `.gitignore` to prevent accidental commits
3. **Logs**: SLURM logs may contain sensitive information - review before sharing
4. **sshpass**: While convenient, consider using SSH keys for production deployments

## Reference Scripts

- `setup_cluster.sh` - One-time cluster environment setup
- `submit_job.sh` - Submit training/inference jobs to SLURM
- `monitor_job.sh` - Check job status and view logs
- `sync_results.sh` - Download results from cluster
- `utils.sh` - Shared utility functions (sourced by other scripts)

## See Also

- [SLURM Documentation](https://slurm.schedmd.com/)
- [Conda Documentation](https://docs.conda.io/)
- [GPU Cluster Best Practices](https://github.com/ContextLab/llm-stylometry)

## Support

For issues or questions, check:
1. Job logs: `./monitor_job.sh <cluster> <job_id> --tail`
2. SLURM status: `sshpass -p "PWD" ssh user@server "squeue"`
3. Cluster resources: `sshpass -p "PWD" ssh user@server "nvidia-smi"`

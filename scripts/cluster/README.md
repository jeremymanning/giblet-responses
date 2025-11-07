# Cluster Management Scripts

This directory contains scripts for managing training and inference jobs on the Dartmouth tensor01 and tensor02 GPU clusters.

## Features

- **Automated setup** of cluster environments (conda, dependencies, code sync)
- **Easy job submission** to SLURM with GPU resource allocation
- **Real-time job monitoring** with log streaming
- **Convenient result retrieval** using rsync
- **Credential-based SSH** for seamless authentication

## Quick Start

### 1. Verify Credentials

Check that credential files exist:

```bash
ls -la ../../../cluster_config/tensor01_credentials.json
ls -la ../../../cluster_config/tensor02_credentials.json
```

These files contain:
- `server`: tensor01.dartmouth.edu or tensor02.dartmouth.edu
- `username`: Your Dartmouth cluster username
- `password`: Your SSH password
- `base_path`: ~/giblet-responses
- `gpus`: 8
- `gpu_type`: A6000

### 2. Install sshpass (if needed)

```bash
# macOS
brew install sshpass

# Linux
sudo apt-get install sshpass  # Debian/Ubuntu
sudo yum install sshpass       # RedHat/CentOS
```

### 3. Setup Cluster

First-time setup (one per cluster):

```bash
./setup_cluster.sh tensor01
# or
./setup_cluster.sh tensor02
```

This:
- Tests SSH connection
- Creates project directories
- Creates conda environment
- Syncs your code
- Installs Python dependencies
- Downloads the dataset

### 4. Submit a Job

```bash
./submit_job.sh tensor01 my_job demo_decoder.py --epochs 100
```

### 5. Monitor Job

```bash
./monitor_job.sh tensor01 12345      # Check status
./monitor_job.sh tensor01 12345 --tail # Follow output
```

### 6. Get Results

```bash
./sync_results.sh tensor01
# Results appear in: results_tensor01/, checkpoints_tensor01/, logs_tensor01/
```

## Scripts Overview

### setup_cluster.sh

**Purpose**: One-time cluster environment setup

**Usage**:
```bash
./setup_cluster.sh <tensor01|tensor02>
```

**What it does**:
1. Tests SSH connection
2. Creates project directories on cluster
3. Verifies/creates conda environment (giblet-env)
4. Syncs code from local machine
5. Installs Python dependencies (requirements.txt)
6. Downloads dataset from Dropbox

**When to use**: Once per cluster, or when you need to resync code/dependencies

### submit_job.sh

**Purpose**: Submit training/inference jobs to SLURM

**Usage**:
```bash
./submit_job.sh <tensor01|tensor02> <job_name> <script_path> [python_args...]
```

**Examples**:
```bash
# Simple job
./submit_job.sh tensor01 training demo_decoder.py

# With arguments
./submit_job.sh tensor01 exp1 train.py --epochs 100 --batch_size 32

# With multiple argument types
./submit_job.sh tensor02 decoder demo_decoder.py --model bert --output results.pkl
```

**Job Configuration**:
- **Partition**: gpu
- **Nodes**: 1
- **GPUs**: 8 × A6000
- **Memory**: 128GB
- **Time limit**: 24 hours
- **CPUs**: 16

**Output**: Job ID and job information file (`.job_info_cluster_jobid.txt`)

### monitor_job.sh

**Purpose**: Check job status and view logs

**Usage**:
```bash
./monitor_job.sh <tensor01|tensor02> <job_id> [options]
```

**Options**:
- `--tail`: Follow log in real-time (like `tail -f`)
- `--error`: Show error log instead of output log

**Examples**:
```bash
./monitor_job.sh tensor01 12345              # Status check
./monitor_job.sh tensor01 12345 --tail       # Follow output
./monitor_job.sh tensor01 12345 --error      # Show errors
./monitor_job.sh tensor02 67890 --error --tail # Follow error log
```

**Features**:
- Shows current job queue status
- Displays job history (for completed jobs)
- Shows detailed job information
- Displays log files
- Can follow logs in real-time

### sync_results.sh

**Purpose**: Download results, checkpoints, and logs from cluster

**Usage**:
```bash
./sync_results.sh <tensor01|tensor02> [options]
```

**Options**:
- `--dry-run`: Preview what would be synced without copying

**Examples**:
```bash
./sync_results.sh tensor01              # Download all results
./sync_results.sh tensor02 --dry-run    # Preview downloads
```

**Downloads to**:
- `results_tensor01/` - Result files
- `checkpoints_tensor01/` - Model checkpoints
- `logs_tensor01/` - SLURM output/error logs
- `output_tensor01/` - Other output files

### utils.sh

**Purpose**: Shared utility functions for other scripts

**Functions** (can be sourced):
- `print_info()`, `print_success()`, `print_warning()`, `print_error()` - Colored output
- `read_json()` - Extract values from credential JSON
- `validate_cluster()` - Validate cluster name
- `test_ssh_connection()` - Test SSH access
- `load_credentials()` - Load credentials into environment variables
- And more...

### example_workflow.sh

**Purpose**: Interactive example demonstrating the complete workflow

**Usage**:
```bash
./example_workflow.sh
```

**Demonstrates**:
1. Cluster setup
2. Job submission
3. Job monitoring
4. Results retrieval

## Common Workflows

### Single Job Submission

```bash
cd /path/to/giblet-responses/scripts/cluster

# Setup (one-time)
./setup_cluster.sh tensor01

# Submit job
./submit_job.sh tensor01 training1 train.py --epochs 100

# Monitor (replace 12345 with actual job ID)
./monitor_job.sh tensor01 12345 --tail

# Get results when done
./sync_results.sh tensor01
```

### Multiple Parameter Sweeps

```bash
# Submit multiple jobs with different seeds
for seed in 1 2 3 4 5; do
  ./submit_job.sh tensor01 seed_$seed train.py --seed $seed &
done
wait

# Or with job ID collection
JOB_IDS=()
for seed in 1 2 3 4 5; do
  JOB_ID=$(./submit_job.sh tensor01 seed_$seed train.py --seed $seed | grep -oP '(?<=Job ID: )\d+')
  JOB_IDS+=($JOB_ID)
done

# Monitor all jobs
for JID in "${JOB_IDS[@]}"; do
  ./monitor_job.sh tensor01 $JID
done

# Sync all results
./sync_results.sh tensor01
```

### Distributed Jobs (using both clusters)

```bash
# Setup both clusters
./setup_cluster.sh tensor01
./setup_cluster.sh tensor02

# Submit jobs to both
./submit_job.sh tensor01 exp1 train.py --seed 1 &
./submit_job.sh tensor02 exp2 train.py --seed 2 &
wait

# Sync results from both
./sync_results.sh tensor01
./sync_results.sh tensor02
```

### Code Update and Resubmit

```bash
# Resync code (setup includes full sync)
./setup_cluster.sh tensor01

# Submit updated job
./submit_job.sh tensor01 updated_v2 train.py --epochs 200
```

## Credentials File Format

The credential JSON files should look like:

```json
{
  "cluster_name": "tensor01",
  "server": "tensor01.dartmouth.edu",
  "username": "f002d6b",
  "password": "your_password_here",
  "base_path": "~/giblet-responses",
  "gpus": 8,
  "gpu_type": "A6000"
}
```

**Security Note**: These files contain passwords - ensure they're in `.gitignore`:

```bash
# In project .gitignore
cluster_config/*credentials.json
```

## Troubleshooting

### sshpass not found

```bash
# macOS
brew install sshpass

# Linux (Ubuntu)
sudo apt-get install sshpass
```

### Connection refused

```bash
# Test SSH manually
PASSWORD=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['password'])")
USERNAME=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['username'])")

sshpass -p "$PASSWORD" ssh $USERNAME@tensor01.dartmouth.edu "echo test"
```

### Job not submitting

1. Check script exists: `ls -la ../../demo_decoder.py`
2. Verify credentials: `cat ../../cluster_config/tensor01_credentials.json`
3. Test SSH: See above
4. Check SLURM: `sshpass -p "PWD" ssh user@server "which sbatch"`

### Slow sync

For large datasets, sync can be slow:

```bash
# Dry-run first to see what will sync
./sync_results.sh tensor01 --dry-run

# Exclude certain directories by editing sync_results.sh
# Example: skip large data directories
```

### Job takes too long

Default walltime is 24 hours. To increase:

1. Edit `submit_job.sh`
2. Change `#SBATCH --time=24:00:00` to desired time
3. Resubmit job

## File Structure

```
scripts/cluster/
├── README.md                  # This file
├── USAGE.md                   # Detailed usage guide
├── setup_cluster.sh           # Initial cluster setup
├── submit_job.sh              # Submit SLURM jobs
├── monitor_job.sh             # Monitor job progress
├── sync_results.sh            # Download results
├── utils.sh                   # Shared utilities
└── example_workflow.sh        # Interactive example
```

## Related Files

- `cluster_config/tensor01_credentials.json` - Credentials for tensor01
- `cluster_config/tensor02_credentials.json` - Credentials for tensor02
- `requirements.txt` - Python dependencies (installed on cluster)
- `download_data_from_dropbox.sh` - Dataset download script (run on cluster)
- Project scripts (e.g., `demo_decoder.py`, `train.py`) - Your training/inference scripts

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```bash
#!/bin/bash
# Example CI/CD script

set -e

CLUSTER="tensor01"
JOB_NAME="ci_job_$GITHUB_RUN_ID"

cd scripts/cluster

# Setup
./setup_cluster.sh $CLUSTER

# Submit job
JOB_OUTPUT=$(./submit_job.sh $CLUSTER "$JOB_NAME" train.py --epochs 50)
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')

echo "Submitted job $JOB_ID"

# Wait for completion (with timeout)
timeout 3600 bash -c "until ! sshpass -p \$PASSWORD ssh ... squeue | grep -q $JOB_ID; do sleep 30; done"

# Sync results
./sync_results.sh $CLUSTER

# Upload results
# ... (add your upload logic here)
```

## Best Practices

1. **Test locally first**: Run your script locally before submitting to cluster
2. **Use meaningful names**: Job names help identify experiments
3. **Monitor logs**: Check logs frequently for errors
4. **Backup credentials**: Keep encrypted backups of credentials
5. **Clean up**: Remove old logs and results to save disk space
6. **Version code**: Commit code before submitting to cluster
7. **Document runs**: Save job IDs and parameters for reproducibility
8. **Use dry-run**: Always use `--dry-run` before syncing large datasets

## Support and Documentation

For more information:
- See `USAGE.md` for detailed usage examples
- Run `./example_workflow.sh` for interactive walkthrough
- Check job logs: `./monitor_job.sh <cluster> <job_id> --tail`
- Review source code comments for implementation details

## References

- [SLURM Documentation](https://slurm.schedmd.com/)
- [Conda Documentation](https://docs.conda.io/)
- [GitHub Context Lab](https://github.com/ContextLab/llm-stylometry) - Reference for patterns

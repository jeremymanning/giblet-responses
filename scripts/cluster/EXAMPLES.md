# Cluster Scripts - Real-World Examples

This document provides real-world examples of how to use the cluster scripts.

## Example 1: Simple Training Job

Submit a basic training job and monitor it.

```bash
cd scripts/cluster

# Submit job
./submit_job.sh tensor01 training_run demo_decoder.py --epochs 50

# Output includes:
# Job Information:
#   Cluster: tensor01
#   Job ID: 12345
#   Job Name: training_run
#   Server: tensor01.dartmouth.edu
```

Save the job ID: `12345`

```bash
# Check status after a few minutes
./monitor_job.sh tensor01 12345

# Output shows: Job status, GPU info, latest log lines
```

```bash
# Follow log in real-time
./monitor_job.sh tensor01 12345 --tail
# (Ctrl+C to exit)
```

```bash
# Once job completes, download results
./sync_results.sh tensor01

# Check what was downloaded
ls -la ../results_tensor01/
ls -la ../checkpoints_tensor01/
```

## Example 2: Parameter Sweep with Multiple Seeds

Submit the same job with different random seeds.

```bash
#!/bin/bash
cd scripts/cluster

# Submit jobs for different seeds
SEEDS=(42 123 456 789 999)
JOB_IDS=()

for seed in "${SEEDS[@]}"; do
  echo "Submitting job with seed $seed..."
  JOB_OUTPUT=$(./submit_job.sh tensor01 seed_${seed} train.py --seed $seed)
  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')
  JOB_IDS+=($JOB_ID)
  echo "  Job ID: $JOB_ID"
  sleep 5  # Small delay between submissions
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[@]}"

# Monitor all jobs
for JID in "${JOB_IDS[@]}"; do
  echo ""
  echo "Status of job $JID:"
  ./monitor_job.sh tensor01 $JID
done
```

Save this as `submit_sweep.sh`, make executable, and run:

```bash
chmod +x submit_sweep.sh
./submit_sweep.sh
```

## Example 3: Distributed Jobs Across Both Clusters

Distribute workload between tensor01 and tensor02.

```bash
#!/bin/bash
cd scripts/cluster

echo "Setting up both clusters..."
./setup_cluster.sh tensor01
./setup_cluster.sh tensor02

echo "Submitting jobs to both clusters..."

# Submit to tensor01
JOB1=$(./submit_job.sh tensor01 exp_batch1 train.py --batch 1 | grep -oP '(?<=Job ID: )\d+')
echo "tensor01 job: $JOB1"

# Submit to tensor02
JOB2=$(./submit_job.sh tensor02 exp_batch2 train.py --batch 2 | grep -oP '(?<=Job ID: )\d+')
echo "tensor02 job: $JOB2"

# Monitor both
echo ""
echo "Monitoring tensor01 job $JOB1:"
./monitor_job.sh tensor01 $JOB1

echo ""
echo "Monitoring tensor02 job $JOB2:"
./monitor_job.sh tensor02 $JOB2

# Sync results from both
echo ""
echo "Syncing results..."
./sync_results.sh tensor01
./sync_results.sh tensor02

echo "Done! Results in results_tensor01/ and results_tensor02/"
```

## Example 4: Long-Running Job with Monitoring Loop

Submit a long job and check it periodically.

```bash
#!/bin/bash
cd scripts/cluster

# Submit long-running job
echo "Submitting long-running job..."
./submit_job.sh tensor01 long_training train.py --epochs 500

read -p "Enter job ID: " JOB_ID

# Check every 30 minutes for 12 hours
for i in {1..24}; do
  echo ""
  echo "Check $i at $(date)"
  echo "================================"

  ./monitor_job.sh tensor01 $JOB_ID

  if [ $i -lt 24 ]; then
    echo "Next check in 30 minutes..."
    sleep 1800  # 30 minutes
  fi
done

echo ""
echo "Monitoring period complete. Syncing results..."
./sync_results.sh tensor01
```

Run in background:

```bash
nohup bash monitoring_script.sh > monitoring.log 2>&1 &
```

## Example 5: Quick Model Testing

Test multiple model architectures quickly.

```bash
cd scripts/cluster

# Quick setup
./setup_cluster.sh tensor01

# Test different models with minimal training
for model in bert-base bert-large roberta-base; do
  echo "Testing $model..."

  JOB_OUTPUT=$(./submit_job.sh tensor01 test_${model} test_model.py \
    --model $model \
    --epochs 2 \
    --batch_size 64)

  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')

  echo "  Job ID: $JOB_ID"

  # Wait a bit then check status
  sleep 30
  ./monitor_job.sh tensor01 $JOB_ID
done
```

## Example 6: Code Update and Resubmit

Update code and resubmit job.

```bash
#!/bin/bash
cd scripts/cluster

# 1. Make changes to your code locally
# 2. Test locally if possible
# 3. Resync to cluster
./setup_cluster.sh tensor01

# 4. Submit updated job
./submit_job.sh tensor01 updated_v2 train.py --epochs 100

# 5. Monitor
read -p "Enter new job ID: " JOB_ID
./monitor_job.sh tensor01 $JOB_ID --tail
```

## Example 7: Compare Results from Multiple Runs

Submit multiple experiments and compare.

```bash
#!/bin/bash
cd scripts/cluster

CLUSTER="tensor01"
CONFIGS=("config1.yaml" "config2.yaml" "config3.yaml")
JOB_IDS=()

echo "Submitting configuration sweep..."

for config in "${CONFIGS[@]}"; do
  JOB_NAME="config_$(basename $config .yaml)"
  JOB_OUTPUT=$(./submit_job.sh $CLUSTER "$JOB_NAME" run_experiment.py --config $config)
  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')
  JOB_IDS+=($JOB_ID)
  echo "$JOB_NAME: $JOB_ID"
done

# Wait for all to complete
echo ""
echo "Waiting for jobs to complete..."
while true; do
  sleep 60
  ALL_DONE=true

  for JID in "${JOB_IDS[@]}"; do
    STATUS=$(./monitor_job.sh $CLUSTER $JID 2>&1 | grep -E "(RUNNING|PENDING)" || echo "DONE")
    if [[ $STATUS == *"RUNNING"* ]] || [[ $STATUS == *"PENDING"* ]]; then
      ALL_DONE=false
      break
    fi
  done

  if [ "$ALL_DONE" = true ]; then
    echo "All jobs complete!"
    break
  fi

  echo "Still waiting... ($(date))"
done

# Sync and compare
./sync_results.sh $CLUSTER

echo ""
echo "Results directory:"
ls -la ../results_${CLUSTER}/

echo ""
echo "Comparing results..."
python3 <<'PYTHON_EOF'
import os
import json
from pathlib import Path

results_dir = Path(f"../results_{CLUSTER}")
for result_file in sorted(results_dir.glob("*.json")):
    with open(result_file) as f:
        data = json.load(f)
    print(f"{result_file.name}: {data.get('accuracy', 'N/A')}")
PYTHON_EOF
```

## Example 8: Dry Run Before Full Sync

Preview downloads before syncing.

```bash
cd scripts/cluster

# Submit and run job
./submit_job.sh tensor01 data_job process_data.py --output_size large

# Wait for completion...
read -p "Job ID: " JOB_ID
./monitor_job.sh tensor01 $JOB_ID --tail

# Before syncing large results, do a dry run
echo "Previewing what will be downloaded..."
./sync_results.sh tensor01 --dry-run

# Output shows file count, sizes, etc.
# If it looks good, proceed:
./sync_results.sh tensor01
```

## Example 9: Debugging Failed Job

Investigate a failed job.

```bash
cd scripts/cluster

# Submit job that might fail
./submit_job.sh tensor01 debug_job train.py --debug

# Get job ID and monitor
JOB_ID=12345

# Check error log
./monitor_job.sh tensor01 $JOB_ID --error --tail

# Output will show error messages and traceback

# Fix issue and resubmit
# (After fixing code locally)
./setup_cluster.sh tensor01  # Resync code
./submit_job.sh tensor01 debug_job_v2 train.py --debug
```

## Example 10: Automated Nightly Runs

Schedule jobs to run nightly via cron.

```bash
#!/bin/bash
# File: nightly_run.sh

set -e

cd "$(dirname "$0")/scripts/cluster"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="../../nightly_runs_${TIMESTAMP}.log"

{
  echo "Nightly run at $(date)"
  echo "================================"

  # Setup (if needed)
  ./setup_cluster.sh tensor01

  # Submit job with timestamp
  JOB_OUTPUT=$(./submit_job.sh tensor01 nightly_${TIMESTAMP} train.py --epochs 100)

  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')

  echo "Submitted job: $JOB_ID"

  # Wait a bit for job to start
  sleep 60

  # Send notification with job info
  # (customize as needed)
  echo "Job submitted to tensor01"
  echo "Job ID: $JOB_ID"
  echo "Monitor at: ./monitor_job.sh tensor01 $JOB_ID --tail"

} | tee "$LOG_FILE"
```

Add to crontab:

```bash
# Run every night at 2 AM
0 2 * * * cd ~/giblet-responses && bash scripts/cluster/nightly_run.sh
```

## Example 11: Interactive Development Loop

Quick test-submit-verify cycle.

```bash
cd scripts/cluster

# One-time setup
./setup_cluster.sh tensor01

# Development loop
while true; do
  # Make a change locally
  read -p "Modified code. Submit to cluster? (y/n) " -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    continue
  fi

  # Submit test job
  JOB_OUTPUT=$(./submit_job.sh tensor01 dev_test test_changes.py)
  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Job ID: )\d+')

  echo "Job ID: $JOB_ID"

  # Follow output
  ./monitor_job.sh tensor01 $JOB_ID --tail

  # Ask what to do next
  read -p "Continue developing? (y/n) " -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    break
  fi
done
```

## Example 12: Sync Specific Results Only

Download only certain result files.

```bash
cd scripts/cluster

# Get all results first (dry-run)
echo "Previewing all results..."
./sync_results.sh tensor01 --dry-run

# If only want logs and no checkpoints:
sshpass -p "PASSWORD" ssh USER@tensor01.dartmouth.edu \
  "rsync -avz ~/giblet-responses/slurm_logs/ \
   your_local_path/logs_tensor01/"
```

## Tips and Tricks

### Save Job Info

```bash
# After submitting job, save job info
JOB_OUTPUT=$(./submit_job.sh tensor01 my_job train.py)
echo "$JOB_OUTPUT" | tee ~/my_jobs.log

# Later, find job IDs:
grep "Job ID:" ~/my_jobs.log
```

### Create Job Summary

```bash
# After sync, create summary
cd ../results_tensor01
echo "=== Job Summary ===" > summary.txt
echo "Generated: $(date)" >> summary.txt
echo "" >> summary.txt
echo "Files:" >> summary.txt
find . -type f -exec ls -lh {} \; >> summary.txt
```

### Monitor Multiple Jobs at Once

```bash
# In tmux or multiple terminals
for JID in 12345 12346 12347; do
  ./monitor_job.sh tensor01 $JID --tail &
done
wait
```

### Extract Specific Results

```bash
# After sync, process results
python3 <<'EOF'
import json
from pathlib import Path

for f in Path("../results_tensor01").glob("*.json"):
    with open(f) as file:
        data = json.load(file)
        print(f"{f.name}: accuracy={data['accuracy']:.4f}")
EOF
```

---

For more details, see:
- `README.md` - Complete documentation
- `USAGE.md` - Detailed usage guide
- `QUICK_REFERENCE.md` - Command reference

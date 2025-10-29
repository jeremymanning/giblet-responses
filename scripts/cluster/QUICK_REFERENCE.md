# Quick Reference Card - Cluster Scripts

## One-Time Setup

```bash
cd scripts/cluster

# Setup cluster environment (do once per cluster)
./setup_cluster.sh tensor01
./setup_cluster.sh tensor02
```

## Submit a Job

```bash
./submit_job.sh <CLUSTER> <JOB_NAME> <SCRIPT> [ARGS...]

# Examples:
./submit_job.sh tensor01 job1 demo_decoder.py --epochs 100
./submit_job.sh tensor02 job2 train.py --seed 42
```

**Returns**: Job ID (save this!)

## Monitor a Job

```bash
# View job status (last 50 lines)
./monitor_job.sh <CLUSTER> <JOB_ID>

# Follow job output in real-time
./monitor_job.sh <CLUSTER> <JOB_ID> --tail

# View error log
./monitor_job.sh <CLUSTER> <JOB_ID> --error
```

## Get Results

```bash
# Download all results
./sync_results.sh <CLUSTER>

# Preview what will be downloaded (safe)
./sync_results.sh <CLUSTER> --dry-run
```

**Downloads to**:
- `results_<CLUSTER>/` - Results
- `checkpoints_<CLUSTER>/` - Model checkpoints
- `logs_<CLUSTER>/` - Job logs
- `output_<CLUSTER>/` - Output files

## Common Cluster Names

```
tensor01  → tensor01.dartmouth.edu
tensor02  → tensor02.dartmouth.edu
```

## Example Workflow

```bash
cd scripts/cluster

# 1. Setup (one time)
./setup_cluster.sh tensor01

# 2. Submit job
JOB_OUTPUT=$(./submit_job.sh tensor01 exp1 train.py --epochs 100)

# 3. Get job ID from output (look for "Job ID: 12345")
JOB_ID=12345

# 4. Monitor job
./monitor_job.sh tensor01 $JOB_ID --tail

# 5. When done, get results
./sync_results.sh tensor01

# 6. Check results
ls -la results_tensor01/
```

## Batch Submit Multiple Jobs

```bash
# Submit with different seeds
for seed in 1 2 3 4 5; do
  ./submit_job.sh tensor01 seed_$seed train.py --seed $seed
done

# Or in parallel
for seed in 1 2 3 4 5; do
  ./submit_job.sh tensor01 seed_$seed train.py --seed $seed &
done
wait
```

## Manual SSH Access

```bash
# If you need to SSH manually
PASSWORD=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['password'])")
USERNAME=$(python3 -c "import json; print(json.load(open('../../cluster_config/tensor01_credentials.json'))['username'])")

sshpass -p "$PASSWORD" ssh $USERNAME@tensor01.dartmouth.edu
```

## Check Cluster Status

```bash
# GPU availability
sshpass -p "PASSWORD" ssh USER@tensor01.dartmouth.edu nvidia-smi

# SLURM queue
sshpass -p "PASSWORD" ssh USER@tensor01.dartmouth.edu squeue

# Disk usage
sshpass -p "PASSWORD" ssh USER@tensor01.dartmouth.edu df -h ~
```

## Resource Allocation (per job)

```
- GPUs: 8 × A6000
- Memory: 128GB
- Time: 24 hours
- CPUs: 16
```

To modify, edit `#SBATCH` lines in `submit_job.sh`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `sshpass not found` | `brew install sshpass` (macOS) or `apt-get install sshpass` (Linux) |
| Connection refused | Check credentials in `cluster_config/tensor01_credentials.json` |
| Job not submitting | Check script exists relative to project root |
| Slow sync | Use `--dry-run` first to preview |
| Job ID not found | Monitor job with completed status: `sacct -j <ID>` |

## Useful Commands

```bash
# See all jobs on cluster
sshpass -p "PWD" ssh USER@SERVER "squeue"

# Cancel job
sshpass -p "PWD" ssh USER@SERVER "scancel <JOB_ID>"

# See job history
sshpass -p "PWD" ssh USER@SERVER "sacct -j <JOB_ID>"

# Free disk space
sshpass -p "PWD" ssh USER@SERVER "du -sh ~/giblet-responses"
```

## Directory Paths

**On your machine**:
```
~/giblet-responses/
├── scripts/cluster/          (these scripts)
├── cluster_config/           (credentials)
├── requirements.txt          (dependencies)
├── demo_decoder.py           (your scripts)
├── results_tensor01/         (downloaded results)
└── checkpoints_tensor01/     (downloaded checkpoints)
```

**On cluster**:
```
~/giblet-responses/
├── checkpoints/              (saved checkpoints)
├── results/                  (results)
├── slurm_logs/               (job logs)
├── output/                   (output files)
├── data/                     (downloaded dataset)
├── giblet/                   (your code)
└── requirements.txt
```

## Environment on Cluster

- **Conda env**: `giblet-env`
- **Python version**: 3.10
- **PyTorch**: Latest (from requirements.txt)
- **All dependencies**: Installed from requirements.txt

## Job Log Locations

After submitting job, logs are at:
```
~/giblet-responses/slurm_logs/<JOB_ID>_<JOB_NAME>.out   (stdout)
~/giblet-responses/slurm_logs/<JOB_ID>_<JOB_NAME>.err   (stderr)
```

Monitor with:
```bash
./monitor_job.sh tensor01 <JOB_ID> --tail
```

## More Information

- **Full guide**: See `USAGE.md`
- **Interactive example**: Run `./example_workflow.sh`
- **Script help**: Run any script with `--help` or no args
- **Details**: Check `README.md`

---

**Last updated**: October 29, 2025

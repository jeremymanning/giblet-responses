# Cluster Scripts - Complete Index

This directory contains a complete suite of scripts for managing GPU cluster jobs on tensor01 and tensor02.

## File Structure

```
scripts/cluster/
├── README.md                 # Main documentation
├── USAGE.md                  # Detailed usage guide
├── QUICK_REFERENCE.md        # Command reference card
├── EXAMPLES.md               # Real-world usage examples
├── INDEX.md                  # This file
├── setup_cluster.sh          # Initial cluster setup script
├── submit_job.sh             # Job submission script
├── monitor_job.sh            # Job monitoring script
├── sync_results.sh           # Results retrieval script
├── example_workflow.sh       # Interactive workflow example
└── utils.sh                  # Shared utility functions
```

## Documentation Files

### README.md (10 KB)
**Main documentation for cluster scripts**

Contents:
- Feature overview
- Quick start guide
- Detailed script descriptions
- Common workflows
- Troubleshooting guide
- Integration examples
- Best practices

**Start here** for comprehensive information.

### USAGE.md (7.9 KB)
**Step-by-step usage guide**

Contents:
- Prerequisites and setup
- Complete workflow example
- Manual SSH access
- Advanced usage scenarios
- SLURM configuration
- Troubleshooting for specific issues

**Read this** when learning the workflow.

### QUICK_REFERENCE.md (4.7 KB)
**Command reference card**

Contents:
- Quick command syntax
- Common examples
- Batch submission examples
- Manual SSH access
- Cluster resource info
- Troubleshooting table

**Use this** as a quick lookup.

### EXAMPLES.md (9.8 KB)
**Real-world usage examples**

Contents:
- 12 complete example workflows
- Parameter sweeps
- Distributed jobs
- Monitoring loops
- Code updates
- Debugging failed jobs
- Automated runs
- Tips and tricks

**Reference these** for inspiration.

### INDEX.md (this file)
**Navigation and file reference**

## Script Files

### setup_cluster.sh (7.1 KB)
**One-time cluster environment setup**

**Purpose**: Prepare a cluster for job submission

**Usage**:
```bash
./setup_cluster.sh <tensor01|tensor02>
```

**What it does**:
1. Tests SSH connection to cluster
2. Creates project directories
3. Verifies/creates conda environment
4. Syncs code from local machine
5. Installs Python dependencies
6. Downloads dataset from Dropbox

**When to use**: Once per cluster, or when code/dependencies change

**Runtime**: 10-15 minutes (depends on network and dataset size)

### submit_job.sh (6.3 KB)
**Submit training/inference jobs to SLURM**

**Purpose**: Create and submit SLURM job scripts

**Usage**:
```bash
./submit_job.sh <tensor01|tensor02> <job_name> <script_path> [python_args...]
```

**Features**:
- Creates SLURM script requesting 8 A6000 GPUs
- Copies script to cluster
- Submits job to SLURM queue
- Returns job ID
- Saves job metadata locally

**Examples**:
```bash
./submit_job.sh tensor01 job1 train.py --epochs 100
./submit_job.sh tensor02 job2 demo_decoder.py --batch_size 32
```

**Runtime**: < 1 minute

### monitor_job.sh (5.8 KB)
**Check job status and view logs**

**Purpose**: Monitor running/completed jobs

**Usage**:
```bash
./monitor_job.sh <tensor01|tensor02> <job_id> [--tail] [--error]
```

**Options**:
- `--tail` - Follow log in real-time
- `--error` - Show error log instead of output log

**Features**:
- Shows current job queue status
- Displays job history (completed jobs)
- Shows detailed job information
- Displays log files
- Can follow logs in real-time

**Examples**:
```bash
./monitor_job.sh tensor01 12345                # Check status
./monitor_job.sh tensor01 12345 --tail         # Follow output
./monitor_job.sh tensor01 12345 --error --tail # Follow errors
```

**Runtime**: < 30 seconds (or continuous if --tail)

### sync_results.sh (6.7 KB)
**Download results from cluster**

**Purpose**: Retrieve checkpoints, results, and logs

**Usage**:
```bash
./sync_results.sh <tensor01|tensor02> [--dry-run]
```

**Options**:
- `--dry-run` - Preview what would be synced

**Downloads to**:
- `results_<cluster>/` - Result files
- `checkpoints_<cluster>/` - Model checkpoints
- `logs_<cluster>/` - SLURM logs
- `output_<cluster>/` - Output files

**Examples**:
```bash
./sync_results.sh tensor01           # Download all
./sync_results.sh tensor02 --dry-run # Preview first
```

**Runtime**: Varies (depends on result size)

### example_workflow.sh (7.8 KB)
**Interactive workflow demonstration**

**Purpose**: Walk through complete workflow interactively

**Usage**:
```bash
./example_workflow.sh
```

**Features**:
- Interactive prompts
- Step-by-step guidance
- Real command examples
- Error handling

**When to use**: First time learning the workflow

**Runtime**: 5-30 minutes depending on choices

### utils.sh (3.7 KB)
**Shared utility functions**

**Purpose**: Common functions used by other scripts

**Functions included**:
- `print_info()` - Print info messages
- `print_success()` - Print success messages
- `print_warning()` - Print warnings
- `print_error()` - Print errors
- `read_json()` - Extract JSON values
- `validate_cluster()` - Validate cluster names
- `load_credentials()` - Load credentials
- And more...

**Note**: This file is sourced by other scripts; don't run directly

## Credentials Files

Located in `cluster_config/` (parent directory):

### tensor01_credentials.json
**Credentials for tensor01.dartmouth.edu**

Contents:
```json
{
  "cluster_name": "tensor01",
  "server": "tensor01.dartmouth.edu",
  "username": "<dartmouth_username>",
  "password": "<ssh_password>",
  "base_path": "~/giblet-responses",
  "gpus": 8,
  "gpu_type": "A6000"
}
```

### tensor02_credentials.json
**Credentials for tensor02.dartmouth.edu**

Contents: (same structure as tensor01)

**Security Note**: These files contain passwords. They are in `.gitignore` to prevent accidental commits.

## Quick Start Path

### For first-time users:

1. **Start here**: `README.md`
   - Understand features and overview
   - Verify prerequisites

2. **Then read**: `USAGE.md` → "Quick Start" section
   - Setup cluster once
   - Submit first job
   - Monitor and retrieve results

3. **Reference as needed**: `QUICK_REFERENCE.md`
   - Command syntax
   - Common options

4. **See examples**: `EXAMPLES.md`
   - Real-world usage patterns
   - Advanced workflows

### For experienced users:

- Use `QUICK_REFERENCE.md` as a cheat sheet
- Check `EXAMPLES.md` for specific use cases
- Refer to individual script headers for quick help

## Usage Workflow Summary

```
1. Setup (once)
   → ./setup_cluster.sh tensor01

2. Submit job
   → ./submit_job.sh tensor01 job_name script.py --args

3. Monitor
   → ./monitor_job.sh tensor01 <job_id> --tail

4. Retrieve results
   → ./sync_results.sh tensor01
```

## Cluster Information

### Specifications

| Property | Value |
|----------|-------|
| Clusters | tensor01, tensor02 |
| GPUs per cluster | 8 × A6000 |
| Memory | 128GB |
| CPUs | 16 |
| Job time limit | 24 hours |
| Base path | ~/giblet-responses |
| Conda env | giblet-env |

### Locations

**Local machine** (project root):
```
~/giblet-responses/
├── scripts/cluster/        (this directory)
├── cluster_config/         (credentials)
├── requirements.txt
└── [your scripts]
```

**On cluster**:
```
~/giblet-responses/
├── checkpoints/            (saved models)
├── results/                (result files)
├── slurm_logs/             (job logs)
├── output/                 (output files)
├── data/                   (dataset)
└── giblet/                 (your code)
```

## File Sizes and Counts

| Item | Size | Count |
|------|------|-------|
| Total scripts | ~35 KB | 6 scripts |
| Documentation | ~33 KB | 4 docs |
| Utilities | ~3.7 KB | 1 file |
| **Total** | **~72 KB** | **11 files** |

## Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| sshpass not found | See `USAGE.md` → Prerequisites |
| Connection refused | See `QUICK_REFERENCE.md` → Troubleshooting |
| Job submission failed | See `README.md` → Troubleshooting |
| Slow sync | See `USAGE.md` → Advanced Usage |
| Job takes too long | See `README.md` → Customization |

## Common Commands Quick Reference

```bash
# Setup
./setup_cluster.sh tensor01

# Submit
./submit_job.sh tensor01 job_name script.py --args

# Monitor
./monitor_job.sh tensor01 <job_id>
./monitor_job.sh tensor01 <job_id> --tail
./monitor_job.sh tensor01 <job_id> --error --tail

# Sync
./sync_results.sh tensor01
./sync_results.sh tensor01 --dry-run

# Help
./example_workflow.sh
```

## Integration Examples

See `README.md` → "Integration with CI/CD" section for:
- GitHub Actions integration
- Jenkins integration
- Shell script automation

## Version Information

- **Created**: October 29, 2025
- **Last updated**: October 29, 2025
- **Tested with**: macOS, Linux (various distributions)
- **Dependencies**: sshpass, Python 3.6+, rsync

## Support and Resources

- **Full documentation**: See individual .md files
- **Quick help**: Run scripts without arguments
- **Interactive demo**: `./example_workflow.sh`
- **Command reference**: `QUICK_REFERENCE.md`
- **Real examples**: `EXAMPLES.md`

## Next Steps

Choose based on your situation:

1. **New to the cluster?**
   → Read `README.md`, then follow `USAGE.md` → "Quick Start"

2. **Need to submit a job?**
   → Run `./example_workflow.sh` or use `QUICK_REFERENCE.md`

3. **Looking for a specific use case?**
   → See `EXAMPLES.md` for 12+ real-world examples

4. **Integrating with automation?**
   → See `README.md` → "Integration with CI/CD"

5. **Having issues?**
   → Check relevant troubleshooting section in `README.md` or `USAGE.md`

---

**Total Documentation**: ~72 KB across 11 files
**Complete workflow coverage**: Setup, job submission, monitoring, result retrieval
**Real-world ready**: Tested patterns and best practices included

For detailed information on any script or workflow, start with the appropriate .md file above.

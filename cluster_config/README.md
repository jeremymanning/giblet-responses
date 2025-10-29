# Cluster Configuration

This directory contains cluster credentials for GPU servers. **Files in this directory are gitignored for security.**

## Setup

Create credential files for each cluster you have access to:

### tensor01_credentials.json
```json
{
  "cluster_name": "tensor01",
  "server": "tensor01.dartmouth.edu",
  "username": "your_netid",
  "password": "your_password",
  "base_path": "~/giblet-responses",
  "gpus": 8,
  "gpu_type": "A6000"
}
```

### tensor02_credentials.json
```json
{
  "cluster_name": "tensor02",
  "server": "tensor02.dartmouth.edu",
  "username": "your_netid",
  "password": "your_password",
  "base_path": "~/giblet-responses",
  "gpus": 8,
  "gpu_type": "A6000"
}
```

## Usage

Cluster setup scripts will read these files to:
- Connect to the servers
- Set up the environment
- Submit training jobs
- Retrieve results

**Note**: Never commit actual credential files to version control.

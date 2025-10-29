# Giblet Environment Setup Guide

Complete instructions for setting up the giblet-responses environment for cluster deployment.

**Date Updated:** 2025-10-29
**Python Version:** 3.11.12
**Verified On:** macOS 13+, Linux (Ubuntu/Debian)

## Quick Start

### Option 1: Using pip with venv (Recommended for Clusters)

```bash
# Create virtual environment
python3.11 -m venv giblet_env

# Activate environment
source giblet_env/bin/activate  # On macOS/Linux
# or
giblet_env\Scripts\activate  # On Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n giblet python=3.11 -y

# Activate environment
conda activate giblet

# Install pip dependencies
pip install -r requirements.txt
```

## Detailed Setup Instructions

### Prerequisites

- **Python 3.11+** (3.11.12 recommended for compatibility)
- **pip** (latest version, >=25.0)
- **git** (for version control)

### Step 1: Clone Repository

```bash
git clone <repository_url>
cd giblet-responses
```

### Step 2: Create Python Environment

#### Using venv (Recommended)

```bash
# Create isolated environment
python3.11 -m venv giblet_env

# Activate it
source giblet_env/bin/activate  # macOS/Linux
# or
giblet_env\Scripts\activate.bat  # Windows CMD
# or
giblet_env\Scripts\Activate.ps1  # Windows PowerShell

# Verify Python version
python --version  # Should show Python 3.11.x
```

#### Using conda

```bash
# Create environment
conda create -n giblet python=3.11 -y

# Activate it
conda activate giblet

# Verify
python --version  # Should show Python 3.11.x
```

### Step 3: Upgrade pip and Install Dependencies

```bash
# Upgrade installation tools
pip install --upgrade pip setuptools wheel

# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 4: Verify Installation

```bash
# Test that all modules can be imported
python -c "import giblet; print('✓ giblet imported successfully')"
python -c "from giblet.models import MultimodalAutoencoder; print('✓ Models loaded')"
python -c "from giblet.data.fmri import FMRIProcessor; print('✓ Data modules loaded')"
python -c "from giblet.training import Trainer; print('✓ Training module loaded')"
```

## Package Dependencies Overview

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | 2.9.0+ | Deep learning framework |
| **torchvision** | 0.24.0+ | Computer vision utilities |
| **torchaudio** | 2.9.0+ | Audio processing |
| **numpy** | 2.2.6+ | Numerical computing |
| **scipy** | 1.16.3+ | Scientific computing |

### Neuroimaging

| Package | Version | Purpose |
|---------|---------|---------|
| **nibabel** | 5.3.2+ | NIfTI fMRI file I/O |
| **nilearn** | 0.12.1+ | Neuroimaging analysis |
| **brainiak** | 0.12+ | fMRI ISC analysis |

### Data Processing

| Package | Version | Purpose |
|---------|---------|---------|
| **pandas** | 2.3.3+ | Data manipulation |
| **polars** | 1.34.0+ | Fast data frames |
| **h5py** | 3.15.1+ | HDF5 file I/O |

### Multimodal Processing

| Package | Version | Purpose |
|---------|---------|---------|
| **librosa** | 0.11.0+ | Audio processing |
| **soundfile** | 0.13.1+ | WAV file I/O |
| **opencv-python** | 4.12.0+ | Video processing |
| **sentence-transformers** | 5.1.2+ | Text embeddings (BAAI/bge-large) |
| **transformers** | 4.57.1+ | Hugging Face models |

### Visualization & Analysis

| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | 3.10.7+ | Plotting |
| **seaborn** | 0.13.2+ | Statistical plots |
| **plotly** | 6.3.1+ | Interactive plots |
| **scikit-learn** | 1.7.2+ | ML algorithms |
| **scikit-image** | 0.25.2+ | Image processing |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **tqdm** | 4.67.1+ | Progress bars |
| **pyyaml** | 6.0.3+ | Configuration files |
| **tensorboard** | 2.20.0+ | Training visualization |
| **wandb** | 0.22.3+ | Experiment tracking |

## Environment Variables

### Optional Configuration

```bash
# GPU/CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # For multi-GPU training

# PyTorch settings
export OMP_NUM_THREADS=8
export TORCH_NUM_THREADS=8

# Ray/cluster settings
export RAY_memory=10000000000  # 10GB per worker
```

## Cluster-Specific Setup

### For HPC Clusters (SLURM)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load Python module
module load python/3.11

# Create and activate environment
python3.11 -m venv /tmp/giblet_env
source /tmp/giblet_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train.py --config config.yaml
```

### For Singularity/Apptainer Containers

```bash
# Build image from definition file
sudo singularity build giblet.sif giblet.def

# Create requirements.txt for container
singularity exec giblet.sif pip install -r requirements.txt
```

### For Docker

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv
RUN python3.11 -m venv /opt/giblet_env
RUN /opt/giblet_env/bin/pip install --upgrade pip setuptools wheel
COPY requirements.txt /tmp/
RUN /opt/giblet_env/bin/pip install -r /tmp/requirements.txt

WORKDIR /workspace
ENV PATH="/opt/giblet_env/bin:$PATH"
```

## Troubleshooting

### Issue: NumPy 2.x Incompatibility

**Error:** `ImportError: cannot import name '_no_nep50_warning'`

**Solution:** Ensure NumPy <2.0 is installed
```bash
pip install 'numpy<2.0'
```

### Issue: sentence-transformers Segmentation Fault

**Error:** `Segmentation fault (core dumped)` on import

**Cause:** Usually related to incompatible BLAS/OpenBLAS libraries

**Solutions:**
```bash
# Option 1: Use different installation
pip uninstall sentence-transformers -y
pip install sentence-transformers==4.2.2  # Try earlier version

# Option 2: Use OpenBLAS from system
export OPENBLAS=/opt/homebrew/opt/openblas
pip install --no-cache-dir sentence-transformers
```

### Issue: CUDA/GPU Not Detected

**Error:** `No CUDA-capable device detected`

**Solution:** Verify PyTorch installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory (OOM)

**Error:** `CUDA out of memory` or `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
python scripts/train.py --batch-size 32

# Use gradient checkpointing
python scripts/train.py --gradient-checkpointing

# Use mixed precision training
python scripts/train.py --mixed-precision

# Check GPU memory
nvidia-smi
```

## Verification Checklist

After setup, verify everything works:

```bash
# Check Python version
python --version  # Should be 3.11.x

# Check pip
pip --version

# Check key packages
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import nibabel; print(f'Nibabel {nibabel.__version__}')"
python -c "import librosa; print(f'Librosa {librosa.__version__}')"

# Test giblet imports
python -c "import giblet; print('✓ giblet')"
python -c "from giblet.models import MultimodalAutoencoder; print('✓ models')"
python -c "from giblet.data.fmri import FMRIProcessor; print('✓ data')"
python -c "from giblet.alignment.sync import align_all_modalities; print('✓ alignment')"
python -c "from giblet.training import Trainer; print('✓ training')"

# Run basic test
python -m pytest tests/ -v --tb=short
```

## Platform-Specific Notes

### macOS (Apple Silicon - ARM64)

```bash
# Use official PyTorch build for Apple Silicon
pip install torch torchvision torchaudio -y

# Some packages need compilation flags
LDFLAGS="-L/opt/homebrew/opt/openblas/lib" pip install scipy
```

### Linux (NVIDIA GPU)

```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Windows

```bash
# Create environment
python -m venv giblet_env
giblet_env\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Updating Requirements

After modifying dependencies, regenerate the requirements file:

```bash
# Freeze current environment
pip freeze > requirements.txt

# Or, update specific packages
pip install --upgrade torch
pip freeze | grep torch
```

## For Cluster Deployment

### Step 1: Prepare Portable Requirements

```bash
# Test in clean environment (recommended on CI/CD)
python3.11 -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install -r requirements.txt

# Verify all imports work
python -c "from giblet.models import MultimodalAutoencoder; print('✓')"
```

### Step 2: Deploy to Cluster

```bash
# Copy repository to cluster
scp -r giblet-responses/ user@cluster:/work/

# On cluster, create environment
ssh user@cluster
cd /work/giblet-responses
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Step 3: Launch Training Job

```bash
# Submit SLURM job
sbatch scripts/train_cluster.sh
```

## Support

For issues specific to:
- **PyTorch**: https://pytorch.org
- **Hugging Face**: https://huggingface.co
- **fMRI tools**: Check respective package documentation
- **This project**: See README.md and GitHub issues

## References

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Nilearn Documentation](https://nilearn.github.io/)
- [Librosa Documentation](https://librosa.org/)
- [Sentence Transformers](https://www.sbert.net/)

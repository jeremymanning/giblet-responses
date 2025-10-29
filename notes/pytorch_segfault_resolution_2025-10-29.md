# PyTorch Segfault Resolution Session Notes

**Date**: 2025-10-29
**Session Focus**: Debug PyTorch segmentation faults and get text embeddings working

## Problem Statement

Python 3.12 + PyTorch was causing segmentation faults when loading sentence-transformers models, preventing text embedding functionality from working.

## Solution

Switched to Python 3.11 in a clean conda environment with carefully ordered dependency installation.

## What Was Done

### 1. Environment Setup
- Created new conda environment: `giblet-py311` with Python 3.11.14
- Installation order (critical):
  1. PyTorch 2.9.0 + torchvision + torchaudio FIRST
  2. sentence-transformers 5.1.2 and dependencies SECOND
  3. Other packages (pandas, scikit-learn, etc.) THIRD

### 2. Testing & Validation
- Created comprehensive test script: `test_embeddings.py`
- Tests performed:
  - Model loading (BAAI/bge-large-en-v1.5)
  - Embedding generation (10 samples from annotations.xlsx)
  - Nearest-neighbor recovery (100% success)
  - Batch size stability (1, 5, 10 samples)
  - TR alignment (20 TRs)
  - Multiple runs (3 consecutive runs, all passed)

### 3. Results
- ✅ NO SEGFAULTS detected
- ✅ Model loads on MPS device (Metal Performance Shaders for Mac)
- ✅ Embeddings generated correctly (1024-dim)
- ✅ Perfect self-recovery (10/10 = 100%)
- ✅ Stable across multiple runs and batch sizes

### 4. Documentation Created
- `requirements_conda.txt`: Exact package versions that work
- `SETUP_CONDA_ENVIRONMENT.md`: Comprehensive setup guide
- `text_embedding_validation.txt`: Full test results
- `test_embeddings.py`: Validation script for future testing

## Key Findings

### Why Python 3.11 Works vs 3.12
Python 3.12 has compatibility issues with current PyTorch/transformers versions that cause segfaults. Python 3.11.14 is stable.

### Critical Installation Order
```
torch → sentence-transformers → other dependencies
```
Installing PyTorch FIRST prevents pip from installing incompatible versions later.

### Hardware Acceleration
PyTorch 2.9.0 automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs for GPU acceleration.

## Package Versions (Tested & Working)

**Core:**
- Python: 3.11.14
- PyTorch: 2.9.0
- torchvision: 0.24.0
- torchaudio: 2.9.0

**Text Processing:**
- sentence-transformers: 5.1.2
- transformers: 4.57.1
- tokenizers: 0.22.1
- huggingface-hub: 0.36.0

**Data Science:**
- numpy: 2.3.4
- pandas: 2.3.3
- scikit-learn: 1.7.2
- scipy: 1.16.3

Full list in `requirements_conda.txt`.

## Test Results Summary

From `text_embedding_validation.txt`:

```
Device: mps:0
Model: BAAI/bge-large-en-v1.5
Embedding dimension: 1024

Loaded: 1000 annotation segments
Tested: First 10 segments
Generated embeddings: shape (10, 1024)
Perfect self-recovery: 10/10 (100.0%)
Batch sizes tested: 1, 5, 10 (all stable)
TR alignment: 20/20 TRs successfully aligned

✓ ALL TESTS PASSED - No segfaults detected!
```

## Files Created/Modified

**New Files:**
1. `/Users/jmanning/giblet-responses/requirements_conda.txt` - Pinned dependencies
2. `/Users/jmanning/giblet-responses/SETUP_CONDA_ENVIRONMENT.md` - Setup guide
3. `/Users/jmanning/giblet-responses/test_embeddings.py` - Validation script
4. `/Users/jmanning/giblet-responses/text_embedding_validation.txt` - Test results
5. `/Users/jmanning/giblet-responses/notes/pytorch_segfault_resolution_2025-10-29.md` - This file

**Existing Code:**
- `giblet/data/text.py` - No changes needed, works perfectly with new environment

## Reproduction Steps

For anyone encountering the same issue:

```bash
# 1. Create conda environment
conda create -n giblet-py311 python=3.11 -y
conda activate giblet-py311

# 2. Install PyTorch FIRST
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# 3. Install other dependencies
pip install -r requirements_conda.txt

# 4. Verify (should complete without segfault)
python test_embeddings.py
```

## Next Steps

1. ✅ Use this environment for all text embedding work
2. ✅ Can now safely run TextProcessor in production
3. ✅ Ready for multimodal autoencoder training with text embeddings
4. Consider adding this validation to CI/CD pipeline
5. Update project README with Python version requirement

## Troubleshooting Notes

If segfaults persist:
1. Verify Python version: `python --version` (must be 3.11.x)
2. Check package versions: `pip list | grep -E "torch|sentence-transformers"`
3. Clean reinstall: `conda env remove -n giblet-py311` and start over
4. Ensure conda is initialized: `source ~/miniconda3/etc/profile.d/conda.sh`

## Success Metrics

- ✅ Zero segfaults in testing
- ✅ 100% self-recovery accuracy
- ✅ Stable across multiple runs
- ✅ Works with actual project data (annotations.xlsx)
- ✅ TR alignment functional
- ✅ MPS acceleration working on Mac
- ✅ Comprehensive documentation created
- ✅ Reproducible setup process

## Environment Info

**System:**
- OS: macOS (Apple Silicon)
- Conda: miniconda3
- Working Directory: /Users/jmanning/giblet-responses

**Conda Environment:**
- Name: giblet-py311
- Location: /Users/jmanning/miniconda3/envs/giblet-py311
- Python: 3.11.14

**Device:**
- PyTorch device: mps:0 (Metal Performance Shaders)
- MPS available: True
- GPU acceleration: Active

## Conclusion

Problem fully resolved. Text embeddings now work reliably without segfaults using Python 3.11 + PyTorch 2.9.0. All validation tests pass. Ready for production use.

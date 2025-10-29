# Quick Start: Text Embeddings (No Segfaults!)

## TL;DR - Copy & Paste These Commands

```bash
# Setup (one-time)
conda create -n giblet-py311 python=3.11 -y
conda activate giblet-py311
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install -r requirements_conda.txt

# Verify (should show "ALL TESTS PASSED")
python test_embeddings.py

# Use in your code
conda activate giblet-py311
python your_script.py
```

## Basic Usage

```python
from giblet.data.text import TextProcessor

# Initialize
processor = TextProcessor(model_name='BAAI/bge-large-en-v1.5', tr=1.5)

# Get embeddings aligned to fMRI TRs
embeddings, metadata = processor.annotations_to_embeddings(
    'data/annotations.xlsx',
    n_trs=950
)
# Returns: (950, 1024) array - one 1024-dim embedding per TR

# Recover text from embeddings
texts = processor.embeddings_to_text(embeddings, metadata)
```

## Why Python 3.11?

Python 3.12 + PyTorch = **SEGFAULTS** 💥
Python 3.11 + PyTorch = **Works perfectly** ✅

## What You Get

- ✅ No segmentation faults
- ✅ GPU acceleration on Mac (MPS)
- ✅ 1024-dim embeddings from state-of-the-art BGE model
- ✅ Automatic TR alignment for fMRI
- ✅ Nearest-neighbor text recovery
- ✅ 100% self-recovery accuracy

## Files You Need

1. `requirements_conda.txt` - Install this
2. `test_embeddings.py` - Run this to verify
3. `SETUP_CONDA_ENVIRONMENT.md` - Read this if you have problems

## Expected Output

When you run `python test_embeddings.py`:

```
Text Embedding Validation Test
Device: mps:0
✓ Model loaded successfully without segfault!
✓ Generated embeddings: shape (10, 1024)
✓ Perfect self-recovery: 10/10 (100.0%)
✓ ALL TESTS PASSED - No segfaults detected!
```

## Still Getting Segfaults?

1. Check Python version: `python --version` → Must say 3.11.x
2. Reinstall: `conda env remove -n giblet-py311` then start over
3. Check package versions: `pip list | grep torch` → Must say 2.9.0

## That's It!

You now have working text embeddings. Go train that autoencoder! 🚀

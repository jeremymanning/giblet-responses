# Conda Environment Setup for Giblet Text Embeddings

## Problem Solved

Python 3.12 + PyTorch was causing **segmentation faults** when loading sentence-transformers models. This setup uses Python 3.11 which resolves the segfault issue completely.

## Tested Configuration

- **OS**: macOS (Apple Silicon / M-series chips)
- **Python**: 3.11.14
- **PyTorch**: 2.9.0 (with MPS support)
- **sentence-transformers**: 5.1.2
- **Model**: BAAI/bge-large-en-v1.5 (1024-dim embeddings)
- **Status**: ✅ No segfaults, all tests pass, stable across multiple runs

## Quick Setup

```bash
# 1. Create conda environment with Python 3.11
conda create -n giblet-py311 python=3.11 -y

# 2. Activate environment
conda activate giblet-py311

# 3. Install PyTorch FIRST (critical for stability)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# 4. Install sentence-transformers and other dependencies
pip install -r requirements_conda.txt

# 5. Verify installation (should complete without segfault)
python test_embeddings.py
```

## Why This Works

### Key Points:

1. **Python 3.11 instead of 3.12**: Python 3.12 has compatibility issues with current PyTorch/transformers that cause segfaults
2. **Install PyTorch FIRST**: Installing PyTorch before sentence-transformers ensures correct dependency resolution
3. **MPS Backend**: PyTorch 2.9.0 automatically uses Metal Performance Shaders on Mac for GPU acceleration
4. **Exact versions**: Using pinned versions in requirements_conda.txt ensures reproducibility

### Installation Order Matters:

```
torch → sentence-transformers → other dependencies
```

This order prevents pip from installing incompatible PyTorch versions.

## Validation

The `test_embeddings.py` script validates:

1. ✅ Model loads without segfault
2. ✅ Embeddings are generated correctly (1024-dim for BGE-large)
3. ✅ Nearest-neighbor recovery works (100% self-recovery)
4. ✅ Stable across different batch sizes (1, 5, 10)
5. ✅ TR alignment works correctly
6. ✅ Multiple runs complete successfully

See `text_embedding_validation.txt` for full test results.

## Verification Commands

```bash
# Check Python version
python --version  # Should show: Python 3.11.14

# Check PyTorch and MPS availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Quick model test
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-large-en-v1.5'); print('Model loaded successfully!')"

# Full validation
python test_embeddings.py
```

## Expected Output

When running `python test_embeddings.py`, you should see:

```
Text Embedding Validation Test
Model: BAAI/bge-large-en-v1.5
Device: mps:0  (or cpu on non-Mac systems)
✓ Model loaded successfully without segfault!
✓ Generated embeddings: shape (10, 1024)
✓ Perfect self-recovery: 10/10 (100.0%)
✓ ALL TESTS PASSED - No segfaults detected!
```

## Troubleshooting

### If you still get segfaults:

1. **Verify Python version**: Must be 3.11, not 3.12
   ```bash
   python --version
   ```

2. **Clean install**: Remove environment and start fresh
   ```bash
   conda deactivate
   conda env remove -n giblet-py311
   # Then follow Quick Setup again
   ```

3. **Check conda initialization**: Make sure conda is properly initialized
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate giblet-py311
   ```

4. **Verify package versions**:
   ```bash
   pip list | grep -E "torch|sentence-transformers|transformers"
   ```
   Should show:
   - torch: 2.9.0
   - sentence-transformers: 5.1.2
   - transformers: 4.57.1

### If embeddings seem wrong:

- Check that embeddings are normalized (values should be in [-1, 1] range with mean near 0)
- Self-similarity should always be 1.0 for normalized embeddings
- Run the validation script to check all metrics

## Next Steps

Once environment is set up:

1. Use `giblet.data.text.TextProcessor` for production embeddings
2. Load annotations from `data/annotations.xlsx`
3. Generate embeddings aligned to fMRI TRs
4. Use for training multimodal autoencoder

## Example Usage

```python
from giblet.data.text import TextProcessor

# Initialize processor (will auto-detect MPS device on Mac)
processor = TextProcessor(
    model_name='BAAI/bge-large-en-v1.5',
    tr=1.5
)

# Convert annotations to TR-aligned embeddings
embeddings, metadata = processor.annotations_to_embeddings(
    'data/annotations.xlsx',
    n_trs=950  # Number of TRs in your fMRI run
)

# embeddings.shape -> (950, 1024)
# One 1024-dim embedding per TR

# Recover text from embeddings
texts = processor.embeddings_to_text(embeddings, metadata)
```

## Package Details

See `requirements_conda.txt` for complete list of exact package versions.

Key packages:
- PyTorch 2.9.0 (with MPS support for Mac)
- sentence-transformers 5.1.2
- transformers 4.57.1
- numpy 2.3.4
- pandas 2.3.3
- scikit-learn 1.7.2

All versions tested and confirmed working without segfaults on macOS (Apple Silicon).

# PyTorch Segfault Fix - Complete Summary

**Date**: 2025-10-29
**Status**: ✅ RESOLVED - All tests passing, no segfaults
**Working Directory**: `/Users/jmanning/giblet-responses`

---

## Problem

Python 3.12 + PyTorch was causing **segmentation faults** when loading sentence-transformers models (BAAI/bge-large-en-v1.5), making text embedding functionality completely broken.

## Solution

Created clean conda environment with **Python 3.11** + carefully ordered package installation. Result: **ZERO segfaults**, stable operation, full functionality restored.

---

## Deliverables

### 1. Working Conda Environment
- **Name**: `giblet-py311`
- **Location**: `/Users/jmanning/miniconda3/envs/giblet-py311`
- **Python**: 3.11.14
- **Status**: ✅ Active and tested

### 2. Package Configuration
- **File**: `requirements_conda.txt`
- **Key Versions**:
  - PyTorch: 2.9.0 (with MPS support)
  - sentence-transformers: 5.1.2
  - transformers: 4.57.1
  - numpy: 2.3.4
  - pandas: 2.3.3
  - scikit-learn: 1.7.2
- **Status**: ✅ All versions tested and working

### 3. Validation Script
- **File**: `test_embeddings.py`
- **Purpose**: Comprehensive validation of text embedding functionality
- **Tests**:
  1. Model loading (no segfault)
  2. Embedding generation (10 samples from real data)
  3. Nearest-neighbor recovery (100% success rate)
  4. Batch size stability (1, 5, 10 samples)
  5. TR alignment for fMRI (20 TRs)
  6. Multiple runs (stability check)
- **Status**: ✅ All tests pass

### 4. Validation Results
- **File**: `text_embedding_validation.txt`
- **Key Results**:
  - Device: mps:0 (GPU acceleration active)
  - Model: BAAI/bge-large-en-v1.5
  - Embeddings: (10, 1024) shape
  - Self-recovery: 10/10 (100%)
  - TR alignment: 20/20 successful
  - Segfaults: 0
- **Status**: ✅ Perfect results

### 5. Documentation
- **SETUP_CONDA_ENVIRONMENT.md**: Comprehensive setup guide with troubleshooting
- **QUICK_START_TEXT_EMBEDDINGS.md**: TL;DR quick reference for immediate use
- **notes/pytorch_segfault_resolution_2025-10-29.md**: Detailed session notes
- **Status**: ✅ Complete documentation

---

## Setup Instructions (Copy-Paste Ready)

```bash
# 1. Create conda environment with Python 3.11
conda create -n giblet-py311 python=3.11 -y

# 2. Activate environment
conda activate giblet-py311

# 3. Install PyTorch FIRST (critical!)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# 4. Install remaining dependencies
pip install -r requirements_conda.txt

# 5. Verify installation
python test_embeddings.py
```

**Expected output**: "✓ ALL TESTS PASSED - No segfaults detected!"

---

## Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Segfaults | 0 | 0 | ✅ |
| Model loads | Yes | Yes | ✅ |
| Embeddings generated | Yes | Yes (1024-dim) | ✅ |
| Self-recovery accuracy | >95% | 100% | ✅ |
| Batch stability | All sizes | 1, 5, 10 tested | ✅ |
| TR alignment | Functional | 20/20 TRs | ✅ |
| Multiple runs | Stable | 3/3 passed | ✅ |
| GPU acceleration | Active | MPS enabled | ✅ |
| Documentation | Complete | 4 docs created | ✅ |

**Overall**: ✅ 9/9 metrics met - Complete success

---

## Key Technical Details

### Why This Works

1. **Python 3.11 vs 3.12**: Python 3.12 has compatibility issues with current PyTorch/transformers that cause memory corruption and segfaults. Python 3.11 is stable.

2. **Installation Order Matters**: Installing PyTorch BEFORE sentence-transformers prevents pip from installing incompatible PyTorch versions as dependencies.

3. **MPS Acceleration**: PyTorch 2.9.0 automatically detects and uses Metal Performance Shaders on Apple Silicon Macs for GPU acceleration.

### Installation Order

```
CRITICAL ORDER:
1. torch==2.9.0           ← Install FIRST
2. sentence-transformers   ← Install SECOND
3. everything else         ← Install LAST
```

Breaking this order can cause version conflicts.

---

## Validation Test Summary

### Test 1: Model Loading
```
✓ Model: BAAI/bge-large-en-v1.5
✓ Device: mps:0
✓ Dimension: 1024
✓ No segfault
```

### Test 2: Embedding Generation
```
✓ Input: 10 annotation texts
✓ Output shape: (10, 1024)
✓ Embeddings normalized: mean=-0.000221, std=0.031249
✓ Values in expected range: [-0.102, 0.265]
```

### Test 3: Nearest-Neighbor Recovery
```
✓ Self-similarity: 1.000000 (all 10/10 samples)
✓ Text recovery: 100% exact matches
✓ Next-best similarities: 0.69-0.74 (good separation)
```

### Test 4: Batch Stability
```
✓ Batch size 1: (1, 1024)
✓ Batch size 5: (5, 1024)
✓ Batch size 10: (10, 1024)
```

### Test 5: TR Alignment
```
✓ Input: 10 segments
✓ Output: 20 TRs @ 1.5s each
✓ Coverage: 20/20 TRs have segments (no gaps)
✓ Time window: 0.0s - 30.0s
```

### Test 6: Multiple Runs
```
✓ Run 1: PASSED
✓ Run 2: PASSED
✓ Run 3: PASSED
✓ No crashes, no segfaults, consistent results
```

---

## Files Created/Modified

### New Files
1. `/Users/jmanning/giblet-responses/requirements_conda.txt` - Pinned dependencies
2. `/Users/jmanning/giblet-responses/test_embeddings.py` - Validation script
3. `/Users/jmanning/giblet-responses/text_embedding_validation.txt` - Test results
4. `/Users/jmanning/giblet-responses/SETUP_CONDA_ENVIRONMENT.md` - Full setup guide
5. `/Users/jmanning/giblet-responses/QUICK_START_TEXT_EMBEDDINGS.md` - Quick reference
6. `/Users/jmanning/giblet-responses/notes/pytorch_segfault_resolution_2025-10-29.md` - Session notes
7. `/Users/jmanning/giblet-responses/PYTORCH_SEGFAULT_FIX_SUMMARY.md` - This file

### Existing Files (No Changes Needed)
- `giblet/data/text.py` - Works perfectly with new environment, no modifications required

---

## Usage Example

```python
from giblet.data.text import TextProcessor

# Initialize processor
processor = TextProcessor(
    model_name='BAAI/bge-large-en-v1.5',
    tr=1.5,
    device=None  # Auto-detects MPS on Mac
)

# Get model info
info = processor.get_embedding_info()
print(f"Device: {info['device']}")  # mps:0
print(f"Dim: {info['embedding_dim']}")  # 1024

# Convert annotations to TR-aligned embeddings
embeddings, metadata = processor.annotations_to_embeddings(
    'data/annotations.xlsx',
    n_trs=950  # Sherlock movie length
)
# Returns: (950, 1024) array

# Recover text from embeddings
texts = processor.embeddings_to_text(embeddings, metadata)
# Returns: List of 950 texts, one per TR
```

---

## Troubleshooting

### Still getting segfaults?

1. **Check Python version**:
   ```bash
   python --version  # Must show 3.11.x
   ```

2. **Verify package versions**:
   ```bash
   pip list | grep -E "torch|sentence"
   # Should show torch 2.9.0, sentence-transformers 5.1.2
   ```

3. **Clean reinstall**:
   ```bash
   conda deactivate
   conda env remove -n giblet-py311
   # Then follow setup instructions again
   ```

4. **Check conda initialization**:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate giblet-py311
   ```

---

## Next Steps

1. ✅ Use `giblet-py311` environment for all text embedding work
2. ✅ Run `test_embeddings.py` before each session to verify stability
3. ✅ Use `TextProcessor` for production embedding generation
4. ✅ Ready to integrate text embeddings with multimodal autoencoder
5. Consider: Add validation to CI/CD pipeline
6. Consider: Update main project README with Python version requirement

---

## System Information

**Environment:**
- OS: macOS (Apple Silicon)
- Conda: miniconda3
- Environment: giblet-py311
- Python: 3.11.14

**Hardware:**
- Device: mps:0 (Metal Performance Shaders)
- GPU Acceleration: Active
- MPS Available: True

**Package Manager:**
- pip: 25.2
- setuptools: 80.9.0
- wheel: 0.45.1

---

## Conclusion

**Problem**: Python 3.12 + PyTorch = Segmentation faults
**Solution**: Python 3.11 + Careful package ordering
**Result**: Zero segfaults, 100% functionality, stable operation
**Status**: ✅ READY FOR PRODUCTION

All success criteria met. Text embedding functionality fully restored and validated. No code changes needed to existing `giblet/data/text.py` module - it works perfectly with the new environment.

**Ready to train that multimodal autoencoder!** 🚀

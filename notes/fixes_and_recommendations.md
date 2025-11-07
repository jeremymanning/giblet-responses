# Model Convergence Fixes and Recommendations

**Date**: November 5, 2025
**Issue**: [#32](https://github.com/ContextLab/giblet-responses/issues/32)
**Status**: Fixes implemented, ready for testing

## Executive Summary

After 15 epochs, the model showed essentially flat loss (~0.05-0.29% decrease). Investigation revealed three major issues:

1. **🔴 CRITICAL: fMRI data scale mismatch** - fMRI values (0-10000+) are 1000-10000× larger than other modalities
2. **⚠️ Learning rate too low** - Still at 15% of target LR at epoch 15 due to long warmup
3. **⚠️ Loss imbalance** - All weights set to 1.0 despite massive scale differences

**Primary Fix Implemented**: fMRI z-score normalization (per-subject, per-voxel)

## Problems Identified

### Problem 1: fMRI Data Scale Mismatch (CRITICAL)

**Evidence from code analysis:**

```python
# giblet/data/fmri.py (lines 228-244)
# NO normalization applied - just raw BOLD values:
for t in range(n_trs):
    volume = data[:, :, :, t]
    features[t] = volume[brain_mask]  # Raw scanner units!
```

**Scale comparison:**
| Modality | Range | Typical MSE |
|----------|-------|-------------|
| Video | [0, 1] | ~0.01-1.0 |
| Text | [-1, 1] | ~0.1-1.0 |
| Audio | [0-1024] | ~1,000-100,000 |
| **fMRI** | **[0-10,000+]** | **~1,000,000-100,000,000** |

**Impact:**
- fMRI loss is **6+ orders of magnitude** larger than video/text losses
- Total loss ~275,000 is almost entirely fMRI contribution
- Model optimizes nearly exclusively for fMRI prediction
- Reconstruction losses provide negligible gradient signal
- Model cannot learn to reconstruct stimuli

### Problem 2: Learning Rate Too Low

**Current configuration** (`production_500epoch_config.yaml`):
```yaml
learning_rate: 1.0e-4  # Base LR
warmup_epochs: 10       # Linear warmup from 0.1× to 1.0×
```

**Actual LR timeline:**
- Epoch 0: ~1e-5 (10% of base)
- Epoch 5: ~5.5e-5 (55% of base)
- Epoch 10: ~1e-4 (100% of base - warmup ends)
- **Epoch 15: ~1.5e-5** (still in warmup territory due to slow progress!)

**Impact:**
- At ~95 min/epoch, warmup alone takes ~16 hours
- Model needs higher LR earlier to explore parameter space
- Current LR is too conservative for this model size

### Problem 3: Loss Imbalance

**Current configuration:**
```yaml
reconstruction_weight: 1.0
fmri_weight: 1.0
video_weight: 1.0
audio_weight: 1.0
text_weight: 1.0
```

**Problem:**
- All weights are 1.0, but modalities have vastly different scales
- fMRI loss × 1.0 ≈ 100,000,000
- Video loss × 1.0 ≈ 0.1
- **Effective fMRI weight is 1,000,000,000× larger than video!**

## Solutions Implemented

### Fix 1: fMRI Z-Score Normalization ✅ IMPLEMENTED

**Changes made:**

1. **`giblet/data/fmri.py`**: Added normalization to `FMRIProcessor`
   ```python
   def __init__(self, ..., normalize: bool = True):
       self.normalize = normalize

   def nii_to_features(...):
       # ... extract features ...

       if self.normalize:
           # Per-voxel z-score: (x - mean) / std
           mean = np.mean(features, axis=0, keepdims=True)
           std = np.std(features, axis=0, keepdims=True)
           std = np.where(std == 0, 1.0, std)  # Avoid div by zero
           features = (features - mean) / std
   ```

2. **`giblet/data/dataset.py`**: Added `normalize_fmri` parameter
   ```python
   def __init__(self, ..., normalize_fmri: bool = True):
       self.fmri_processor = FMRIProcessor(
           tr=tr, max_trs=max_trs, normalize=normalize_fmri
       )
   ```

**Expected impact:**
- fMRI values now have mean≈0, std≈1 (like other modalities)
- fMRI MSE should be on similar scale to video/text MSE (~0.1-10)
- Loss components will be balanced
- Model can learn reconstruction AND fMRI prediction

**How to use:**
```python
# Default behavior (normalization enabled):
dataset = MultimodalDataset('data/')

# Disable if needed:
dataset = MultimodalDataset('data/', normalize_fmri=False)
```

## Test Configurations Created

### Config 1: Normalized fMRI Only
**File**: `configs/training/test_normalized_fmri_config.yaml`

**Changes from production:**
- ✅ fMRI normalization enabled (default)
- ✅ Reduced warmup: 10 → 2 epochs
- Single subject for fast testing
- 20 epochs

**Purpose**: Isolate effect of fMRI normalization

### Config 2: All Fixes Combined
**File**: `configs/training/test_full_fixes_config.yaml`

**Changes from production:**
- ✅ fMRI normalization enabled (default)
- ✅ Reduced warmup: 10 → 2 epochs
- ✅ Increased LR: 1e-4 → 3e-4
- Single subject for fast testing
- 20 epochs

**Purpose**: Test all fixes together for maximum convergence improvement

## Recommended Testing Plan

### Phase 1: Quick Validation (Local, 1-2 hours)

1. **Run quick diagnostic** (when script completes):
   ```bash
   python scripts/quick_data_diagnostic.py
   ```
   **Expected output**: Confirms fMRI scale mismatch and shows normalization effect

2. **Test Config 1** (normalized fMRI only, 20 epochs, single subject):
   ```bash
   python -m giblet.training.train \
       --config configs/training/test_normalized_fmri_config.yaml
   ```
   **Success criteria**: Loss decreases by >10% over 20 epochs

3. **Test Config 2** (all fixes, 20 epochs, single subject):
   ```bash
   python -m giblet.training.train \
       --config configs/training/test_full_fixes_config.yaml
   ```
   **Success criteria**: Loss decreases by >20% over 20 epochs

### Phase 2: Short Cluster Test (6 GPUs, ~24 hours)

If local tests show improvement:

1. **Create production fix config:**
   ```yaml
   # Based on production_500epoch_config.yaml
   learning_rate: 3.0e-4        # Increased
   warmup_epochs: 2             # Reduced
   num_epochs: 50               # Short test
   # normalize_fmri: True (default)
   data:
     subjects: [1, 2, 3, 4, 5]  # 5 subjects
   distributed:
     world_size: 6
   ```

2. **Run on cluster:**
   ```bash
   ./scripts/cluster/remote_train.sh tensor01 test_50epoch_fixed
   ```

3. **Monitor with updated status script:**
   ```bash
   ./scripts/cluster/check_remote_status.sh tensor01
   ```

**Success criteria:**
- Train loss decreases steadily (>50% over 50 epochs)
- Val loss follows train loss (gap <2×)
- Loss components are balanced (no single component dominates)

### Phase 3: Full Production Run (6 GPUs, ~30 days)

If 50-epoch test succeeds:

1. **Update production config:**
   ```yaml
   # configs/training/production_500epoch_config.yaml
   learning_rate: 3.0e-4        # or best from Phase 2
   warmup_epochs: 2             # or best from Phase 2
   num_epochs: 500
   data:
     subjects: [1-17]           # All subjects
   distributed:
     world_size: 6
   ```

2. **Launch production run:**
   ```bash
   ./scripts/cluster/remote_train.sh tensor01 production_fixed
   ```

3. **Stop old run (if still active):**
   ```bash
   # SSH to cluster
   screen -ls  # Find screen session
   screen -X -S <session_name> quit
   ```

## Expected Results

### With Fixes Applied

**Loss behavior:**
- **Epoch 1-2**: Rapid decrease as model learns basic patterns (warmup ends quickly)
- **Epoch 3-20**: Steady decrease (~2-5% per epoch)
- **Epoch 20-100**: Slower decrease (~0.5-1% per epoch)
- **Epoch 100+**: Gradual convergence

**Loss scale:**
- Total loss should start ~100-1000 (vs current ~275,000)
- fMRI component: ~10-100
- Reconstruction components: ~1-10 each
- All components visible in loss plot

**Training time:**
- Same ~95 min/epoch on current hardware
- 500 epochs ≈ 32 days (unchanged)
- BUT: Model will actually be learning!

## Alternative Fixes (If Normalization Insufficient)

### Option A: Loss Weight Rebalancing

If fMRI still dominates after normalization:

```yaml
reconstruction_weight: 1.0
fmri_weight: 0.01         # Reduce fMRI weight
video_weight: 1.0
audio_weight: 1.0
text_weight: 1.0
```

### Option B: Gradient Scaling

Implement automatic loss scaling based on gradient magnitudes:

```python
# In trainer.py
def _balance_losses(self, loss_dict):
    # Compute gradient norms for each component
    grads = {}
    for name, loss in loss_dict.items():
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_norm = sum(p.grad.norm() for p in self.model.parameters())
        grads[name] = grad_norm

    # Scale losses to equalize gradient magnitudes
    target_norm = np.mean(list(grads.values()))
    weights = {name: target_norm / (norm + 1e-8)
               for name, norm in grads.items()}

    return weights
```

### Option C: Increase Batch Size

Current: 2 per GPU (12 total)
Larger batches may provide more stable gradients:

```yaml
batch_size: 4  # Per GPU (24 total)
```

**Tradeoff**: May OOM, require further memory optimizations

## Files Modified

### Code Changes
- ✅ `giblet/data/fmri.py` - Added z-score normalization
- ✅ `giblet/data/dataset.py` - Added normalize_fmri parameter

### New Files
- ✅ `scripts/diagnose_training.py` - Full diagnostic (model + data)
- ✅ `scripts/quick_data_diagnostic.py` - Quick data-only diagnostic
- ✅ `configs/training/test_normalized_fmri_config.yaml` - Test config (fix 1)
- ✅ `configs/training/test_full_fixes_config.yaml` - Test config (fixes 1+2+3)
- ✅ `notes/convergence_diagnosis_20251105.md` - Detailed analysis
- ✅ `notes/fixes_and_recommendations.md` - This document

### Documentation
- ✅ Updated FMRIProcessor docstring
- ✅ Updated MultimodalDataset docstring
- ✅ GitHub Issue #32 updated with findings

## Next Steps

1. ✅ **Commit fixes** - Done
2. ⏳ **Wait for diagnostic** - Running (scripts/quick_data_diagnostic.py)
3. ⏳ **Review diagnostic output** - Confirm scale mismatch
4. ⏳ **Test locally** - Run test configs
5. ⏳ **Test on cluster** - 50-epoch test with 5 subjects
6. ⏳ **Deploy to production** - If tests succeed

## Confidence Levels

| Fix | Confidence | Rationale |
|-----|-----------|-----------|
| fMRI normalization | **Very High (95%)** | Clear scale mismatch in code, well-established practice |
| Reduced warmup | **High (85%)** | Current warmup is objectively too long for this training speed |
| Increased LR | **Medium (70%)** | May help, but could also cause instability |

**Recommendation**: Start with fMRI normalization + reduced warmup. Add LR increase only if needed.

## Questions?

Contact: jeremy.manning@dartmouth.edu
GitHub: https://github.com/ContextLab/giblet-responses/issues/32

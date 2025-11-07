# Model Convergence Diagnosis - November 5, 2025

## Summary

After 15 epochs of training (3% of 500 total), both training and validation losses are essentially flat with minimal decrease (~0.05-0.29%). This document summarizes findings from investigating potential causes.

## Problem Statement

**Training Progress:**
- 15/500 epochs completed (3.0%)
- ~95.5 minutes per epoch
- Projected completion: ~32 days (December 8, 2025)

**Loss Behavior:**
- **Train Loss**: 275,824 - 276,118 (range: 294, or 0.11% variation)
  - Epoch 1: 275,959 → Epoch 15: 275,825
  - **Net change: -134 (-0.05%)**

- **Validation Loss**: 204,916 - 210,534 (range: 5,618, or 2.7% variation)
  - Epoch 1: 207,225 → Epoch 15: 206,622
  - **Net change: -603 (-0.29%)**

**Conclusion**: Losses are essentially flat, indicating model is not learning effectively.

## Investigation Findings

### 1. Learning Rate Configuration

**Config file**: `configs/training/production_500epoch_config.yaml`

**Settings:**
- `learning_rate`: 1e-4 (0.0001)
- `scheduler_type`: 'cosine'
- `warmup_epochs`: 10
- `min_lr`: 1e-6
- `batch_size`: 2 per GPU (12 total across 6 GPUs)
- `optimizer`: AdamW (8-bit via bitsandbytes)
- `weight_decay`: 1e-5

**Warmup Behavior:**
- Uses LinearLR: starts at 0.1× LR, ramps to 1.0× over 10 epochs
- Epoch 0: ~1e-5 (0.00001)
- Epoch 15: ~1.5e-5 (0.000015) - still in warmup phase!

**Analysis:**
✅ **MAJOR FINDING**: At epoch 15, we're still at 0.000015 LR (only 15% of target LR).
- The warmup is too long (10 epochs) for this slow training
- At ~95 min/epoch, warmup alone takes ~16 hours
- Model needs higher LR earlier to make meaningful progress

### 2. Scheduler Implementation

**Code**: `giblet/training/trainer.py:357-399`

```python
# Cosine annealing with warmup
warmup_steps = len(train_loader) * warmup_epochs
self.scheduler = CosineAnnealingLR(
    self.optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=min_lr,
)
self.warmup_scheduler = LinearLR(
    self.optimizer,
    start_factor=0.1,  # Start at 10% of base LR
    end_factor=1.0,    # Ramp to 100% of base LR
    total_iters=warmup_steps,
)
```

**Schedule Timeline:**
- Epochs 0-9: Linear warmup from 1e-5 → 1e-4
- Epochs 10-499: Cosine annealing from 1e-4 → 1e-6

**Current Position (Epoch 15):**
- Should be PAST warmup, but still at low LR due to slow progress
- Need to verify actual LR being used during training

### 3. Data Preprocessing

**Video**: `giblet/data/video.py`
- Normalized to [0, 1] range
- Shape: (90, 160) × 38 frames = 43,200 features per TR
- Frame skip: 4 (memory optimization)

**Audio**: `giblet/data/audio.py`
- Using EnCodec: discrete codes (integers)
- Converted to float for loss computation
- 112 frames per TR × 8 codebooks = large integer values

**Text**: `giblet/data/text.py`
- Embeddings from language model (likely normalized)
- 3,072 dimensions

**fMRI**: `giblet/data/fmri.py`
- Raw voxel values (NO normalization found!)
- 85,810 voxels
- Values are BOLD signal intensities (arbitrary units)
- Typically range from 0-10,000+ depending on scanner

✅ **MAJOR FINDING**: fMRI data appears to NOT be normalized!
- BOLD values are in raw scanner units
- Video is [0,1], audio is discrete codes, text is normalized embeddings
- fMRI values may be 1000-10000×larger than other modalities
- This creates massive scale mismatch in loss computation

### 4. Loss Function

**Code**: `giblet/training/losses.py`

**Components:**
1. **Reconstruction Loss** (MSE):
   - Video MSE
   - Audio MSE (codes converted to float)
   - Text MSE

2. **fMRI Matching Loss** (MSE):
   - MSE between predicted and target fMRI

**Config Weights:**
- `reconstruction_weight`: 1.0
- `fmri_weight`: 1.0
- `video_weight`: 1.0
- `audio_weight`: 1.0
- `text_weight`: 1.0

**Issue**: All weights are 1.0, but modalities have vastly different scales:
- Video: [0, 1] → MSE ~ 0.01-1.0
- Audio: discrete codes [0-1024] → MSE ~ 1000-100000
- Text: normalized embeddings [-1, 1] → MSE ~ 0.1-1.0
- fMRI: raw BOLD [0-10000] → MSE ~ 1000000-100000000

✅ **MAJOR FINDING**: fMRI loss likely dominates total loss!
- fMRI MSE is orders of magnitude larger than other losses
- Model optimizes almost exclusively for fMRI prediction
- Reconstruction losses contribute negligible gradient signal

### 5. Optimizer Configuration

**Optimizer**: AdamW (8-bit via bitsandbytes)
- `lr`: 1e-4 (during warmup: 1e-5 to 1.5e-5 for first 15 epochs)
- `weight_decay`: 1e-5
- `gradient_clip_val`: 1.0

**Analysis:**
- 8-bit optimizer should work fine (memory optimization)
- LR is reasonable for final training, but too low during warmup
- Gradient clipping at 1.0 may be too aggressive given large fMRI gradients

### 6. Model Architecture

**Code**: `giblet/models/autoencoder.py`

**Key Stats:**
- 8.36B parameters
- Uses gradient checkpointing (memory optimization)
- Bottleneck: 2048 dimensions
- Video encoder: ResNet-style architecture
- Audio encoder: Transformer
- Decoder: MLP with 2048 hidden dims

**Potential Issues:**
- Massive model may need more careful initialization
- Gradient checkpointing adds computational cost
- Large parameter count = more potential for vanishing/exploding gradients

## Root Causes (Hypotheses)

### Primary Suspects:

1. **Data Scale Mismatch** (HIGH CONFIDENCE)
   - fMRI values are 1000-10000× larger than other modalities
   - fMRI loss dominates, model ignores reconstruction
   - Need to normalize fMRI or reweight losses

2. **Learning Rate Too Low** (HIGH CONFIDENCE)
   - Still in warmup at epoch 15 (0.000015 vs target 0.0001)
   - Warmup is too long for slow training
   - Model needs higher LR earlier

3. **Loss Imbalance** (MEDIUM CONFIDENCE)
   - All loss weights are 1.0 despite scale differences
   - fMRI loss likely 1000000× larger than video loss
   - Need to rebalance weights or use loss normalization

### Secondary Suspects:

4. **Gradient Flow Issues** (UNKNOWN - need diagnostics)
   - 8.36B parameter model may have vanishing/exploding gradients
   - Gradient checkpointing may exacerbate issues
   - Need to measure gradient magnitudes

5. **Initialization** (UNKNOWN - need diagnostics)
   - Large model may benefit from better initialization
   - Default PyTorch init may not be optimal

6. **Batch Size Too Small** (LOW CONFIDENCE)
   - Batch size = 2 per GPU (12 total)
   - Very small for such a large model
   - May cause noisy gradients

## Recommended Actions

### Immediate (High Priority):

1. **Run Diagnostic Script**
   ```bash
   python scripts/diagnose_training.py \
       --config configs/training/production_500epoch_config.yaml \
       --checkpoint checkpoints/checkpoint_epoch_15.pt \
       --data-samples 100 \
       --loss-batches 10 \
       --output diagnostic_report.txt
   ```

2. **Analyze Current Training**
   - Extract actual LR values from logs
   - Measure loss component magnitudes
   - Check gradient norms

3. **Test Fixes** (create test configs):
   - **Fix A**: Normalize fMRI data (z-score per subject)
   - **Fix B**: Reduce warmup to 2-3 epochs
   - **Fix C**: Increase initial LR to 3e-4
   - **Fix D**: Rebalance loss weights (reduce fMRI weight to 0.001)

### Medium Term:

4. **Implement Data Normalization**
   - Add fMRI z-score normalization to `FMRIProcessor`
   - Per-subject mean/std calculation
   - Store normalization params for denormalization

5. **Create Loss Scaling Strategy**
   - Automatically balance loss components
   - Use gradient magnitude ratios
   - Implement dynamic loss weighting

6. **Test Smaller Model**
   - Create 1B parameter version
   - Faster iteration for debugging
   - Easier to analyze gradient flow

### Long Term:

7. **Hyperparameter Search**
   - Grid search over LR, warmup, loss weights
   - Use smaller dataset for speed
   - Find optimal configuration

8. **Architecture Improvements**
   - Better initialization (Xavier, He, etc.)
   - Layer normalization
   - Skip connections
   - Residual blocks

## Next Steps

1. ✅ Create GitHub issue #32
2. ✅ Create diagnostic script
3. ⏳ Run diagnostics on current checkpoint
4. ⏳ Analyze diagnostic report
5. ⏳ Create test configs with fixes
6. ⏳ Launch short test runs (10 epochs each)
7. ⏳ Compare results and choose best fix
8. ⏳ Launch full production run with fixes

## Files Created

- `scripts/diagnose_training.py` - Diagnostic script
- `notes/convergence_diagnosis_20251105.md` - This file
- GitHub Issue #32 - Tracking issue

## References

- Config: `configs/training/production_500epoch_config.yaml`
- Trainer: `giblet/training/trainer.py`
- Losses: `giblet/training/losses.py`
- Data: `giblet/data/`
- Log: `logs/training_6gpu_20251104_223000.log` (remote)
- Checkpoint: `checkpoints/checkpoint_epoch_15.pt` (remote)

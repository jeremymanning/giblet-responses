# Issue #30: Memory Optimization Analysis

**Date:** 2025-11-02
**Model Size:** 8.5B parameters (8,499,653,234)
**Target:** Fit in 47.4 GB per GPU (A6000/A100)

## Memory Usage Breakdown

### Float32 (Original)
- Parameters: 31.66 GB
- Gradients: 31.66 GB
- Optimizer (Adam): 63.33 GB
- **Total: 126.65 GB** ❌ Does NOT fit

### Bfloat16 (Model + Grads Only)
- Parameters: 15.83 GB
- Gradients: 15.83 GB
- Optimizer (Adam fp32): 63.33 GB
- **Total: 94.99 GB** ❌ Does NOT fit

### Bfloat16 + Mixed Precision Optimizer
- Parameters: 15.83 GB
- Gradients: 15.83 GB
- Optimizer (Adam bf16): 31.66 GB
- **Total: 63.33 GB** ❌ Still does NOT fit
- **With 20% overhead: 76.0 GB** ❌ Definitely does NOT fit

## Root Cause
The model is fundamentally **too large** for single 47.4GB GPUs, even with aggressive bfloat16 optimization.

**Minimum required:** 63.3 GB (without activations/overhead)
**Available:** 47.4 GB
**Shortfall:** 15.9 GB (25% over budget)

## Solutions Implemented (This Session)

### 1. ✅ Model dtype: bfloat16
- **File:** `scripts/train.py` lines 191-204
- **Change:** Convert model to bfloat16 after creation
- **Savings:** 15.83 GB (parameters) + 15.83 GB (gradients) = 31.66 GB total
- **Code:**
  ```python
  if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
      model = model.bfloat16()
  ```

### 2. ✅ Input data: bfloat16
- **File:** `giblet/data/dataset.py` lines 537-541
- **Change:** Return bfloat16 tensors from dataset
- **Savings:** Small (batch size dependent), reduces data transfer overhead
- **Code:**
  ```python
  target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
  tensor = torch.from_numpy(array).to(target_dtype)
  ```

### 3. ✅ Loss computation: float32
- **File:** `giblet/training/trainer.py` lines 480-496, 590-605
- **Change:** Cast outputs to float32 before loss computation
- **Benefit:** Numerical stability without memory overhead (loss is small)
- **Code:**
  ```python
  with autocast(dtype=torch.bfloat16):
      outputs = self.model(video, audio, text, fmri_target=fmri)
      outputs_fp32 = {k: v.float() if torch.is_tensor(v) else v for k, v in outputs.items()}
      loss, loss_dict = self.criterion(outputs_fp32, ...)
  ```

## Additional Strategies Needed

The current changes **reduce memory but are still insufficient**. Additional strategies required:

### Option 1: Gradient Checkpointing ⭐ RECOMMENDED
- **Savings:** ~30-50% activation memory
- **Trade-off:** 20-30% slower training
- **Implementation:** Add `torch.utils.checkpoint` in forward passes
- **Status:** NOT YET IMPLEMENTED

### Option 2: Reduce Batch Size
- **Current:** batch_size=64 per GPU
- **Recommendation:** Try batch_size=16 or 32
- **Savings:** ~75% activation memory (64→16)
- **Trade-off:** Slower convergence, may need more epochs

### Option 3: Model Architecture Changes
- **Reduce bottleneck_dim:** 2048 → 1024 or 512
- **Reduce intermediate layers:** Remove layer or reduce dims
- **Savings:** Potentially 50% model size
- **Trade-off:** Reduced model capacity

### Option 4: Distributed Data Parallel (DDP) with Gradient Accumulation
- **Current:** DDP across 8 GPUs
- **Enhancement:** Accumulate gradients over 2-4 steps
- **Effective batch size:** 64 * 8 * 4 = 2048
- **Per-GPU memory:** Reduced activations

### Option 5: DeepSpeed/FSDP (Fully Sharded Data Parallel)
- **Framework:** DeepSpeed ZeRO or PyTorch FSDP
- **Sharding:** Shard optimizer states across GPUs
- **Savings:** optimizer_states / num_gpus
- **Trade-off:** More complex setup, communication overhead

## Recommended Next Steps

1. **Immediate:** Test current bfloat16 changes on tensor01 to confirm memory usage
2. **If OOM persists:** Implement gradient checkpointing
3. **If still OOM:** Reduce batch size to 16 or 32
4. **Long-term:** Consider DeepSpeed ZeRO-2 for optimizer state sharding

## Testing Plan

1. ✅ Local testing: Verify code runs without errors (CPU fallback)
2. ⏳ **tensor01 single GPU:** Test with batch_size=1 to measure base memory
3. ⏳ **tensor01 single GPU:** Test with batch_size=16 to check if fits
4. ⏳ **tensor01 8 GPUs:** Test full distributed training
5. ⏳ **Monitor:** Track actual GPU memory usage via `nvidia-smi`

## Files Modified

1. `scripts/train.py` - Model bfloat16 conversion
2. `giblet/data/dataset.py` - Input data bfloat16 conversion
3. `giblet/training/trainer.py` - Loss computation float32 casting
4. `notes/issue30_memory_optimization_analysis.md` - This file

## Commits

- [ ] Commit bfloat16 optimization changes
- [ ] Test on tensor01 and report results
- [ ] Implement gradient checkpointing if needed
- [ ] Final commit with working solution

## References

- Issue #30: "CUDA out of memory" error during training
- PyTorch autocast docs: https://pytorch.org/docs/stable/amp.html
- PyTorch gradient checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/

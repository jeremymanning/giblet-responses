# Phase 5 Next Steps - Test Fixing Continuation

**Date:** 2025-11-04
**Current Commit:** 739c7ba
**Status:** In progress - encoder tests done, autoencoder tests partially done

---

## Summary of Progress

### ✅ Completed
1. **tests/models/test_encoder.py:** 20/21 passing (1 GPU test skipped) ✅
   - Fixed VideoEncoder API mismatch (input_height/width → input_dim)
   - Fixed AudioEncoder temporal concatenation (frames_per_tr=1 for tests)
   - Updated parameter count assertions for temporal concat models

2. **tests/models/test_autoencoder.py:** Partially fixed
   - ✅ All bottleneck_dim assertions fixed (8000 → 2048)
   - ✅ test_forward_pass_eval fixed with temporal parameters
   - ❌ **17 more test methods need temporal parameters**

### 🔄 In Progress
**tests/models/test_autoencoder.py** - Need to add `video_frames_per_tr=1, audio_frames_per_tr=1` to 17 more test methods

---

## Immediate Next Steps

### 1. Complete test_autoencoder.py

Add temporal parameters to these lines:
```
34:  test_autoencoder_init
85:  test_forward_pass_train_no_fmri
122: test_forward_pass_train_with_fmri
152: test_forward_pass_different_weights (already has custom params - ADD to them)
175: test_backward_pass
198: test_encode_only
223: test_decode_only
238: test_parameter_count
265: test_custom_dimensions (already has custom dims - ADD to them)
292: test_checkpoint_save_load
337: test_checkpoint_with_optimizer
365: test_gpu_forward (GPU test)
383: test_reconstruction_quality_sanity
438: test_prepare_requires_init
451: test_full_training_step
477: test_batch_consistency
516: test_gradient_accumulation
547: test_multiple_epochs
577: test_eval_train_mode_switch
```

**Pattern to apply:**
```python
# Before:
model = MultimodalAutoencoder()

# After:
model = MultimodalAutoencoder(
    video_frames_per_tr=1,  # Single frame for unit testing
    audio_frames_per_tr=1   # Single time step for unit testing
)
```

**Special cases:**
- Lines 152, 265: Already have custom parameters - ADD temporal params to existing ones
- Line 365: GPU test - add params but keep `.cuda()`
- Line 223: test_decode_only - doesn't create autoencoder, uses bottleneck directly

### 2. Fix Remaining Test Files

Priority order (28 files total):
1. tests/models/test_decoder.py
2. tests/models/test_audio_encoder_encodec.py
3. tests/models/test_audio_decoder_encodec.py
4. tests/models/test_encoder_demo.py
5. tests/data/ files (8 files)
6. tests/integration/ files (2 files)
7. tests/diagnostics/ files (3 files)
8. tests/utils/test_visualization.py
9. tests/ root level files (9 files)

---

## Key Technical Patterns

### Issue #29 Changes (Root Cause)
- **VideoEncoder:** Conv2D → Linear for temporal concatenation
  - Old API: `input_height`, `input_width`
  - New API: `input_dim` (flattened)
  - Default `input_dim=1,641,600` (38 frames × 90 × 160 × 3)

- **AudioEncoder:** Expects flattened temporal input
  - Input shape: `(batch, input_mels × frames_per_tr)`
  - Default: `2048 × 65 = 133,120` features

- **MultimodalEncoder:**
  - Default `bottleneck_dim=2048` (NOT 8000!)
  - Default `video_frames_per_tr=19`
  - Default `audio_frames_per_tr=65`

### Test Fix Strategy
1. **For unit tests:** Use `video_frames_per_tr=1, audio_frames_per_tr=1`
2. **For integration tests:** Can use defaults or still use single frames
3. **Parameter count assertions:** Update to match large temporal concat models

### Common Fixes
```python
# VideoEncoder
encoder = VideoEncoder(
    input_dim=43200,  # Single frame: 90 × 160 × 3
    output_features=1024
)

# AudioEncoder
encoder = AudioEncoder(
    input_mels=128,
    frames_per_tr=1,  # Single time step for testing
    output_features=256
)

# MultimodalEncoder/Autoencoder
model = MultimodalAutoencoder(
    video_frames_per_tr=1,  # Single frame for unit testing
    audio_frames_per_tr=1   # Single time step for unit testing
)
```

---

## Documentation Bug Found

**giblet/models/autoencoder.py:**
- Line 48 (docstring): Says `bottleneck_dim : int, default=8000`
- Line 73 (code): Actually `bottleneck_dim: int = 2048`

**Fix needed:** Update docstring to match code (2048, not 8000)

---

## Testing Strategy

### Running Tests
```bash
# Single file
PYTHONPATH=/Users/jmanning/giblet-responses:$PYTHONPATH python -m pytest tests/models/test_encoder.py -v --tb=short -k "not slow"

# Full models directory
PYTHONPATH=/Users/jmanning/giblet-responses:$PYTHONPATH python -m pytest tests/models/ -v --tb=short -k "not slow"

# All tests (warning: very slow due to imports)
PYTHONPATH=/Users/jmanning/giblet-responses:$PYTHONPATH python -m pytest tests/ -v --tb=short -k "not slow"
```

### Commit Strategy
- Commit after each file or small group of files
- Use descriptive messages linking to Issue #31
- Include test results in commit message

---

## Current Commits
- `d964b4f`: Fix encoder tests (13 failures → 20/21 passing)
- `739c7ba`: Partial fix for autoencoder tests (bottleneck_dim update)
- `a4822be`: Phase 5 fixture/import fixes

---

## Todo List Progress

[1. ✅ Fix tests/models/test_encoder.py]
[2. 🔄 Fix tests/models/test_autoencoder.py (partial - bottleneck dims done)]
[3. ⏭️  Fix tests/models/test_decoder.py]
[4. ⏭️  Fix tests/models/test_audio_encoder_encodec.py]
[5. ⏭️  Fix tests/models/test_audio_decoder_encodec.py]
... (24 more files)

---

## References

- **Phase 5 Roadmap:** notes/PHASE5_ROADMAP.md
- **Issue #29:** Encoder refactoring (Conv2D → Linear)
- **Issue #31:** Repository reorganization (5 phases)
- **Session notes:** notes/2025-11-04_session_issue31_continued.md
- **Failure analysis:** notes/2025-11-04_issue31_phase5_test_failures.md

---

**Ready to continue:** Complete autoencoder tests, then systematically work through remaining 26 test files.

# Session: Issue #31 Phase 5 Continuation

**Date:** 2025-11-04
**Continuation from:** Previous session (ran out of context)
**Status:** In progress - fixing encoder tests

---

## Session Progress

### ✅ Completed

1. **Updated Issue #31** with progress report showing partial Phase 5 completion
2. **Verified test status:**
   - tests/test_sync.py: 23/23 passing ✅
   - tests/models/test_encoder.py: 7/21 passing, 13 failing ❌

3. **Analyzed root cause:**
   - VideoEncoder and AudioEncoder refactored in Issue #29 from Conv2D to Linear (temporal concatenation)
   - Tests NOT updated to match new API
   - Tests use old API expecting `input_height/width` parameters

4. **Created documentation:**
   - [notes/2025-11-04_issue31_phase5_test_failures.md](./2025-11-04_issue31_phase5_test_failures.md)
   - Comprehensive failure analysis with fixes needed

### ✅ Completed (Continued)

5. **Fixed tests/models/test_encoder.py:**
   - Updated 13 failing test methods to match current encoder API
   - Applied single-frame testing strategy (`video_frames_per_tr=1`, `audio_frames_per_tr=1`)
   - Updated parameter count assertions to match temporal concat model sizes
   - **Result:** 20/21 passing, 1 skipped (GPU test) ✅

6. **Verified fixes work:**
   - Ran `pytest tests/models/test_encoder.py`
   - All tests pass successfully
   - Total test time: 107.98s

### ✅ Completed (Continued)

7. **Partial fix for tests/models/test_autoencoder.py:**
   - Fixed bottleneck_dim assertions (8000 → 2048) using replace_all
   - Fixed test_forward_pass_eval with video_frames_per_tr=1, audio_frames_per_tr=1
   - **Remaining:** Need to add temporal parameters to 17 more test methods

### 🔄 Current Work

**Task:** Complete tests/models/test_autoencoder.py, then continue with other test files

**Autoencoder fixes remaining:**
- 17 more test methods need `video_frames_per_tr=1, audio_frames_per_tr=1` parameters
- All at lines: 34, 85, 122, 152, 175, 198, 223, 238, 292, 337, 365, 383, 438, 451, 477, 516, 547, 577

**Key insights from encoder test fixes:**
- VideoEncoder takes `input_dim` (flattened), not `input_height/width`
- VideoEncoder expects temporal concatenation by default:
  - Default `input_dim=1,641,600` (38 frames × 90 × 160 × 3)
  - Tests provide single frames: 43,200 (1 frame × 90 × 160 × 3)
- AudioEncoder expects flattened temporal: `input_mels × frames_per_tr`
  - Default expects: 2048 × 65 = 133,120 features
  - Tests provide: 128 features
- MultimodalEncoder creates sub-encoders with temporal concat defaults:
  - VideoEncoder: `input_dim = video_frames_per_tr × H × W × 3` (default 19 frames = 820,800)
  - AudioEncoder: `input_dim = audio_mels × audio_frames_per_tr` (default 2048 × 65 = 133,120)

---

## Required Fixes for tests/models/test_encoder.py

### 1. VideoEncoder Tests (lines 30-67)

**test_video_encoder_init:**
```python
# OLD (lines 32-36):
encoder = VideoEncoder(
    input_height=90,      # ❌ Doesn't exist
    input_width=160,      # ❌ Doesn't exist
    output_features=1024
)

# FIX:
encoder = VideoEncoder(
    input_dim=43200,  # 90 × 160 × 3 (single frame for testing)
    output_features=1024
)
# Update assertions:
assert encoder.input_dim == 43200
assert encoder.output_features == 1024
```

**test_video_encoder_forward:**
```python
# OLD (lines 43-47):
encoder = VideoEncoder(
    input_height=90,
    input_width=160,
    output_features=1024
)

# FIX:
encoder = VideoEncoder(
    input_dim=43200,  # Match single frame input
    output_features=1024
)
# Keep 4D input (encoder auto-flattens):
x = torch.randn(batch_size, 3, 90, 160)
```

**test_video_encoder_parameter_count:**
```python
# Line 67: Update assertion
assert n_params < 50_000_000  # ❌ Current: 6.7B

# FIX:
assert n_params < 7_000_000_000  # Updated for temporal concat model
```

### 2. AudioEncoder Tests (lines 83-108)

**test_audio_encoder_forward:**
```python
# OLD (lines 85-93):
encoder = AudioEncoder(
    input_mels=128,
    output_features=256
)
x = torch.randn(batch_size, 128)  # ❌ Wrong size

# FIX Option A (single frame testing):
encoder = AudioEncoder(
    input_mels=128,
    frames_per_tr=1,  # Single frame for testing
    output_features=256
)
x = torch.randn(batch_size, 128)  # Now matches 128×1=128

# FIX Option B (temporal concat):
encoder = AudioEncoder(
    input_mels=128,
    frames_per_tr=65,  # Default
    output_features=256
)
x = torch.randn(batch_size, 128 * 65)  # Flattened temporal
```

**test_audio_encoder_parameter_count:**
```python
# Line 108: Update assertion
assert n_params < 10_000_000  # ❌ Current: 68M

# FIX:
assert n_params < 70_000_000  # Updated for temporal concat model
```

### 3. MultimodalEncoder Tests (lines 166-395)

**All MultimodalEncoder tests providing single frames need:**

```python
# OLD (lines 168-174):
encoder = MultimodalEncoder()  # Uses defaults: 19 video frames, 65 audio frames
video = torch.randn(1, 3, 90, 160)      # ❌ Single frame
audio = torch.randn(1, 2048)             # ❌ Single time step

# FIX Option A (single frame testing):
encoder = MultimodalEncoder(
    video_frames_per_tr=1,    # Single frame for testing
    audio_frames_per_tr=1     # Single time step for testing
)
video = torch.randn(1, 3, 90, 160)
audio = torch.randn(1, 2048)

# FIX Option B (temporal concat - more realistic):
encoder = MultimodalEncoder()  # Use defaults
# Provide temporal concatenation:
video = torch.randn(1, 3 * 19, 90, 160)  # 19 frames flattened
# OR: video = torch.randn(1, 820800)  # Pre-flattened
audio = torch.randn(1, 2048 * 65)  # 65 frames flattened
```

**test_encoder_parameter_count:**
```python
# Line 254: Update assertion
assert param_dict['total'] < 2_000_000_000  # ❌ Current: 4.9B

# FIX:
assert param_dict['total'] < 5_000_000_000  # Updated for temporal concat model
```

---

## Recommended Fix Strategy

**Use Option A (single-frame testing) for simplicity:**

1. VideoEncoder tests: Use `input_dim=43200` for single frame
2. AudioEncoder tests: Use `frames_per_tr=1` for single time step
3. MultimodalEncoder tests: Use `video_frames_per_tr=1`, `audio_frames_per_tr=1`
4. Update all parameter count assertions to match current model sizes

**Rationale:**
- Unit tests should be fast (single frames)
- Integration tests can use temporal concat (already marked @pytest.mark.integration)
- Keeps test inputs simple and clear
- Matches current training reality (temporal concat is expensive)

---

## Next Steps

1. ✅ Document findings (this file)
2. Apply fixes to tests/models/test_encoder.py
3. Run tests to verify fixes
4. Commit test fixes
5. Continue with remaining test files
6. Complete Phase 5

---

**Files Modified This Session:**
- notes/2025-11-04_issue31_phase5_test_failures.md (created)
- notes/2025-11-04_session_issue31_continued.md (this file)

**Commit Status:**
- Previous commit: a4822be (Phase 5 fixture/import fixes)
- Pending: Test fixes for encoder tests

---

**Ready to proceed with test fixes**

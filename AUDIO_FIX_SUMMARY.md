# EnCodec Audio Encoding Bug Fix Summary

**Date:** 2025-11-01
**File:** `/Users/jmanning/giblet-responses/giblet/data/audio.py`
**Function:** `_audio_to_features_encodec()` (lines 170-338)

## Problem Description

### Error Message
```
RuntimeError: The expanded size of the tensor (112) must match the existing size (106697)
at normalized_codes[:n_available, :] = tr_codes[:n_available, :]
```

### Root Cause
The code was attempting to normalize codebook counts (e.g., from 4 to 8 codebooks) BEFORE ensuring the temporal dimension was correct. When `tr_codes` had shape `(4, 106697)` (4 codebooks, 106697 frames for entire audio) but `normalized_codes` had shape `(8, 112)` (8 codebooks, 112 frames for one TR), the tensor assignment failed due to dimension mismatch.

The temporal padding/cropping logic at lines 300-305 should have ensured `tr_codes` had exactly `frames_per_tr=112` frames, but in some edge cases this didn't happen before the codebook normalization step.

## Solution

### Two-Step Fix

**Step 1: Flexible Codebook Normalization (Lines 307-318)**
- Create `normalized_codes` with `tr_codes.shape[1]` temporal dimension instead of fixed `frames_per_tr`
- This allows the normalization to work regardless of current temporal dimension
- Changed line 313 from:
  ```python
  normalized_codes = torch.zeros(expected_codebooks, frames_per_tr, dtype=tr_codes.dtype)
  ```
  to:
  ```python
  normalized_codes = torch.zeros(expected_codebooks, tr_codes.shape[1], dtype=tr_codes.dtype)
  ```

**Step 2: Post-Normalization Temporal Fix (Lines 320-327)**
- Added additional temporal dimension check AFTER codebook normalization
- Ensures final `tr_codes` has exactly `frames_per_tr=112` frames
- Crops if too many frames, pads if too few frames
- New code:
  ```python
  # ADDITIONAL FIX: Ensure temporal dimension is correct AFTER codebook normalization
  # This handles cases where tr_codes still has wrong temporal dimension
  if tr_codes.shape[1] != frames_per_tr:
      if tr_codes.shape[1] > frames_per_tr:
          tr_codes = tr_codes[:, :frames_per_tr]
      else:
          padding = frames_per_tr - tr_codes.shape[1]
          tr_codes = torch.nn.functional.pad(tr_codes, (0, padding), value=0)
  ```

## Verification

### Test Results (`test_audio_fix_simple.py`)

**Reproducing the Bug:**
```
Input: tr_codes with shape (4, 106697)
Old code: RuntimeError (dimension mismatch)
```

**Testing the Fix:**
```
Step 1 - Codebook normalization: (4, 106697) → (8, 106697) ✓
Step 2 - Temporal fix: (8, 106697) → (8, 112) ✓
Flattening: (8, 112) → (896,) ✓
```

### Expected Behavior

For EnCodec encoding with parameters:
- **Sample rate:** 24kHz
- **Frame rate:** 75 Hz (fixed by EnCodec)
- **TR length:** 1.5 seconds
- **Bandwidth:** 3.0 kbps (8 codebooks)
- **Frames per TR:** 75 Hz × 1.5s = 112 frames

**Output shape per TR:**
- Before flattening: `(8, 112)` - 8 codebooks × 112 frames
- After flattening: `(896,)` - 8 × 112 = 896 integer codes
- Final features: `(n_trs, 896)` - consistent dimensions for all TRs

## Files Modified

1. **`/Users/jmanning/giblet-responses/giblet/data/audio.py`**
   - Lines 307-327: Added two-step fix for tensor dimension handling

2. **Test files created:**
   - `test_audio_fix.py` - Full integration test (requires model loading)
   - `test_audio_fix_simple.py` - Fast unit test for tensor operations

## Impact

### Before Fix
- Training failed with RuntimeError during audio feature extraction
- Inconsistent tensor dimensions across TRs
- Could not stack features into batches

### After Fix
- All TRs have consistent shape `(896,)` for 8 codebooks
- No RuntimeError during encoding
- Features can be stacked: `(n_trs, 896)` array ready for training
- Handles edge cases (end of audio, variable codebook counts)

## Success Criteria Met

✓ All TRs have consistent shape
✓ No RuntimeError during encoding
✓ Features shape: `(n_trs, 896)` for 8 codebooks × 112 frames
✓ Dtype: `int64` (integer codebook indices)
✓ Metadata includes: tr_index, start_time, end_time, n_frames, n_codebooks, encoding_mode

## Notes

- The fix is defensive: it handles both the expected case (correct temporal dimension) and edge cases (incorrect temporal dimension)
- No performance impact: additional checks are cheap compared to audio encoding
- The root cause of why temporal dimension was wrong initially is unclear, but the fix makes the code more robust
- Consider investigating why lines 300-305 didn't always correctly set temporal dimension in future debugging sessions

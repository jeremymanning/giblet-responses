# Issue #28 EnCodec Fix Session Notes

**Date:** November 2, 2025
**Issue:** #28 - Systematic audio codec debugging
**Status:** RESOLVED ✅

## Summary

Successfully diagnosed and fixed critical EnCodec dimension bug that was blocking training.

## Bug Identified

**Location:** `giblet/data/audio.py:260`

**Problem:**
```python
# BUGGY CODE:
codes = encoded.audio_codes[0].cpu()  # Returns 3D tensor [1, n_codebooks, n_frames]
```

The code was only removing the list index `[0]` but not the batch dimension, resulting in:
- Expected shape: `[n_codebooks, n_frames]` (2D)
- Actual shape: `[1, n_codebooks, n_frames]` (3D)

When slicing `codes[:, start:end]`, it was selecting dimension 1 (size 4) instead of the frames dimension, resulting in shape `[1, 4, all_frames]` instead of `[n_codebooks, frames_in_window]`.

## Fix Applied

**Commit:** 8730dfe

```python
# FIXED CODE:
codes = encoded.audio_codes[0][0].cpu()  # Correctly returns 2D [n_codebooks, n_frames]
```

## Verification

### Unit Tests (Cluster - tensor01)
Tested with progressive TR counts using real Sherlock audio:
- ✅ 5 TRs: Shape (5, 896) - PASSED
- ✅ 10 TRs: Shape (10, 896) - PASSED
- ✅ 20 TRs: Shape (20, 896) - PASSED
- ✅ 50 TRs: Shape (50, 896) - PASSED
- ✅ 100 TRs: Shape (100, 896) - PASSED

All TRs have consistent dimensions: 896 codes = 8 codebooks × 112 frames/TR

### Integration Tests
✅ Dataset loading with EnCodec
✅ Feature caching (0.7 MB for full Sherlock)
✅ Training pipeline initialization
✅ Audio preprocessing in full training run

Test output from training run:
```
2. Processing audio...
  Computing EnCodec features...
  Cached 0.7 MB
  Audio features: (946, 896)  ✅
```

## New Issue Discovered

Training revealed **separate** issue (not EnCodec-related):
- **Issue #29:** Video encoder uses Conv2d which expects 4D input
- Video features from temporal concatenation are 2D: `[batch, 1641600]`
- Training crashes at first forward pass in video encoder

This is a video encoder architecture problem, NOT an EnCodec issue.

## Test Suite Created

**File:** `tests/data/test_audio_encodec_extended.py`
**Commit:** b021e49

Comprehensive tests with REAL Sherlock data (NO MOCKS):
- test_encodec_sherlock_5trs
- test_encodec_sherlock_100trs
- test_encodec_dimension_consistency
- test_encodec_quality_sherlock (with STOI metrics)
- test_encodec_round_trip_sherlock

## Commits

1. 8730dfe - Fix critical EnCodec dimension bug
2. b021e49 - Add comprehensive test suite

## Outcome

**Issue #28 Status:** RESOLVED

EnCodec component is fully fixed and working. Audio preprocessing verified on cluster with real Sherlock data up to 100 TRs.

**Next Steps:**
- Close Issue #28 (EnCodec fixed)
- Address Issue #29 (video encoder architecture)
- Optional: DAC evaluation (Path 2) and macOS mutex fix (Path 3) can be deferred

## Key Learnings

1. Always check tensor shapes at EVERY step when debugging dimension mismatches
2. Test directly on production environment (cluster) to avoid macOS-specific issues
3. Verify fixes with progressive scaling (5 → 10 → 20 → 50 → 100 TRs)
4. Monitor training closely - don't assume it works just because it launches

## Files Modified

- `giblet/data/audio.py` (1 line changed)
- `tests/data/test_audio_encodec_extended.py` (new file, 229 lines)

## Related Issues

- #26: Temporal concatenation (completed)
- #27: Integration testing (completed)
- #28: EnCodec debugging (THIS ISSUE - RESOLVED)
- #29: Video encoder architecture (NEW - created during this session)

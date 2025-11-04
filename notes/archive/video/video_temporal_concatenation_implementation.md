# Video Temporal Concatenation Implementation

**Date:** 2025-11-01
**Issue:** #26, Task 1.1
**Status:** ✓ Complete

## Summary

Successfully implemented video temporal concatenation for the multimodal fMRI autoencoder project. The implementation replaces the previous frame averaging approach with concatenation of all frames within each TR window [t-TR, t].

## Changes Made

### 1. Updated VideoProcessor Class

**File:** `/Users/jmanning/giblet-responses/giblet/data/video.py`

#### Key Changes:

1. **Module Documentation**
   - Updated docstrings to reflect temporal concatenation approach
   - Added clear explanation of window alignment [t-TR, t]

2. **video_to_features() Method**
   - **Previous behavior:** Averaged frames within each TR
   - **New behavior:** Concatenates all frames from temporal window
   - Features per TR: `frames_per_tr × height × width × 3`
   - For TR=1.5s @ 25fps: ~38 frames → 1,641,600 features per TR

3. **Edge Case Handling:**
   - **First TR:** Zero-pads frames before video start
   - **Last TR:** Zero-pads if incomplete window
   - **Consistency:** All TRs guaranteed same dimensions via padding

4. **features_to_video() Method**
   - Updated to handle concatenated temporal windows
   - Correctly reshapes features: `(n_trs, frames_per_tr, H, W, C)`
   - Writes all frames from each window sequentially

5. **Configurable TR Length**
   - TR length already configurable via `tr` parameter
   - Automatically calculates `frames_per_tr = round(fps × tr)`
   - Works with any TR length (tested: 1.0s, 1.5s, 2.0s)

### 2. Comprehensive Test Suite

**File:** `/Users/jmanning/giblet-responses/tests/data/test_video_temporal.py`

#### Tests Implemented (11 total, all passing):

**TestTemporalConcatenation:**
1. `test_consistent_dimensions_across_trs` - Verifies all TRs have same dimension
2. `test_different_tr_lengths` - Tests TR=1.0s, 1.5s, 2.0s
3. `test_first_tr_edge_case` - Validates first TR handling
4. `test_last_tr_edge_case` - Validates last TR handling
5. `test_zero_padding_behavior` - Confirms zero padding works
6. `test_temporal_window_alignment` - Verifies [t-TR, t] alignment
7. `test_feature_extraction_normalization` - Tests normalization
8. `test_max_trs_truncation` - Tests max_trs parameter
9. `test_reconstruction_roundtrip` - Tests video reconstruction

**TestDimensionConsistency:**
10. `test_all_trs_same_dimension` - Verifies dimension consistency
11. `test_dimension_calculation` - Validates dimension math

#### Test Results:
```
11 passed in 3.88s
```

### 3. Demo Script

**File:** `/Users/jmanning/giblet-responses/test_video_temporal_demo.py`

Demonstrates temporal concatenation with real Sherlock stimulus:
- Tests multiple TR lengths
- Shows dimension breakdown
- Verifies consistency
- Displays metadata

## Technical Specifications

### Dimension Calculations

For given parameters:
- `H` = target height (default: 90)
- `W` = target width (default: 160)
- `C` = channels (3 for RGB)
- `TR` = repetition time (default: 1.5s)
- `fps` = video frame rate (25fps for Sherlock)

Then:
- `frames_per_tr = round(fps × TR)`
- `features_per_frame = H × W × C`
- `features_per_tr = frames_per_tr × features_per_frame`

### Examples:

| TR (s) | FPS | Frames/TR | Features/Frame | Features/TR |
|--------|-----|-----------|----------------|-------------|
| 1.0    | 25  | 25        | 43,200         | 1,080,000   |
| 1.5    | 25  | 38        | 43,200         | 1,641,600   |
| 2.0    | 25  | 50        | 43,200         | 2,160,000   |

## Feature Matrix Shape

**Output:** `(n_trs, features_per_tr)` where:
- `n_trs`: Number of TRs in video
- `features_per_tr`: `frames_per_tr × H × W × C`

**Example for Sherlock (1426s @ 25fps, TR=1.5s):**
- Input video: 35,655 frames
- Output features: `(950, 1,641,600)`
- 950 TRs × 1,641,600 features per TR

## Temporal Window Alignment

Each TR window is `[t-TR, t]`:
- TR 0: [0.0s - 1.5s]
- TR 1: [1.5s - 3.0s]
- TR 2: [3.0s - 4.5s]
- ...

Frames are extracted from the window preceding each TR endpoint.

## Edge Case Handling

### First TR (Before Video Start)
```python
if frame_idx < 0:
    # Create zero-padded frame
    zero_frame = np.zeros((H, W, C), dtype=np.float32)
```

### Last TR (After Video End)
```python
elif frame_idx >= total_frames:
    # Create zero-padded frame
    zero_frame = np.zeros((H, W, C), dtype=np.float32)
```

### Incomplete Windows
```python
if len(tr_frames) < frames_per_tr:
    # Pad with zeros to reach frames_per_tr
    n_padding = frames_per_tr - len(tr_frames)
    for _ in range(n_padding):
        zero_frame = np.zeros((H, W, C), dtype=np.float32)
        tr_frames.append(zero_frame)
```

## Validation Results

### Demo Output (Sherlock Stimulus):

```
TR = 1.0s:
  Features per TR: 1,080,000
  Dimension consistency: ✓ PASS

TR = 1.5s:
  Features per TR: 1,641,600
  Dimension consistency: ✓ PASS

TR = 2.0s:
  Features per TR: 2,160,000
  Dimension consistency: ✓ PASS
```

### Test Output:
```
✓ All 3 TRs have consistent dimensions: 1641600 features
  (38 frames × 90 × 160 × 3 channels)
✓ TR=1.0s: 25 frames per TR, 5 TRs, 1080000 features per TR
✓ TR=1.5s: 38 frames per TR, 3 TRs, 1641600 features per TR
✓ TR=2.0s: 50 frames per TR, 2 TRs, 2160000 features per TR
✓ First TR properly handles edge case
✓ Last TR properly handles edge case
✓ Zero padding behavior verified (has_zero_frames: True)
✓ Temporal windows correctly aligned to [t-TR, t]
✓ Normalization working correctly
✓ max_trs truncation works correctly
✓ Reconstruction roundtrip successful
  Output: 160x90 @ 25.0fps, 76 frames
✓ All 3 TRs have identical dimensions: 1641600
✓ Dimension calculation correct:
  frames_per_tr: 38
  features_per_frame: 43200
  total_features: 1641600
```

## Success Criteria Met

- ✓ All TRs have same feature size
- ✓ No dimension mismatches
- ✓ TR length configurable
- ✓ Tests pass (11/11)
- ✓ Edge cases handled properly
- ✓ Documentation complete

## Usage Example

```python
from giblet.data.video import VideoProcessor

# Create processor with TR=1.5s
processor = VideoProcessor(
    target_height=90,
    target_width=160,
    tr=1.5,
    normalize=True
)

# Extract features with temporal concatenation
features, metadata = processor.video_to_features("video.mp4")

# features.shape: (n_trs, 1641600)
# Each TR contains 38 concatenated frames
```

## Next Steps

This implementation satisfies Task 1.1 of Issue #26. Subsequent tasks will build on this:
- Task 1.2: Audio temporal concatenation
- Task 1.3: Text temporal alignment
- Task 2.1: Update encoder/decoder for new dimensions

## Files Modified

1. `/Users/jmanning/giblet-responses/giblet/data/video.py` - Core implementation
2. `/Users/jmanning/giblet-responses/tests/data/test_video_temporal.py` - Test suite
3. `/Users/jmanning/giblet-responses/test_video_temporal_demo.py` - Demo script
4. `/Users/jmanning/giblet-responses/notes/video_temporal_concatenation_implementation.md` - This document

## References

- Issue #26: Temporal Concatenation for Multimodal Features
- PR #27: Video temporal concatenation (to be created)
- Original video.py: Used frame averaging within TR windows
- New video.py: Concatenates all frames within TR windows

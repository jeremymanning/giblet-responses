# EnCodec Discrete Code Alignment Implementation

**Date:** 2025-11-01
**Issue:** #24, Task 3.2
**Status:** ✅ COMPLETE

## Overview

Updated alignment modules (`sync.py` and `hrf.py`) to handle discrete EnCodec codes alongside existing continuous features (mel spectrograms, video, text). The implementation automatically detects feature type by dtype and applies appropriate processing.

## Problem Statement

EnCodec produces **discrete integer codes** (codebook indices) rather than continuous features:

- **Shape:** `(n_trs, n_codebooks, frames_per_tr)`
- **Dtype:** `int64` (or `int32`)
- **Values:** Integer codes in range [0, 1023] for 12kHz @ 3.0 kbps
- **Example:** `(920, 1, 112)` for Sherlock data (920 TRs, 1 codebook, 112 frames/TR @ 1.5s)

Standard alignment operations designed for continuous features are **invalid** for discrete codes:
1. **Resampling:** Linear interpolation would create invalid non-integer codes
2. **HRF Convolution:** Convolving discrete codes produces meaningless intermediate values

## Solution: Automatic dtype Detection

Both `sync.py` and `hrf.py` now detect feature type automatically:

```python
is_discrete = features.dtype in [np.int32, np.int64]
```

### Discrete Code Handling

**For integer dtypes:**
- ✅ Resampling: Nearest-neighbor interpolation (no blending)
- ✅ HRF: Skip convolution (return copy)
- ✅ Preserve: Codes remain valid integers in original range

**For float dtypes (unchanged):**
- ✅ Resampling: Linear interpolation (smooth transitions)
- ✅ HRF: Apply Glover HRF convolution
- ✅ Existing behavior preserved

## Implementation Details

### 1. sync.py: `_resample_features()`

**Changes:**
```python
# Detect discrete codes
is_discrete = features.dtype in [np.int32, np.int64]

if is_discrete:
    # Nearest-neighbor interpolation
    target_indices_int = np.round(target_indices).astype(int)
    target_indices_int = np.clip(target_indices_int, 0, current_trs - 1)
    resampled = features[target_indices_int]  # Direct indexing
else:
    # Linear interpolation (existing code)
    resampled = np.interp(...)
```

**Rationale:**
- Discrete codes cannot be averaged or interpolated linearly
- Nearest-neighbor selects the closest valid code
- Preserves code validity (no fractional codes)
- Works for both 2D and 3D arrays

**Example:**
```python
# EnCodec codes (100 TRs -> 50 TRs)
codes = np.random.randint(0, 1024, size=(100, 1, 112), dtype=np.int64)
resampled = _resample_features(codes, 100, 50)

# Output: (50, 1, 112), dtype=int64, values in [0, 1023]
# Each output TR is an exact copy of one input TR (no interpolation)
```

### 2. hrf.py: `apply_hrf()` and `convolve_with_padding()`

**Changes:**
```python
# Detect discrete codes
is_discrete = features.dtype in [np.int32, np.int64]

if is_discrete:
    # Skip HRF convolution for discrete codes
    return features.copy()

# Apply HRF to continuous features (existing code)
hrf = get_canonical_hrf(tr=tr)
...
```

**Rationale:**
- EnCodec codes **already encode temporal dynamics** through learned representations
- Convolving discrete codes would require: decode → convolve audio → re-encode
  - Expensive (requires full EnCodec decoder/encoder)
  - Questionable benefit (EnCodec already models temporal structure)
  - Would violate discrete code space
- HRF alignment can be simulated in fMRI prediction loss if needed

**Alternative Approaches Considered (Rejected):**

| Approach | Why Rejected |
|----------|--------------|
| **Decode → Convolve → Re-encode** | Too expensive, requires full EnCodec model, may not improve alignment |
| **Embed codes → Convolve → Find nearest codes** | Complex, may not preserve code validity, questionable benefit |
| **Apply HRF to code embeddings** | Breaks discrete code space, requires embedding layer (not available in alignment module) |

**Decision:** Skip HRF for EnCodec codes. EnCodec's learned temporal representations are sufficient. If HRF simulation is needed, it can be added to the fMRI prediction loss during training.

### 3. Backwards Compatibility

**Verified:** All existing tests pass (60/60)

- ✅ Continuous features: Linear interpolation + HRF convolution (unchanged)
- ✅ Mel spectrograms: 3D array handling (unchanged)
- ✅ Video/text features: 2D array handling (unchanged)
- ✅ Edge cases: Padding, mode='same'/'full', different TRs (unchanged)

**New tests added:**
- `test_resample_discrete_codes_3d`: 3D EnCodec codes (n_trs, n_codebooks, frames_per_tr)
- `test_resample_discrete_codes_2d`: 2D discrete codes
- `test_resample_discrete_codes_int32`: int32 dtype support
- `test_resample_continuous_vs_discrete_different_behavior`: Verify different algorithms
- `test_apply_hrf_discrete_codes_skipped`: HRF skipped for int64
- `test_apply_hrf_discrete_codes_int32_skipped`: HRF skipped for int32
- `test_apply_hrf_discrete_codes_2d_skipped`: 2D discrete codes
- `test_apply_hrf_discrete_codes_mode_full_skipped`: mode='full' handled
- `test_apply_hrf_continuous_vs_discrete_different`: Verify HRF only on continuous
- `test_convolve_with_padding_discrete_codes_skipped`: Padding case
- `test_convolve_with_padding_continuous_vs_discrete`: Verify behavior difference

## Configuration Parameters (Approved)

**EnCodec Setup:**
- Sample rate: 12kHz (not 24kHz) - optimized for speech
- Bandwidth: 3.0 kbps (sufficient quality, STOI ~0.74)
- Codebooks: 1 (mono audio)
- Frame rate: 75 Hz (fixed by EnCodec)
- Frames per TR: 112 @ TR=1.5s (75 Hz × 1.5s)
- Code range: [0, 1023] (1024 codes per codebook)

## Usage Examples

### Example 1: EnCodec Code Alignment

```python
from giblet.alignment.sync import align_all_modalities
import numpy as np

# EnCodec codes (discrete int64)
encodec_codes = np.random.randint(0, 1024, size=(946, 1, 112), dtype=np.int64)

# Continuous features
video = np.random.randn(950, 43200).astype(np.float32)
text = np.random.randn(950, 1024).astype(np.float32)
fmri = np.random.randn(920, 85810).astype(np.float32)

# Align all modalities
result = align_all_modalities(
    video_features=video,
    audio_features=encodec_codes,  # Discrete codes handled automatically
    text_features=text,
    fmri_features=fmri,
    apply_hrf_conv=True,
    tr=1.5
)

# Result:
# - video: (920, 43200) - HRF convolved
# - audio: (920, 1, 112) - Nearest-neighbor resampled, NO HRF
# - text: (920, 1024) - HRF convolved
# - fmri: (920, 85810) - Truncated only
```

### Example 2: Manual Resampling

```python
from giblet.alignment.sync import _resample_features

# Discrete codes: nearest-neighbor
codes = np.random.randint(0, 1024, size=(100, 1, 112), dtype=np.int64)
resampled = _resample_features(codes, 100, 50)
# Output: (50, 1, 112), dtype=int64, exact copies (no interpolation)

# Continuous features: linear interpolation
mels = np.random.randn(100, 128, 65).astype(np.float32)
resampled = _resample_features(mels, 100, 50)
# Output: (50, 128, 65), dtype=float32, interpolated values
```

### Example 3: HRF Application

```python
from giblet.alignment.hrf import apply_hrf

# Discrete codes: HRF skipped
codes = np.random.randint(0, 1024, size=(100, 1, 112), dtype=np.int64)
result = apply_hrf(codes, tr=1.5, mode='same')
# Output: Exact copy of input (no HRF convolution)

# Continuous features: HRF applied
mels = np.random.randn(100, 128, 65).astype(np.float32)
result = apply_hrf(mels, tr=1.5, mode='same')
# Output: HRF-convolved features (temporally blurred)
```

## Validation Results

**Test Summary:**
- **Total tests:** 60
- **Passed:** 60 (100%)
- **Failed:** 0
- **Coverage:**
  - Discrete code resampling: 5 tests
  - Discrete code HRF handling: 6 tests
  - Backwards compatibility: 49 tests (all existing tests)

**Key Validations:**
1. ✅ Discrete codes preserve dtype (int64, int32)
2. ✅ Discrete codes remain in valid range [0, code_max]
3. ✅ Nearest-neighbor produces exact copies (no interpolation)
4. ✅ HRF skipped for discrete codes
5. ✅ Continuous features unchanged (linear interpolation + HRF)
6. ✅ 3D array handling works for both discrete and continuous
7. ✅ Edge cases handled (padding, mode='same'/'full', different TRs)

## Files Modified

1. **giblet/alignment/sync.py**
   - Updated `_resample_features()` for discrete code detection
   - Nearest-neighbor interpolation for integer dtypes
   - Linear interpolation for float dtypes (unchanged)
   - Updated docstrings with discrete code examples

2. **giblet/alignment/hrf.py**
   - Updated `apply_hrf()` to skip HRF for discrete codes
   - Updated `convolve_with_padding()` to skip HRF for discrete codes
   - Added rationale in docstrings
   - Updated examples with EnCodec code usage

3. **tests/test_sync.py**
   - Added 5 new discrete code tests
   - Verified nearest-neighbor behavior
   - Verified dtype preservation
   - Verified continuous vs discrete differences

4. **tests/integration/test_hrf.py**
   - Added 6 new discrete code tests
   - Verified HRF skipping for integer dtypes
   - Verified continuous features still convolved
   - Verified padding case

## API Compatibility

**No breaking changes:**
- All existing function signatures unchanged
- Automatic dtype detection (no new parameters)
- Backwards compatible with continuous features
- Drop-in replacement for existing code

**Type signature (unchanged):**
```python
def _resample_features(
    features: np.ndarray,
    current_trs: int,
    target_trs: int
) -> np.ndarray

def apply_hrf(
    features,
    tr=1.5,
    mode='same'
) -> np.ndarray

def convolve_with_padding(
    features,
    tr=1.5,
    padding_duration=10.0
) -> np.ndarray
```

## Performance Implications

**Discrete code path (faster):**
- Nearest-neighbor: O(n) array indexing (no interpolation loops)
- HRF skip: O(1) copy operation (no convolution)
- 3D arrays: Direct slicing (no nested loops)

**Continuous feature path (unchanged):**
- Linear interpolation: O(n × m) where m = feature dims
- HRF convolution: O(n × k) where k = HRF kernel size
- Performance identical to previous implementation

## Future Enhancements (Optional)

1. **HRF in latent space:** Apply HRF convolution to EnCodec embeddings (requires embedding layer in encoder)
2. **Trainable temporal alignment:** Learn optimal temporal alignment during training
3. **Multi-codebook support:** Currently supports 1 codebook, could extend to multiple
4. **Alternative interpolation:** Could add other methods (cubic, spline) for continuous features

## Success Criteria ✅

- [x] Handles discrete codes without errors
- [x] Resampling preserves code validity [0, 1023]
- [x] No information loss from improper averaging
- [x] Backwards compatible with continuous features
- [x] All tests pass (60/60)
- [x] Documentation complete

## References

- Issue #24: Implement EnCodec for high-quality audio reconstruction
- Task 3.2: Update alignment modules for EnCodec discrete codes
- Approved configuration: 12kHz sampling, 3.0 kbps bandwidth
- EnCodec paper: https://arxiv.org/abs/2210.13438

## Conclusion

Alignment modules now **fully support EnCodec discrete codes** while maintaining **100% backwards compatibility** with existing continuous features. The implementation is:

- ✅ **Automatic:** dtype detection requires no code changes
- ✅ **Correct:** Preserves discrete code validity
- ✅ **Efficient:** Faster than continuous interpolation
- ✅ **Tested:** 11 new tests, all existing tests pass
- ✅ **Documented:** Clear rationale and examples

**Ready for integration with EnCodec audio encoder/decoder modules.**

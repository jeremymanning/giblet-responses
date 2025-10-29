# Temporal Synchronization Module Usage Guide

## Overview

The temporal synchronization module (`giblet.alignment.sync`) aligns all stimulus modalities (video, audio, text) to a common temporal grid based on fMRI TRs and applies HRF convolution to stimulus features.

## Quick Start

```python
from giblet.alignment.sync import align_all_modalities

# Align all modalities and apply HRF convolution
result = align_all_modalities(
    video_features=video_data,      # (950, 43200)
    audio_features=audio_data,      # (946, 128)
    text_features=text_data,        # (950, 1024)
    fmri_features=fmri_data,        # (920, 85810)
    apply_hrf_conv=True,            # Apply HRF to stimulus features
    tr=1.5                          # fMRI repetition time (seconds)
)

# All outputs have shape (920, n_features)
video_aligned = result['video']     # (920, 43200)
audio_aligned = result['audio']     # (920, 128)
text_aligned = result['text']       # (920, 1024)
fmri_aligned = result['fmri']       # (920, 85810)
```

## API Reference

### `align_all_modalities()`

Main function for aligning all modalities to a common temporal grid.

**Parameters:**
- `video_features` (ndarray): Video features, shape (n_video_trs, n_video_features)
- `audio_features` (ndarray): Audio features, shape (n_audio_trs, n_audio_features)
- `text_features` (ndarray): Text embeddings, shape (n_text_trs, n_text_features)
- `fmri_features` (ndarray): fMRI voxel timeseries, shape (n_fmri_trs, n_fmri_features)
- `apply_hrf_conv` (bool, default=True): Whether to apply HRF convolution to stimulus features
- `tr` (float, default=1.5): Repetition time in seconds
- `hrf_padding_duration` (float, default=10.0): Padding duration for HRF convolution (seconds)

**Returns:**
Dictionary with keys:
- `'video'`: Aligned video features, shape (target_trs, n_video_features)
- `'audio'`: Aligned audio features, shape (target_trs, n_audio_features)
- `'text'`: Aligned text features, shape (target_trs, n_text_features)
- `'fmri'`: Truncated fMRI features, shape (target_trs, n_fmri_features)
- `'n_trs'`: Target number of TRs (int)
- `'video_orig_trs'`: Original video TR count
- `'audio_orig_trs'`: Original audio TR count
- `'text_orig_trs'`: Original text TR count
- `'fmri_orig_trs'`: Original fMRI TR count

### `get_alignment_info()`

Helper function to get summary statistics about an alignment result.

**Parameters:**
- `alignment_result` (dict): Result dictionary from `align_all_modalities()`

**Returns:**
Dictionary with alignment statistics including:
- `'n_trs'`: Aligned number of TRs
- `'video_features'`: Number of video features
- `'audio_features'`: Number of audio features
- `'text_features'`: Number of text features
- `'fmri_features'`: Number of fMRI features
- Original TR counts for each modality

## Alignment Strategy

### Temporal Synchronization
1. Determines the **minimum** number of TRs across all modalities
2. Resamples stimulus features (video, audio, text) to match this minimum
3. Truncates fMRI to the minimum number of TRs
4. Uses linear interpolation for resampling to preserve feature continuity

For Sherlock dataset:
- Video: 950 TRs → 920 TRs
- Audio: 946 TRs → 920 TRs
- Text: 950 TRs → 920 TRs
- fMRI: 920 TRs (reference)
- **Total duration**: 920 TRs × 1.5s/TR = 1380s ≈ 23 minutes

### HRF Convolution
When `apply_hrf_conv=True`:
- **Only stimulus features** (video, audio, text) are convolved with HRF
- **fMRI is NOT convolved** (already observed BOLD response)
- Uses Glover HRF model from nilearn
- Applies padding to minimize edge effects

This predicts what the BOLD response would be given the stimulus, which can be compared against observed fMRI.

## Example: Full Workflow

```python
import numpy as np
from giblet.data.video import VideoProcessor
from giblet.data.audio import AudioProcessor
from giblet.data.text import TextProcessor
from giblet.data.fmri import FMRIProcessor
from giblet.alignment.sync import align_all_modalities, get_alignment_info

# Load data from files
video_proc = VideoProcessor()
audio_proc = AudioProcessor()
text_proc = TextProcessor()
fmri_proc = FMRIProcessor(max_trs=920)

# Extract features
video_feat, _ = video_proc.video_to_features('path/to/video.mp4')
audio_feat, _ = audio_proc.audio_to_features('path/to/video.mp4')
text_feat, _ = text_proc.annotations_to_embeddings('path/to/annotations.xlsx', n_trs=950)
fmri_feat, coords, _ = fmri_proc.nii_to_features('path/to/fmri.nii.gz')

# Align all modalities
aligned = align_all_modalities(
    video_features=video_feat,
    audio_features=audio_feat,
    text_features=text_feat,
    fmri_features=fmri_feat,
    apply_hrf_conv=True
)

# Get summary info
info = get_alignment_info(aligned)
print(f"Aligned to {info['n_trs']} TRs (~{info['n_trs'] * 1.5 / 60:.1f} minutes)")

# Access aligned features
video = aligned['video']
audio = aligned['audio']
text = aligned['text']
fmri = aligned['fmri']

# All have shape (920, n_features)
# Ready for autoencoder training!
```

## Implementation Details

### Feature Resampling
- Uses `np.interp()` for 1D linear interpolation
- Applied independently to each feature dimension
- Preserves dtype (float32, etc.)

### HRF Application
- Uses `convolve_with_padding()` from `giblet.alignment.hrf`
- Adds 10 seconds of zero padding before/after signal
- Performs full convolution then trims to original size
- Eliminates edge artifacts

### Quality Checks
The module includes automatic verification that:
- All outputs have the target number of TRs
- Feature dimensions are preserved
- All values are finite (no NaN or Inf)
- Metadata is accurate

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_sync.py -v
```

Tests cover:
- Basic alignment without/with HRF
- Metadata preservation
- Output shape validation
- Minimum TR selection
- HRF convolution effects
- Edge cases (already aligned, single feature, etc.)

## Performance Notes

- Typical alignment of Sherlock dataset: ~1-2 seconds
- Memory requirement: ~3-5 GB for full 950×43200 video features
- HRF convolution adds ~5-10 seconds per modality
- Linear interpolation is efficient for downsampling/upsampling

## See Also

- `giblet.alignment.hrf`: HRF convolution utilities
- `giblet.data.video`: Video feature extraction
- `giblet.data.audio`: Audio feature extraction
- `giblet.data.text`: Text embedding extraction
- `giblet.data.fmri`: fMRI feature extraction

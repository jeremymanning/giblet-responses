# giblet/alignment - Temporal Alignment and HRF Convolution

This module handles temporal synchronization of all stimulus modalities (video, audio, text) to the fMRI temporal grid, and applies Hemodynamic Response Function (HRF) convolution to predict BOLD responses.

## Overview

All multimodal data must be aligned to a common temporal grid based on fMRI **repetition time (TR)**, which is 1.5 seconds by default. This module provides two key capabilities:

1. **Temporal Synchronization** ([sync.py](sync.py)): Align video/audio/text features to fMRI TRs
2. **HRF Convolution** ([hrf.py](hrf.py)): Convolve stimulus features with canonical HRF to predict BOLD

```
Raw Stimulus Features          Aligned to TRs           HRF Convolved
├─ Video (25 fps)          →   Video (920 TRs)     →   BOLD prediction
├─ Audio (24 kHz)          →   Audio (920 TRs)     →   BOLD prediction
└─ Text (variable timing)  →   Text (920 TRs)      →   BOLD prediction
```

## Components

### HRF Convolution ([hrf.py](hrf.py))

Provides canonical Hemodynamic Response Function for converting stimulus features to predicted BOLD responses.

**Key Functions:**

#### get_canonical_hrf()

Get the canonical HRF kernel using the Glover (1999) model.

```python
from giblet.alignment.hrf import get_canonical_hrf

# Get HRF kernel
hrf = get_canonical_hrf(tr=1.5, duration=32.0)

# HRF properties:
# - Peak at ~5-6 seconds post-stimulus
# - Returns to baseline ~15-30 seconds
# - Normalized to peak of 1.0
# - Shape: (n_samples,) where n_samples = ceil(duration / tr)

print(hrf.shape)  # (21,) for tr=1.5, duration=32
```

**Parameters:**
- `tr` (float): Repetition time in seconds (default: 1.5)
- `duration` (float): Duration of HRF kernel in seconds (default: 32.0)

**Returns:**
- `hrf` (np.ndarray): HRF kernel normalized to peak=1.0

---

#### apply_hrf()

Convolve stimulus features with HRF to predict BOLD response.

```python
from giblet.alignment.hrf import apply_hrf
import numpy as np

# Stimulus features: (n_trs, n_features)
stimulus = np.random.randn(920, 1024)

# Apply HRF convolution
bold_prediction = apply_hrf(stimulus, tr=1.5)

# Output: (920, 1024) - Same shape as input
# Features are now temporally smoothed and delayed to match BOLD timing
```

**Parameters:**
- `features` (np.ndarray): Stimulus features with shape (n_trs, n_features)
- `tr` (float): Repetition time in seconds (default: 1.5)
- `duration` (float): HRF kernel duration (default: 32.0)

**Returns:**
- `convolved` (np.ndarray): HRF-convolved features with shape (n_trs, n_features)

**What it does:**
1. Convolves each feature independently with the canonical HRF
2. Introduces temporal smoothing and 5-6 second delay (peak BOLD response)
3. Normalizes to preserve feature magnitudes
4. Handles edge effects with zero-padding

---

#### convolve_with_padding()

Low-level HRF convolution with explicit padding control.

```python
from giblet.alignment.hrf import convolve_with_padding, get_canonical_hrf

hrf = get_canonical_hrf(tr=1.5)
features = np.random.randn(920, 1024)

# Convolve with custom padding
result = convolve_with_padding(
    features=features,
    hrf=hrf,
    mode='same'  # Return same size as input
)
```

**Parameters:**
- `features` (np.ndarray): Input features (n_trs, n_features)
- `hrf` (np.ndarray): HRF kernel (n_samples,)
- `mode` (str): Padding mode ('same', 'valid', 'full')

**Returns:**
- `convolved` (np.ndarray): Convolved features

---

### Temporal Synchronization ([sync.py](sync.py))

Aligns all modalities to common TR grid and applies HRF convolution.

**Key Functions:**

#### align_all_modalities()

Main function to align video, audio, text, and fMRI to common temporal grid.

```python
from giblet.alignment.sync import align_all_modalities

# Raw features from processors
video_features = video_processor.extract_features('video.m4v')      # (920, 259200)
audio_features = audio_processor.extract_features('video.m4v')      # (920, 128)
text_features = text_processor.extract_features('annotations.xlsx') # (920, 1024)
fmri_data = fmri_processor.extract_features(['sub-01.nii.gz'])     # (920, 85810)

# Align all modalities
aligned = align_all_modalities(
    video=video_features,
    audio=audio_features,
    text=text_features,
    fmri=fmri_data,
    apply_hrf=True,  # Convolve stimuli with HRF
    tr=1.5
)

# Returns dictionary:
# {
#     'video': (n_trs, 259200),     # HRF-convolved if apply_hrf=True
#     'audio': (n_trs, 128),          # HRF-convolved if apply_hrf=True
#     'text': (n_trs, 1024),          # HRF-convolved if apply_hrf=True
#     'fmri': (n_trs, 85810),         # Always unmodified
#     'n_trs': int                    # Minimum TRs across modalities
# }
```

**Parameters:**
- `video` (np.ndarray): Video features (n_trs, video_dim)
- `audio` (np.ndarray): Audio features (n_trs, audio_dim)
- `text` (np.ndarray): Text features (n_trs, text_dim)
- `fmri` (np.ndarray): fMRI data (n_trs, n_voxels)
- `apply_hrf` (bool): Whether to convolve stimuli with HRF (default: True)
- `tr` (float): Repetition time in seconds (default: 1.5)

**Returns:**
- `aligned` (dict): Dictionary with aligned features

**What it does:**
1. Finds minimum number of TRs across all modalities
2. Truncates all features to minimum TR count
3. Optionally convolves stimulus features (video/audio/text) with HRF
4. Returns all features aligned to common temporal grid

**Note**: fMRI data is never convolved (it's already the BOLD response)

---

#### _resample_features()

Internal function to resample features to target number of TRs.

```python
from giblet.alignment.sync import _resample_features

# Resample from 900 TRs to 920 TRs
features = np.random.randn(900, 1024)
resampled = _resample_features(features, current_trs=900, target_trs=920)

print(resampled.shape)  # (920, 1024)
```

**Parameters:**
- `features` (np.ndarray): Input features (current_trs, n_features)
- `current_trs` (int): Current number of TRs
- `target_trs` (int): Target number of TRs

**Returns:**
- `resampled` (np.ndarray): Resampled features (target_trs, n_features)

**Interpolation:**
- **Float dtypes**: Linear interpolation (smooth transitions)
- **Integer dtypes**: Nearest-neighbor interpolation (preserves discrete codes)

**Use case**: Handles slight mismatches in TR counts due to rounding or truncation.

---

## HRF Theory

### What is the HRF?

The **Hemodynamic Response Function (HRF)** describes how the brain's BOLD signal responds to neural activity:

```
Neural Activity (stimulus)  →  HRF Convolution  →  BOLD Signal (fMRI)
```

**Key Properties:**
- **Delay**: BOLD response peaks 5-6 seconds after stimulus
- **Duration**: Returns to baseline after 15-30 seconds
- **Shape**: Positive peak followed by small undershoot
- **Non-linearity**: Approximately linear for typical stimuli

### Glover HRF Model

This module uses the **Glover (1999)** canonical HRF model:

```python
import matplotlib.pyplot as plt
from giblet.alignment.hrf import get_canonical_hrf

hrf = get_canonical_hrf(tr=1.5, duration=32.0)
time = np.arange(len(hrf)) * 1.5

plt.plot(time, hrf)
plt.xlabel('Time (seconds)')
plt.ylabel('HRF amplitude')
plt.title('Canonical Glover HRF')
plt.axvline(5, color='r', linestyle='--', label='Peak at ~5s')
plt.legend()
plt.show()
```

**Model parameters** (from nilearn implementation):
- Peak time: ~5-6 seconds
- Undershoot delay: ~15 seconds
- Normalization: Peak = 1.0

### Why Convolve with HRF?

When predicting brain responses from stimuli, we must account for the HRF:

**Without HRF convolution:**
```
Stimulus at t=0  →  Model predicts immediate response  ❌ Wrong
```

**With HRF convolution:**
```
Stimulus at t=0  →  HRF convolution  →  Model predicts peak response at t=5s  ✅ Correct
```

**In practice:**
```python
# Option 1: Direct stimulus features (no HRF)
dataset = MultimodalDataset(apply_hrf=False)
# Use when: Model learns HRF internally or using raw stimulus timing

# Option 2: HRF-convolved features (recommended)
dataset = MultimodalDataset(apply_hrf=True)
# Use when: Model should predict BOLD directly from stimuli
```

---

## Temporal Alignment Strategy

### Alignment Process

All modalities start with different temporal resolutions:

| Modality | Native Resolution | After Processing |
|----------|-------------------|------------------|
| **Video** | 25 fps (40 ms/frame) | 920 TRs (1.5s bins) |
| **Audio** | 24 kHz (0.04 ms/sample) | 920 TRs (1.5s bins) |
| **Text** | Variable timing | 920 TRs (1.5s bins) |
| **fMRI** | 1.5s TR (native) | 920 TRs (1.5s bins) |

**Alignment steps:**
1. **Extract features** at native resolution (done by data processors)
2. **Aggregate to TRs** using temporal concatenation or averaging
3. **Truncate to minimum TRs** across all modalities
4. **Apply HRF** to stimulus features (optional)

### TR Binning Example

For TR=1.5s with 25 fps video:

```
Video frames:         F0 F1 F2 ... F37 | F38 F39 ... F75 | F76 F77 ...
                      └─── TR 0 ────┘   └─── TR 1 ────┘   └─── TR 2 ───
fMRI TRs:             [TR 0: t=0-1.5s] [TR 1: t=1.5-3.0s] [TR 2: t=3.0-4.5s]
```

**Each TR contains:**
- Video: ~37 frames concatenated (with `frame_skip=2`: ~18 frames)
- Audio: 36,000 samples encoded to 128-dim (EnCodec)
- Text: All embeddings from segments overlapping [t-TR, t]

---

## Usage Examples

### Basic HRF Convolution

```python
import numpy as np
from giblet.alignment.hrf import apply_hrf

# Simulate stimulus features (920 TRs, 1024 features)
stimulus = np.random.randn(920, 1024)

# Apply HRF to predict BOLD
bold_prediction = apply_hrf(stimulus, tr=1.5)

# Visualize single feature
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(stimulus[:100, 0], label='Raw stimulus')
plt.xlabel('TR')
plt.ylabel('Amplitude')
plt.title('Original stimulus feature')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bold_prediction[:100, 0], label='HRF-convolved')
plt.xlabel('TR')
plt.ylabel('Amplitude')
plt.title('Predicted BOLD (after HRF)')
plt.legend()
plt.tight_layout()
plt.show()
```

### Complete Multimodal Alignment

```python
from giblet.data import VideoProcessor, AudioProcessor, TextProcessor, FMRIProcessor
from giblet.alignment.sync import align_all_modalities

# Initialize processors
video_proc = VideoProcessor(tr=1.5, frame_skip=2)
audio_proc = AudioProcessor(tr=1.5, use_encodec=True)
text_proc = TextProcessor(tr=1.5)
fmri_proc = FMRIProcessor(tr=1.5, max_trs=920)

# Extract features
video = video_proc.extract_features('data/stimuli_Sherlock.m4v')
audio = audio_proc.extract_features('data/stimuli_Sherlock.m4v')
text = text_proc.extract_features('data/annotations.xlsx')
fmri = fmri_proc.extract_features(['data/sherlock_nii/sub-01.nii.gz'])

# Align all modalities with HRF
aligned = align_all_modalities(
    video=video,
    audio=audio,
    text=text,
    fmri=fmri,
    apply_hrf=True,
    tr=1.5
)

# All features now aligned to 920 TRs
print("Video:", aligned['video'].shape)   # (920, 259200) - HRF-convolved
print("Audio:", aligned['audio'].shape)   # (920, 128) - HRF-convolved
print("Text:", aligned['text'].shape)     # (920, 1024) - HRF-convolved
print("fMRI:", aligned['fmri'].shape)     # (920, 85810) - Unmodified
```

### Training with/without HRF

```python
from giblet.data import MultimodalDataset

# Option 1: HRF-convolved features (recommended)
dataset_hrf = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    apply_hrf=True  # Stimulus features convolved with HRF
)

# Model predicts fMRI directly from HRF-convolved stimuli
# Loss: MSE(predicted_fmri, actual_fmri)

# Option 2: Raw stimulus features
dataset_raw = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    apply_hrf=False  # Raw stimulus features
)

# Model must learn HRF internally
# More flexible but harder to train
```

---

## Technical Details

### HRF Convolution Mathematics

Given stimulus features $S(t)$ and HRF kernel $h(t)$:

$$
\text{BOLD}(t) = (S * h)(t) = \int S(\tau) \cdot h(t - \tau) d\tau
$$

**Discrete implementation:**
```python
# For each feature independently:
for i in range(n_features):
    bold_prediction[:, i] = np.convolve(stimulus[:, i], hrf, mode='same')
```

**Normalization:**
- HRF kernel normalized to peak=1.0
- Convolution preserves feature magnitudes
- No additional scaling needed

### Edge Effects

**Problem**: Convolution at boundaries requires padding

**Solution**: Zero-padding with `mode='same'`

**Impact**:
- First ~6 TRs: Incomplete HRF influence (ramp-up)
- Last ~6 TRs: Incomplete HRF influence (ramp-down)
- Middle TRs: Accurate HRF convolution

**Mitigation**: Use train/val split to avoid edge TRs in evaluation

### Resampling Strategy

**Linear interpolation** (float features):
```python
from scipy.interpolate import interp1d
f = interp1d(old_indices, features, axis=0, kind='linear')
resampled = f(new_indices)
```

**Nearest-neighbor** (integer codes):
```python
from scipy.interpolate import interp1d
f = interp1d(old_indices, codes, axis=0, kind='nearest')
resampled = f(new_indices).astype(int)
```

---

## References

- **Glover (1999)**: "Deconvolution of impulse response in event-related BOLD fMRI." *NeuroImage*, 9(4), 416-429.
- **nilearn HRF implementation**: https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.glover_hrf.html

---

## Related Modules

- **[giblet/data/](../data/)** - Data processors that produce features to be aligned
- **[giblet/models/](../models/)** - Models that consume aligned features

For questions or issues, see the main project [README.md](../../README.md).

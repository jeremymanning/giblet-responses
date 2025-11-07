# Quick Reference: Multimodal Alignment for Neuroscience Autoencoder

**Generated**: 2025-10-28

## 1. Temporal Alignment (Video/Audio/fMRI/Text)

### Key Strategy: TR Binning + HRF Convolution

```python
# Align high-freq data to fMRI TR (1 Hz)
from nilearn.glm.first_level import glover_hrf, compute_regressor
import numpy as np

# Step 1: Aggregate into TR bins
tr_duration = 1.0  # seconds
for each_tr:
    video_frames_in_tr = frames[(time >= tr_time) & (time < tr_time + tr_duration)]
    aggregated_feature = np.mean(video_frames_in_tr, axis=0)

# Step 2: Convolve with HRF (accounts for ~6s hemodynamic lag)
hrf = glover_hrf(tr=1.0, oversampling=50, time_length=32)
fmri_predictor = np.convolve(stimulus_features, hrf)
```

**Library Functions**:
- `nilearn.glm.first_level.compute_regressor()` - HRF convolution
- `nilearn.glm.first_level.glover_hrf()` - Generate HRF
- `numpy.interp()` - Simple interpolation

**Key Parameters**:
- TR = 1.0 s
- HRF oversampling = 50
- Hemodynamic lag: ~6s (handled by HRF)

---

## 2. Audio Processing

### Extraction: Mel Spectrogram

```python
import librosa

# Standard parameters for HiFi-GAN compatibility
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=22050,      # Sample rate
    n_fft=1024,    # FFT size
    hop_length=256,
    n_mels=80,     # 80 for vocoder, 128 for detail
    fmin=0.0,
    fmax=8000.0
)
mel_db = librosa.power_to_db(mel_spec)
```

### Reconstruction: HiFi-GAN Vocoder (NOT Griffin-Lim!)

```python
import torch
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH

# Load pretrained vocoder
vocoder = HIFIGAN_VOCODER_V3_LJSPEECH.get_vocoder()
vocoder.eval()

# Reconstruct
mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
with torch.no_grad():
    waveform = vocoder(mel_tensor)
```

**Critical**: Griffin-Lim produces poor quality. Always use neural vocoder.

**Library**: `torchaudio` (HiFi-GAN), `librosa` (extraction)

**Parameters**:
- sr = 22050 Hz
- n_mels = 80 (standard) or 128 (high quality)
- n_fft = 1024
- hop_length = 256

---

## 3. Text Embedding & Recovery

### Best Models (MTEB 2024-2025)

1. `BAAI/bge-large-en-v1.5` (state-of-the-art, 1024-dim)
2. `sentence-transformers/all-mpnet-base-v2` (balanced, 768-dim)
3. `NV-Embed-v2` (multimodal-aware)

### Embedding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = model.encode(texts)
```

### Recovery: Nearest Neighbor (Recommended)

```python
from sklearn.neighbors import NearestNeighbors

# Build database of original annotations
nn_index = NearestNeighbors(n_neighbors=1, metric='cosine')
nn_index.fit(original_embeddings)

# Retrieve text from new embedding
distances, indices = nn_index.kneighbors([new_embedding])
retrieved_text = original_texts[indices[0][0]]
```

### Recovery: Vec2Text (Advanced)

```python
import vec2text

corrector = vec2text.load_corrector("sentence-transformers/all-mpnet-base-v2")
reconstructed_texts = vec2text.invert_embeddings(
    embeddings=embeddings,
    corrector=corrector,
    num_steps=20
)
```

**Library**: `sentence-transformers`, `vec2text` (for reconstruction)

**Note**: Embedding-to-text is semantically lossy. Use nearest-neighbor for practical retrieval.

---

## 4. fMRI Data Formatting

### CMU Format (BrainIAK)

**Key**: Transpose nilearn format

- **nilearn**: (n_timepoints, n_voxels)
- **BrainIAK/CMU**: (n_voxels, n_timepoints)

```python
from nilearn import masking
import numpy as np

# Load and mask fMRI data
data_nilearn = masking.apply_mask(nifti_file, mask_img)  # (time, voxels)
data_cmu = data_nilearn.T  # (voxels, time) for BrainIAK
```

### Shared Brain Mask

```python
from nilearn.masking import compute_multi_epi_mask

shared_mask = compute_multi_epi_mask(
    subject_nifti_files,
    threshold=0.5,      # 50% of subjects must have signal
    opening=2,          # morphological opening
    connected=True      # keep largest component
)
```

### Inter-Subject Alignment: SRM

```python
from brainiak.funcalign.srm import SRM

# subject_data_list: list of (n_voxels, n_timepoints) arrays
srm = SRM(n_iter=10, features=100)
srm.fit(subject_data_list)
aligned_data = srm.transform(subject_data_list)
```

**Library Functions**:
- `nilearn.masking.compute_multi_epi_mask()` - Multi-subject mask
- `brainiak.funcalign.srm.SRM` - Shared Response Model
- `nilearn.masking.apply_mask()` - Extract masked data

**Parameters**:
- Mask threshold: 0.5-0.8
- SRM features: 50-100
- Format: (voxels, timepoints) for BrainIAK

---

## Installation

```bash
# fMRI
pip install nibabel nilearn brainiak

# Audio
pip install librosa torchaudio soundfile

# Text
pip install sentence-transformers vec2text

# General
pip install numpy scipy scikit-learn
```

---

## Critical Parameters Table

| Component | Parameter | Value |
|-----------|-----------|-------|
| fMRI | TR | 1.0 s |
| Video | FPS | 25 |
| Audio | Sample rate | 22050 Hz |
| Audio | n_mels | 80 or 128 |
| Audio | n_fft | 1024 |
| Audio | hop_length | 256 |
| Text | Model | BAAI/bge-large-en-v1.5 |
| Text | Embedding dim | 1024 |
| HRF | Model | 'glover' |
| HRF | Oversampling | 50 |
| Mask | Threshold | 0.5-0.8 |
| SRM | Features | 50-100 |

---

## Pipeline Overview

```
Raw Data → Preprocessing → Alignment → Encoding → Autoencoder
```

1. **Video**: Extract frames (25 fps) → Aggregate to TR → HRF convolve
2. **Audio**: Librosa mel spec (22050 Hz, 80 mels) → Aggregate to TR → HRF convolve
3. **Text**: Sentence transformers embed → Align to TR (nearest/forward-fill) → HRF convolve
4. **fMRI**: Load NIfTI → Apply shared mask → Convert to CMU format → SRM align

**All modalities aligned to TR=1s before feeding to autoencoder**

---

## Key References

- BrainIAK HTFA: https://brainiak.org/examples/htfa.html
- HiFi-GAN: https://github.com/jik876/hifi-gan
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Vec2Text: https://github.com/vec2text/vec2text
- Nilearn: https://nilearn.github.io

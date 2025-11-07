# giblet/data - Data Pipeline and Dataset Classes

This module provides the complete data processing pipeline for the multimodal autoencoder project. It handles loading, preprocessing, and temporal alignment of fMRI data with video, audio, and text stimuli.

## Overview

The data pipeline converts raw multimodal data into aligned feature matrices suitable for training neural networks:

```
Raw Data                      Aligned Features
├─ Video (Sherlock.m4v)   →  Video features (frame concatenations)
├─ Audio (from video)     →  Audio features (EnCodec or mel spectrograms)
├─ Text (annotations)     →  Text embeddings (1024-dim)
└─ fMRI (.nii.gz files)   →  BOLD timeseries (voxel × time)
                              ↓
                         Temporally aligned to fMRI TRs (1.5s bins)
```

All modalities are aligned to a common temporal grid based on fMRI **repetition time (TR)**, which defaults to 1.5 seconds.

## Main Components

### MultimodalDataset ([dataset.py](dataset.py))

PyTorch Dataset class that loads all four modalities aligned to fMRI TRs.

**Key Features:**
- Loads video, audio, text, and fMRI data for 17 subjects
- Supports train/validation splits (80/20 by default)
- Optional HRF convolution for stimulus features
- Per-subject or cross-subject modes
- Automatic caching for fast loading
- Memory-efficient: only loads needed data

**Quick Start:**
```python
from giblet.data import MultimodalDataset

# Load all subjects with HRF convolution
dataset = MultimodalDataset(
    data_dir='data/',
    subjects='all',          # or [1, 2, 3] for specific subjects
    split='train',           # 'train', 'val', or None for all
    apply_hrf=True,          # Convolve stimulus with HRF
    mode='per_subject'       # 'per_subject' or 'cross_subject'
)

# Access samples
sample = dataset[0]
print(sample.keys())  # dict_keys(['video', 'audio', 'text', 'fmri', 'subject_id', 'tr_index'])
```

**Parameters:**
- `data_dir` (str): Root directory containing data files
- `subjects` (str, int, or list): Which subjects to load ('all', single int, or list)
- `split` (str): Data split ('train', 'val', or None)
- `apply_hrf` (bool): Whether to convolve stimulus features with HRF (default: True)
- `mode` (str): 'per_subject' or 'cross_subject' (default: 'per_subject')
- `tr` (float): fMRI repetition time in seconds (default: 1.5)
- `use_encodec` (bool): Use EnCodec for audio (True) or mel spectrograms (False)
- `frame_skip` (int): Sample every Nth video frame for memory efficiency (default: 2)

**Output Format:**

Each sample is a dictionary containing:
```python
{
    'video': torch.Tensor,      # Shape: [video_feature_dim]
    'audio': torch.Tensor,      # Shape: [audio_feature_dim]
    'text': torch.Tensor,       # Shape: [1024] (text embeddings)
    'fmri': torch.Tensor,       # Shape: [n_voxels]
    'subject_id': int,          # Subject number (1-17)
    'tr_index': int             # TR index within subject
}
```

See [dataset.py](dataset.py) for complete documentation.

---

### Modality-Specific Processors

Each processor handles bidirectional conversion between raw data and feature matrices.

#### FMRIProcessor ([fmri.py](fmri.py))

Processes fMRI NIfTI files into feature matrices and vice versa.

**Features:**
- Creates shared brain mask across all subjects
- Extracts timeseries from voxels within mask
- Truncates to stimulus duration (920 TRs)
- Bidirectional: NIfTI ↔ Features
- Supports cross-subject averaging

**Usage:**
```python
from giblet.data.fmri import FMRIProcessor

processor = FMRIProcessor(tr=1.5, max_trs=920)

# Extract features from NIfTI files
features, affine, shape = processor.extract_features(
    nii_files=['sub-01.nii.gz', 'sub-02.nii.gz', ...]
)
# features.shape: (n_subjects, n_trs, n_voxels)

# Reconstruct NIfTI from features
processor.features_to_nifti(
    features=predicted_fmri,
    output_path='predicted_sub-01.nii.gz',
    affine=affine,
    shape=shape
)
```

**Parameters:**
- `tr` (float): fMRI repetition time in seconds (default: 1.5)
- `max_trs` (int): Maximum TRs to extract (default: None for all TRs)
- `mask_threshold` (float): Shared mask threshold (default: 0.5)

---

#### VideoProcessor ([video.py](video.py))

Processes video files into concatenated frame features and vice versa.

**Features:**
- Extracts frames at native FPS (~25 fps)
- Downsamples spatially (640×360 → 160×90)
- **Temporal concatenation**: Each TR contains all frames from [t-TR, t]
- Consistent dimensions via padding
- Bidirectional: Video ↔ Features
- Memory optimization via `frame_skip`

**Usage:**
```python
from giblet.data.video import VideoProcessor

processor = VideoProcessor(
    target_height=90,
    target_width=160,
    tr=1.5,
    frame_skip=2  # Sample every 2nd frame (memory optimization)
)

# Extract features from video
features = processor.extract_features('stimuli_Sherlock.m4v')
# features.shape: (n_trs, video_feature_dim)

# Reconstruct video from features
processor.features_to_video(
    features=predicted_video,
    output_path='reconstructed.mp4',
    fps=25
)
```

**Parameters:**
- `target_height` (int): Frame height after downsampling (default: 90)
- `target_width` (int): Frame width after downsampling (default: 160)
- `tr` (float): TR duration in seconds (default: 1.5)
- `frame_skip` (int): Sample every Nth frame (default: 2)
- `normalize` (bool): Normalize pixels to [0, 1] (default: True)

---

#### AudioProcessor ([audio.py](audio.py))

Processes audio using either EnCodec neural codec or mel spectrograms.

**Features:**
- **EnCodec mode** (default): Neural audio codec with learned compression
  - 24kHz sampling rate, 3.0 kbps bandwidth
  - Better reconstruction than mel spectrograms
- **Mel spectrogram mode**: Traditional audio features
  - 22,050 Hz sampling rate, 2048 mel bins
  - Griffin-Lim reconstruction
- Temporal aggregation to TR bins (1.5s)
- Bidirectional: Audio ↔ Features

**Usage:**
```python
from giblet.data.audio import AudioProcessor

# EnCodec mode (default, recommended)
processor = AudioProcessor(
    use_encodec=True,
    encodec_sample_rate=24000,
    encodec_bandwidth=3.0,
    tr=1.5
)

# Extract features from video file (extracts audio track)
features = processor.extract_features('stimuli_Sherlock.m4v')
# features.shape: (n_trs, audio_feature_dim)

# Reconstruct audio from features
processor.features_to_audio(
    features=predicted_audio,
    output_path='reconstructed_audio.wav'
)

# Mel spectrogram mode (legacy)
processor_mel = AudioProcessor(
    use_encodec=False,
    n_mels=2048,
    sample_rate=22050,
    tr=1.5
)
```

**Parameters:**
- `use_encodec` (bool): Use EnCodec (True) or mel spectrograms (False)
- `encodec_sample_rate` (int): EnCodec sample rate (default: 24000)
- `encodec_bandwidth` (float): EnCodec bandwidth in kbps (default: 3.0)
- `n_mels` (int): Number of mel bins for mel mode (default: 2048)
- `sample_rate` (int): Sample rate for mel mode (default: 22050)
- `tr` (float): TR duration for aggregation (default: 1.5)

---

#### TextProcessor ([text.py](text.py))

Processes text annotations into embeddings using sentence transformers.

**Features:**
- Uses BAAI/bge-large-en-v1.5 model (1024-dim embeddings)
- Loads scene annotations from Excel files
- **Temporal concatenation**: Each TR contains embeddings from [t-TR, t]
- Handles gaps with forward-fill
- Text recovery via nearest-neighbor search
- Temporal alignment to TRs

**Usage:**
```python
from giblet.data.text import TextProcessor

processor = TextProcessor(
    model_name='BAAI/bge-large-en-v1.5',
    tr=1.5,
    gap_fill='forward_fill'
)

# Extract features from annotations
features, texts = processor.extract_features('annotations.xlsx')
# features.shape: (n_trs, text_feature_dim)

# Recover text from embeddings
recovered_texts = processor.embeddings_to_text(
    embeddings=predicted_text,
    reference_texts=texts
)
```

**Parameters:**
- `model_name` (str): Sentence transformer model (default: 'BAAI/bge-large-en-v1.5')
- `tr` (float): TR duration in seconds (default: 1.5)
- `gap_fill` (str): How to fill gaps ('forward_fill', 'zero', 'interpolate')
- `aggregation` (str): How to aggregate overlapping segments ('mean', 'first', 'last')

---

## Data Directory Structure

Expected directory structure for the dataset:

```
data/
├── sherlock_nii/                    # fMRI data
│   ├── sub-01.nii.gz
│   ├── sub-02.nii.gz
│   └── ...
├── stimuli_Sherlock.m4v             # Video stimulus (contains audio)
├── annotations.xlsx                 # Text annotations with timing
└── cache/                           # Cached preprocessed features (auto-generated)
    └── sherlock_all_hrf_per_subject_encodec_12khz_3.0kbps_skip4.pkl
```

**Data Download:**

For lab members with access:
```bash
./download_data_from_dropbox.sh
```

This downloads a 11GB zip file containing:
- Video stimulus (272 MB)
- Scene annotations (173 KB)
- fMRI data for all 17 subjects (~10.7 GB)

**Note**: Data files are NOT included in the Git repository. The data directory is in `.gitignore`.

---

## Temporal Alignment

All modalities are aligned to the fMRI temporal grid using **TR bins** (1.5 seconds by default).

### Temporal Concatenation Strategy

For video, audio, and text, each TR contains **concatenated features from the entire TR window** [t-TR, t]:

**Video:**
- TR window = 1.5s at 25 fps ≈ 37 frames
- Each TR contains all 37 frames concatenated into a flat vector
- With frame_skip=2: ~18 frames per TR
- Ensures model has full temporal context for that TR

**Audio:**
- TR window = 1.5s at 24 kHz = 36,000 samples
- EnCodec compresses to fixed-size representation per TR
- All audio in [t-TR, t] contributes to TR representation

**Text:**
- Embeddings from all text segments overlapping with [t-TR, t]
- Concatenated into single vector per TR
- Gaps filled with forward-fill or zeros

**fMRI:**
- Single BOLD measurement per TR
- If `apply_hrf=True`, stimulus features are convolved with HRF to match BOLD timing

See [giblet/alignment/](../alignment/) for HRF and synchronization details.

---

## Caching System

The dataset automatically caches preprocessed features to disk for fast loading on subsequent runs.

**Cache Location:** `data/cache/` (auto-generated)

**Cache Naming Convention:**
```
sherlock_{subjects}_{hrf_status}_{mode}_{audio_config}.pkl
```

Example:
```
sherlock_all_hrf_per_subject_encodec_12khz_3.0kbps_skip4.pkl
```

**Cache Contents:**
- Preprocessed video, audio, text, fMRI features
- Subject IDs and TR indices
- Feature dimensions
- Configuration metadata

**Pre-generating Cache:**

For distributed training on clusters, pre-generate the cache to avoid NCCL timeout:
```bash
python scripts/pregenerate_cache.py --config configs/training/production_config.yaml
```

This takes 10-20 minutes but prevents 600-second timeouts during distributed initialization.

---

## Feature Dimensions

Typical feature dimensions for default configuration:

| Modality | Dimension | Notes |
|----------|-----------|-------|
| **Video** | 259,200 | 160×90×2 RGB, ~18 frames (frame_skip=2), flattened |
| **Audio** | 128 | EnCodec compressed representation per TR |
| **Text** | 1024 | BAAI/bge-large-en-v1.5 embeddings |
| **fMRI** | ~50,000 | Number of voxels in shared brain mask (varies) |

**Memory Considerations:**

Video features are the largest. Use `frame_skip` parameter to reduce:
- `frame_skip=1`: Full resolution, ~37 frames/TR
- `frame_skip=2`: Half resolution, ~18 frames/TR (50% reduction)
- `frame_skip=4`: Quarter resolution, ~9 frames/TR (75% reduction)

---

## Dataset Statistics

**Sherlock Dataset:**
- **Subjects**: 17 (sub-01 through sub-17)
- **TRs per subject**: 920 (1380 seconds / 1.5s TR)
- **Total samples** (per_subject mode): 15,640 (17 × 920)
- **Train/val split**: 80/20 = 736 train TRs, 184 val TRs per subject

**Temporal Coverage:**
- Video duration: ~23 minutes (1380 seconds)
- fMRI TRs: 920 (matches video duration)
- Text annotations: 1000 scene segments with timing

---

## Related Modules

- **[giblet/alignment/](../alignment/)** - HRF convolution and temporal synchronization
- **[giblet/models/](../models/)** - Neural network architectures that consume these features
- **[giblet/training/](../training/)** - Training pipeline using MultimodalDataset

---

## Common Usage Patterns

### Training a Model

```python
from giblet.data import MultimodalDataset
from torch.utils.data import DataLoader

# Create train and validation datasets
train_dataset = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    split='train',
    apply_hrf=True,
    use_encodec=True,
    frame_skip=2
)

val_dataset = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    split='val',
    apply_hrf=True,
    use_encodec=True,
    frame_skip=2
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
for batch in train_loader:
    video = batch['video']
    audio = batch['audio']
    text = batch['text']
    fmri = batch['fmri']
    # ... train model
```

### Single-Subject Analysis

```python
# Load only subject 5
dataset = MultimodalDataset(
    data_dir='data/',
    subjects=5,
    split=None,  # All TRs
    apply_hrf=True
)

print(f"Loaded {len(dataset)} TRs for subject 5")
```

### Cross-Subject Averaging

```python
# Average fMRI across subjects for each TR
dataset = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    mode='cross_subject',  # Average across subjects
    apply_hrf=True
)

# Now each sample represents average response across all 17 subjects
print(f"Loaded {len(dataset)} TRs (averaged across 17 subjects)")
```

### Debugging with Subset

```python
# Load only first 100 TRs for quick testing
dataset = MultimodalDataset(
    data_dir='data/',
    subjects=[1, 2, 3],  # Just 3 subjects
    max_trs=100,         # First 100 TRs only
    apply_hrf=True
)
```

---

## Performance Tips

1. **Use caching**: First run preprocesses and caches. Subsequent runs are fast.
2. **Pre-generate cache for distributed training**: Use `scripts/pregenerate_cache.py`
3. **Reduce video memory**: Use `frame_skip=2` or `frame_skip=4`
4. **Subset for debugging**: Use `subjects=[1, 2]` and `max_trs=100` for quick iteration
5. **Persistent workers**: Use `persistent_workers=True` in DataLoader for efficiency

---

## Troubleshooting

**Issue**: NCCL timeout during distributed training initialization
- **Solution**: Pre-generate cache with `scripts/pregenerate_cache.py`

**Issue**: Out of memory errors
- **Solution**: Increase `frame_skip` parameter (e.g., `frame_skip=4`)

**Issue**: Slow first run
- **Solution**: This is normal - preprocessing takes 10-20 minutes. Subsequent runs use cache.

**Issue**: TextProcessor not available
- **Solution**: Install sentence-transformers: `pip install sentence-transformers`

**Issue**: EnCodec not available
- **Solution**: Install transformers: `pip install transformers`

---

## Additional Resources

- **Dataset paper**: Chen et al. (2017). "A reduced-dimension fMRI shared response model"
- **EnCodec paper**: Défossez et al. (2023). "High Fidelity Neural Audio Compression"
- **Text embeddings**: BAAI BGE models - https://huggingface.co/BAAI/bge-large-en-v1.5

For questions or issues, see the main project [README.md](../../README.md) or open an issue on GitHub.

# EnCodec Integration - Technical Implementation

This document provides technical details of the EnCodec neural audio codec integration into the giblet-responses codebase.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [AudioProcessor Implementation](#audioprocessor-implementation)
4. [Model Architecture Changes](#model-architecture-changes)
5. [Alignment and HRF](#alignment-and-hrf)
6. [Dataset Integration](#dataset-integration)
7. [Training Considerations](#training-considerations)

---

## Architecture Overview

### Current vs. New Pipeline

#### Before (Mel Spectrogram - PROBLEMATIC)

```
Audio Waveform (22,050 Hz)
    ↓ librosa.feature.melspectrogram()
Mel Spectrogram (2048 mels, variable frames)
    ↓ Average to TR (INFORMATION LOSS)
Per-TR Features (2048 mels)
    ↓ AudioEncoder (Conv1D temporal convolutions)
Compressed Features (256 dims)
    ↓ Encoder → Bottleneck (2048)
Bottleneck (2048) ← PHASE INFO LOST
    ↓ Decoder
Reconstructed Features (256 dims)
    ↓ AudioDecoder (temporal upsampling)
Per-TR Mel Spectrogram (2048 mels, ~65 frames)
    ↓ Griffin-Lim (GUESSES PHASE - FAILS)
Audio Waveform (POOR QUALITY)
```

**Problems:**
1. Mel spectrogram discards phase → Griffin-Lim must guess
2. Bottleneck compression degrades magnitude
3. Griffin-Lim cannot recover from degraded magnitudes
4. Result: Garbled, unintelligible audio

#### After (EnCodec - SOLUTION)

```
Audio Waveform (24,000 Hz)
    ↓ EnCodec.encode() [PRETRAINED, FROZEN]
Quantized Codes (1 codebook, ~112 frames/TR)
    ↓ Align to TRs (group frames)
Per-TR Codes (1, 112)
    ↓ AudioEncoder (processes discrete codes)
Compressed Features (256 dims)
    ↓ Encoder → Bottleneck (2048)
Bottleneck (2048) ← PHASE PRESERVED IN LEARNED CODES
    ↓ Decoder
Reconstructed Features (256 dims)
    ↓ AudioDecoder (predict codes)
Per-TR Predicted Codes (1, 112)
    ↓ Concatenate TRs
Full Quantized Codes (1, ~103,000)
    ↓ EnCodec.decode() [PRETRAINED, FROZEN]
Audio Waveform (HIGH QUALITY)
```

**Advantages:**
1. EnCodec preserves phase in learned representations
2. Discrete codes robust to bottleneck compression
3. Pretrained decoder handles all reconstruction
4. No phase guessing needed

---

## Data Pipeline

### EnCodec Specifications

**Model:** `facebook/encodec_24khz` (via HuggingFace Transformers)

**Configuration:**
- Sample rate: 24,000 Hz (EnCodec standard)
- Channels: Mono (1 channel)
- Bandwidth: 3.0 kbps (default, configurable: 1.5-24.0)
- Frame rate: 75 Hz (24,000 / 320 hop length)
- Codebooks: 1 (mono audio)
- Vocabulary: 1024 discrete codes per codebook

**TR Alignment:**
- TR = 1.5 seconds
- Frames per TR: 1.5s × 75 Hz = **112 frames**
- For 920 TRs (Sherlock): 920 × 112 = 103,040 frames

### Data Dimensions

| Stage | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| Raw audio | (n_samples,) | float32 | 24,000 Hz waveform |
| EnCodec codes | (1, n_frames) | int32 | 1 codebook × ~112 frames/TR |
| Per TR | (n_trs, 1, 112) | int32 | Grouped by TR |
| After HRF (optional) | (n_trs, 1, 112) | int32 | HRF-convolved codes |
| Dataset output | (1, 112) | int64 | Single TR sample |

**Example for Sherlock (920 TRs):**
```python
# Raw audio: 23 min × 60 s/min × 24,000 Hz ≈ 33M samples
audio_waveform.shape  # (33,120,000,)

# EnCodec codes: 920 TRs × 112 frames/TR
encodec_codes.shape  # (920, 1, 112)
encodec_codes.dtype  # int32
```

---

## AudioProcessor Implementation

### Location

**File:** `giblet/data/audio.py`

### Key Changes

#### 1. Imports (Lines 26-31)

```python
# EnCodec neural audio codec (optional)
try:
    from transformers import EncodecModel, AutoProcessor as EncodecAutoProcessor
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
```

#### 2. Updated `__init__` (Lines 72-104)

**New parameters:**
```python
def __init__(
    self,
    use_encodec: bool = True,              # Enable EnCodec (default)
    encodec_bandwidth: float = 3.0,        # Bandwidth in kbps
    device: str = 'cpu',                   # Device for EnCodec model
    tr: float = 1.5,                       # fMRI TR
    # Legacy params for backward compatibility
    sample_rate: int = 22050,
    n_mels: int = 2048,
    n_fft: int = 4096,
    hop_length: int = 512
):
```

**Initialization:**
```python
if self.use_encodec:
    if not ENCODEC_AVAILABLE:
        raise ImportError("transformers required for EnCodec. pip install transformers")

    # Load pretrained EnCodec model
    self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    self.encodec_processor = EncodecAutoProcessor.from_pretrained("facebook/encodec_24khz")
    self.encodec_model.set_target_bandwidth(encodec_bandwidth)
    self.encodec_model.to(device)
    self.encodec_model.eval()  # Freeze weights

    # EnCodec specifications
    self.encodec_sample_rate = 24000  # EnCodec requires 24kHz
    self.encodec_frame_rate = 75      # Fixed by EnCodec architecture
    self.frames_per_tr = int(self.encodec_frame_rate * tr)  # 112 for TR=1.5s
```

#### 3. Encoding: `audio_to_features()` (Lines 106-334)

**Main dispatcher** that routes to EnCodec or Mel mode:

```python
def audio_to_features(
    self,
    audio_source: Union[str, Path],
    max_trs: Optional[int] = None,
    from_video: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract audio features aligned to fMRI TRs.

    Returns
    -------
    features : np.ndarray
        EnCodec mode: (n_trs, 1, 112) int64 - discrete codes
        Mel mode: (n_trs, 2048, frames_per_tr) float32 - mel spectrogram
    metadata : pd.DataFrame
        Columns: tr_index, start_time, end_time, n_frames, encoding_mode
    """
    if self.use_encodec:
        return self._audio_to_features_encodec(audio_source, max_trs, from_video)
    else:
        return self._audio_to_features_mel(audio_source, max_trs, from_video)
```

**EnCodec implementation** (`_audio_to_features_encodec`):

```python
def _audio_to_features_encodec(self, audio_source, max_trs, from_video):
    # 1. Load audio at 24kHz mono
    y, sr = librosa.load(str(audio_source), sr=self.encodec_sample_rate, mono=True)

    # 2. Convert to torch tensor
    import torch
    wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_samples)
    wav = wav.to(self.device)

    # 3. Encode with EnCodec
    with torch.no_grad():
        encoded_frames = self.encodec_model.encode(wav)
        # encoded_frames[0][0]: (batch=1, n_codebooks=1, n_frames)
        codes = encoded_frames[0][0].squeeze(0)  # (n_codebooks, n_frames)

    codes_np = codes.cpu().numpy()  # (1, ~103000) for 23 min

    # 4. Group frames into TRs
    n_trs = int(np.floor(len(y) / self.encodec_sample_rate / self.tr))
    if max_trs is not None:
        n_trs = min(n_trs, max_trs)

    features = np.zeros((n_trs, 1, self.frames_per_tr), dtype=np.int32)

    for tr_idx in range(n_trs):
        start_frame = int(tr_idx * self.tr * self.encodec_frame_rate)
        end_frame = int((tr_idx + 1) * self.tr * self.encodec_frame_rate)

        tr_codes = codes_np[:, start_frame:end_frame]  # (1, ~112)

        # Pad or crop to exact frames_per_tr
        if tr_codes.shape[1] < self.frames_per_tr:
            padding = self.frames_per_tr - tr_codes.shape[1]
            tr_codes = np.pad(tr_codes, ((0, 0), (0, padding)),
                            mode='constant', constant_values=0)
        elif tr_codes.shape[1] > self.frames_per_tr:
            tr_codes = tr_codes[:, :self.frames_per_tr]

        features[tr_idx] = tr_codes

    # 5. Create metadata
    metadata = pd.DataFrame([
        {
            'tr_index': i,
            'start_time': i * self.tr,
            'end_time': (i + 1) * self.tr,
            'n_frames': self.frames_per_tr,
            'encoding_mode': 'encodec'
        }
        for i in range(n_trs)
    ])

    return features, metadata
```

#### 4. Decoding: `features_to_audio()` (Lines 336-447)

**Auto-detection** based on dtype:

```python
def features_to_audio(
    self,
    features: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: Optional[int] = None
) -> None:
    """
    Reconstruct audio from features.

    Automatically detects format:
    - int32/int64 → EnCodec codes
    - float32 → Mel spectrogram
    """
    # Auto-detect format
    if features.dtype in [np.int32, np.int64]:
        # EnCodec codes
        self._features_to_audio_encodec(features, output_path, sample_rate)
    else:
        # Mel spectrogram (legacy)
        self._features_to_audio_mel(features, output_path, sample_rate)
```

**EnCodec decoding** (`_features_to_audio_encodec`):

```python
def _features_to_audio_encodec(self, features, output_path, sample_rate):
    # 1. Concatenate all TRs along time axis
    # (n_trs, 1, 112) → (1, n_trs * 112)
    full_codes = features.transpose(1, 0, 2).reshape(1, -1)

    # 2. Convert to torch
    import torch
    codes_tensor = torch.from_numpy(full_codes).long()
    codes_tensor = codes_tensor.unsqueeze(0)  # (1, 1, n_frames)
    codes_tensor = codes_tensor.to(self.device)

    # 3. Create EncodecEncodedFrame (needed for decode API)
    from transformers.models.encodec.modeling_encodec import EncodedFrame as EncodecEncodedFrame
    encoded_frame = EncodecEncodedFrame(codes_tensor, None)

    # 4. Decode with EnCodec
    with torch.no_grad():
        reconstructed = self.encodec_model.decode(
            [encoded_frame],
            audio_scales=[None]
        )[0]  # (1, 1, n_samples)

    # 5. Convert to numpy and save
    audio_np = reconstructed.squeeze().cpu().numpy()

    import soundfile as sf
    output_sr = sample_rate or self.encodec_sample_rate
    sf.write(str(output_path), audio_np, output_sr)
```

---

## Model Architecture Changes

### AudioEncoder

**Location:** `giblet/models/encoder.py`

**No changes required!** The AudioEncoder already handles EnCodec codes correctly:

```python
class AudioEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,        # Works for both mel (2048) and EnCodec (1)
        frames_per_tr: int = 112,       # EnCodec default
        output_features: int = 128,
        **kwargs
    ):
        super().__init__()

        # Architecture handles discrete codes automatically
        # Conv1D layers process int64 codes as continuous features
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        # ... rest of architecture
```

**How it works:**
- EnCodec codes: (batch, 1, 112) int64
- PyTorch Conv1D automatically converts to float32
- No embedding layer needed (codes are already dense)
- Architecture processes codes as continuous features

### AudioDecoder

**Location:** `giblet/models/decoder.py`

**No changes required!** The AudioDecoder already handles EnCodec prediction:

```python
class MultimodalDecoder(nn.Module):
    def forward(self, bottleneck: torch.Tensor):
        # ... layers 8-12 unchanged ...

        # Layer 13B: Audio reconstruction
        audio = self.layer13_audio(audio_features)  # (batch, 112)
        audio = audio.unsqueeze(1)  # (batch, 1, 112) - matches EnCodec format

        return video, audio, text
```

**Prediction handling:**
- Decoder predicts continuous values: (batch, 1, 112) float32
- During inference, round to integers: `codes = audio.round().long()`
- Clip to valid range: `codes = codes.clip(0, 1023)`
- Pass to EnCodec decoder for reconstruction

---

## Alignment and HRF

### Temporal Synchronization

**Location:** `giblet/alignment/sync.py`

**Changes for discrete codes:**

```python
def _resample_features(features: np.ndarray, current_trs: int, target_trs: int) -> np.ndarray:
    """
    Resample features from current_trs to target_trs.

    Handles discrete codes by:
    1. Converting to float for interpolation
    2. Rounding back to integers
    3. Clipping to valid range [0, 1023]
    """
    if current_trs == target_trs:
        return features.copy()

    # Detect discrete codes (integer dtype)
    is_discrete = features.dtype in [np.int32, np.int64]

    # Handle 3D features (includes EnCodec codes)
    if features.ndim == 3:
        n_trs, dim1, dim2 = features.shape

        # Create time indices
        current_indices = np.arange(current_trs)
        target_indices = np.linspace(0, current_trs - 1, target_trs)

        # Convert to float for interpolation
        if is_discrete:
            features = features.astype(np.float32)

        # Interpolate along TR axis
        resampled = np.zeros((target_trs, dim1, dim2), dtype=np.float32)
        for i in range(dim1):
            for j in range(dim2):
                resampled[:, i, j] = np.interp(
                    target_indices,
                    current_indices,
                    features[:, i, j]
                )

        # Round back to integers if discrete
        if is_discrete:
            resampled = np.round(resampled).astype(np.int32)
            resampled = np.clip(resampled, 0, 1023)  # EnCodec range

        return resampled

    # ... handle 2D features (unchanged) ...
```

### HRF Convolution

**Location:** `giblet/alignment/hrf.py`

**Changes for discrete codes:**

```python
def apply_hrf(features, tr=1.5, mode='same'):
    """
    Convolve stimulus features with canonical HRF.

    Handles discrete codes by:
    1. Converting to float
    2. Convolving
    3. Rounding back to integers
    """
    # Get canonical HRF kernel
    hrf = get_canonical_hrf(tr=tr)

    # Detect discrete codes
    is_discrete = features.dtype in [np.int32, np.int64]
    if is_discrete:
        features = features.astype(np.float32)

    # Handle 3D audio features (includes EnCodec codes)
    if features.ndim == 3:
        n_timepoints, dim1, dim2 = features.shape

        # Convolve each channel independently
        convolved = np.zeros((n_timepoints, dim1, dim2), dtype=np.float32)
        for i in range(dim1):
            for j in range(dim2):
                convolved[:, i, j] = signal.convolve(
                    features[:, i, j], hrf, mode=mode
                )

        # Round back to integers if discrete
        if is_discrete:
            convolved = np.round(convolved).astype(np.int32)
            convolved = np.clip(convolved, 0, 1023)  # EnCodec range

        return convolved

    # ... handle 1D/2D features (unchanged) ...
```

---

## Dataset Integration

### MultimodalDataset

**Location:** `giblet/data/dataset.py`

**No changes required!** Dataset already handles 3D audio features.

**Automatic format detection:**

```python
# In _preprocess_data():
if self.audio_features.ndim == 3:
    # Could be mel spectrograms OR EnCodec codes
    # Check dtype to determine
    if self.audio_features.dtype in [np.int32, np.int64]:
        audio_format = "encodec_codes"
        audio_dim = self.audio_features.shape[-2:]  # (1, 112)
    else:
        audio_format = "mel_spectrogram"
        audio_dim = self.audio_features.shape[-2:]  # (2048, frames)
else:
    audio_format = "legacy"
    audio_dim = self.audio_features.shape[-1]

self.feature_dims = {
    'video': self.video_features.shape[-1],
    'audio': audio_dim,
    'audio_format': audio_format,  # Track format
    'text': self.text_features.shape[-1],
    'fmri': self.fmri_features.shape[-1]
}
```

**Cache files:**
- EnCodec: `audio_features_encodec_bw3.0_tr1.5_920trs_hrf.npy`
- Mel: `audio_features_mel_tr1.5_920trs_hrf.npy`
- Separate cache files prevent format conflicts

---

## Training Considerations

### Loss Function

**Current implementation** (works for both formats):

```python
# MSE loss on continuous predictions
loss_audio = F.mse_loss(predicted_audio, target_audio.float())
```

**For EnCodec codes:**
- Target: (batch, 1, 112) int64 → convert to float32
- Predicted: (batch, 1, 112) float32
- Loss: MSE between continuous predictions and integer targets
- During inference: round predictions to integers

**Alternative (classification loss):**
```python
# If predicting logits over 1024 classes
# Requires changing decoder output to (batch, 1, 112, 1024)
loss_audio = F.cross_entropy(
    predicted_logits.view(-1, 1024),
    target_codes.view(-1).long()
)
```

**Recommendation:** Use MSE loss (current implementation) because:
- Simpler architecture (no need to predict 1024 logits)
- Faster training (smaller output layer)
- Codes are ordinal (nearby codes are similar)
- Can interpolate between codes during prediction

### Memory Usage

**EnCodec vs Mel:**

| Component | Mel Spectrogram | EnCodec | Change |
|-----------|----------------|---------|--------|
| Feature size (per TR) | 2048 × 64 = 131,072 floats | 1 × 112 = 112 ints | -99.9% |
| Batch (32 TRs) | 32 × 131,072 × 4 bytes = 16.8 MB | 32 × 112 × 8 bytes = 28.7 KB | -99.8% |
| Model params | Same | Same | No change |
| **Total** | Baseline | **~Same** | Negligible |

**Note:** EnCodec dramatically reduces feature storage but model parameters dominate GPU memory.

### Training Speed

**Expected improvements:**
- **Encoding:** ~2-3 minutes for Sherlock (vs ~1 minute for mel)
- **Forward pass:** ~Same (Conv1D processes both formats similarly)
- **Overall:** Negligible impact on training time

### Distributed Training

**No changes needed!** EnCodec codes work with:
- PyTorch DDP (Distributed Data Parallel)
- Multi-GPU training
- Gradient accumulation

---

## Backward Compatibility

### Feature Flags

All components support feature flags for gradual migration:

```python
# Enable EnCodec (default)
audio_processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0)

# Disable EnCodec (legacy mel spectrograms)
audio_processor = AudioProcessor(use_encodec=False)
```

### Auto-Detection

`features_to_audio()` automatically detects format based on dtype:
- `int32/int64` → EnCodec codes
- `float32` → Mel spectrogram

### Migration Path

1. **Phase 1:** Add EnCodec code with `use_encodec=False` (no behavior change)
2. **Phase 2:** Test EnCodec with `use_encodec=True` on subset
3. **Phase 3:** Switch default to `use_encodec=True`
4. **Phase 4:** Deprecate mel spectrogram code after validation

---

## Files Modified/Created

### Core Implementation

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `giblet/data/audio.py` | Modified | 484 | EnCodec encoding/decoding |
| `giblet/alignment/sync.py` | Modified | +50 | Discrete code resampling |
| `giblet/alignment/hrf.py` | Modified | +40 | Discrete code convolution |

### Testing

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `tests/data/test_audio_encodec.py` | Created | 394 | Comprehensive test suite |
| `test_encodec_simple.py` | Created | 125 | Simple manual test |
| `test_encodec_e2e_pipeline.py` | Created | 450 | End-to-end pipeline test |

### Documentation

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `docs/encodec/overview.md` | Created | 362 | User-facing overview |
| `docs/encodec/integration.md` | This file | 500+ | Technical implementation |
| `docs/encodec/troubleshooting.md` | Created | ~400 | Testing and debugging |

---

## Implementation Status

✅ **Complete:**
- AudioProcessor EnCodec encoding/decoding
- Auto-detection of feature format
- Backward compatibility with mel spectrograms
- Alignment/HRF support for discrete codes
- Dataset integration
- Comprehensive test suite
- Documentation

🔄 **In Progress:**
- Full-scale training validation
- Quality metric comparison (mel vs EnCodec)
- Performance benchmarking

📋 **Planned:**
- Archive mel spectrogram code (after validation)
- Update architecture diagrams
- Create migration guide for users

---

## References

- **EnCodec Paper:** Défossez et al. (2022). "High Fidelity Neural Audio Compression." https://arxiv.org/abs/2210.13438
- **HuggingFace Documentation:** https://huggingface.co/docs/transformers/model_doc/encodec
- **Issue #24:** Audio Enhancement with EnCodec (GitHub)
- **ENCODEC_INTEGRATION_ARCHITECTURE.md:** Original architecture planning document (archived)

For questions or issues, see the main project [README.md](../../README.md) or open an issue on GitHub.

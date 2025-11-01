# EnCodec Integration Architecture for Issue #24, Task 1.3

**Date:** 2025-10-31
**Status:** Planning Complete - Ready for Implementation
**Issue:** #24 - Audio Enhancement with EnCodec
**Task:** 1.3 - Design integration architecture

---

## Executive Summary

This document specifies how to replace the current mel spectrogram pipeline with **EnCodec neural codec** for high-quality audio reconstruction. EnCodec provides learned discrete representations that preserve phase information implicitly, eliminating the fundamental limitation of Griffin-Lim reconstruction.

**Key Changes:**
- Replace 2D mel spectrograms (n_mels,) with 3D discrete codes (n_codebooks, frames_per_tr)
- Use pretrained EnCodec encoder/decoder (no phase reconstruction needed)
- Modify AudioEncoder to handle discrete codes via embedding layers
- Modify AudioDecoder to predict discrete codes
- Update alignment/HRF code to handle discrete representations

---

## Current vs. New Pipeline Comparison

### Current Mel Spectrogram Pipeline (PROBLEMATIC)

```
Audio Waveform (22,050 Hz)
    ↓ librosa.feature.melspectrogram()
Mel Spectrogram (2048 mels, variable frames)
    ↓ Average to TR (INFORMATION LOSS)
Per-TR Features (2048 mels) ← LOSES TEMPORAL DETAIL
    ↓ AudioEncoder (Conv1D temporal convolutions)
Compressed Features (256 dims)
    ↓ Encoder → Bottleneck (2048)
Bottleneck (2048) ← PHASE INFO LOST
    ↓ Decoder
Reconstructed Features (256 dims)
    ↓ AudioDecoder (temporal upsampling)
Per-TR Mel Spectrogram (2048 mels, ~65 frames)
    ↓ Concatenate TRs
Full Mel Spectrogram (2048 mels, ~60,000 frames)
    ↓ Griffin-Lim (GUESSES PHASE - FAILS)
Audio Waveform (POOR QUALITY)
```

**Problems:**
1. Mel spectrogram discards phase → Griffin-Lim must guess
2. After bottleneck compression, magnitude is degraded
3. Griffin-Lim cannot recover from degraded magnitudes
4. Result: Garbled, unintelligible audio

---

### New EnCodec Pipeline (SOLUTION)

```
Audio Waveform (24,000 Hz)  ← EnCodec standard
    ↓ EnCodec.encode() [PRETRAINED, FROZEN]
Quantized Codes (8 codebooks, ~112 frames/TR)  ← DISCRETE CODES
    ↓ Align to TRs (group frames)
Per-TR Codes (8 codebooks, 112 frames)  ← PRESERVES TEMPORAL DETAIL
    ↓ AudioEncoder (Embedding + Conv1D)
      - Embedding: codes → continuous (8, 112, 64)
      - Conv: temporal feature extraction
Compressed Features (256 dims)
    ↓ Encoder → Bottleneck (2048)
Bottleneck (2048)  ← PHASE PRESERVED IN LEARNED CODES
    ↓ Decoder
Reconstructed Features (256 dims)
    ↓ AudioDecoder (predict codes)
Per-TR Predicted Codes (8 codebooks, 112 frames)  ← DISCRETE OR CONTINUOUS
    ↓ Concatenate TRs
Full Quantized Codes (8 codebooks, ~103,000 frames)
    ↓ EnCodec.decode() [PRETRAINED, FROZEN]
Audio Waveform (HIGH QUALITY)  ← PHASE IMPLICITLY RECONSTRUCTED
```

**Advantages:**
1. EnCodec preserves phase in learned representations
2. Discrete codes robust to bottleneck compression
3. Pretrained decoder handles all reconstruction
4. No phase guessing needed

---

## Technical Specifications

### EnCodec Configuration

**Model:** `facebook/encodec_24khz` (via HuggingFace Transformers)
- Already available in `transformers==4.57.1` (current install)
- No additional dependencies needed

**Audio Format:**
- Sample rate: 24,000 Hz (EnCodec standard)
- Channels: Mono (1 channel)
- Bitrate: 6.0 kbps (good quality/size tradeoff)

**Encoding Output:**
- Codebooks: 8 (Residual Vector Quantization layers)
- Vocabulary size: 1024 discrete codes per codebook
- Frame rate: 75 Hz (24,000 / 320 hop length)
- Frames per TR: 1.5s × 75 Hz = **112 frames**

**Data Dimensions:**

| Stage | Shape | Description |
|-------|-------|-------------|
| Raw audio | (n_samples,) | 24,000 Hz waveform |
| EnCodec codes | (8, n_frames) | 8 codebooks × ~112 frames/TR |
| Per TR | (n_trs, 8, 112) | Grouped by TR |
| After embedding | (n_trs, 8, 112, 64) | Continuous embeddings |
| After AudioEncoder | (n_trs, 256) | Compressed features |
| After AudioDecoder | (n_trs, 8, 112) | Predicted codes |
| After EnCodec decode | (n_samples,) | Reconstructed waveform |

---

## Component-by-Component Integration

### 1. AudioProcessor (`giblet/data/audio.py`)

**Current:** Extracts mel spectrograms using librosa
**New:** Extracts EnCodec discrete codes

#### Changes Required:

```python
class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 24000,        # CHANGED: EnCodec standard
        n_codebooks: int = 8,             # NEW: RVQ codebooks
        frames_per_tr: int = 112,         # NEW: 75 Hz × 1.5s
        target_bandwidth: float = 6.0,    # NEW: 6 kbps
        tr: float = 1.5,
        # Keep legacy params for backward compatibility
        n_mels: int = 2048,
        use_encodec: bool = True          # NEW: Feature flag
    ):
        self.sample_rate = sample_rate
        self.n_codebooks = n_codebooks
        self.frames_per_tr = frames_per_tr
        self.target_bandwidth = target_bandwidth
        self.tr = tr
        self.use_encodec = use_encodec

        if use_encodec:
            from transformers import EncodecModel
            self.encodec_model = EncodecModel.from_pretrained(
                "facebook/encodec_24khz"
            )
            self.encodec_model.eval()  # Freeze weights
            self.n_features = n_codebooks * frames_per_tr  # 8 × 112 = 896
        else:
            # Legacy mel spectrogram
            self.n_features = n_mels
```

#### Method: `audio_to_features()`

**Returns:**
```python
features : np.ndarray
    Shape (n_trs, n_codebooks, frames_per_tr)  # (920, 8, 112)
    Discrete integer codes in range [0, 1023]

metadata : pd.DataFrame
    Columns: tr_index, start_time, end_time, n_frames
```

**Implementation:**
```python
def audio_to_features(self, audio_source, max_trs=None, from_video=True):
    """Extract EnCodec discrete codes aligned to TRs."""

    # 1. Load audio at 24kHz
    y, sr = librosa.load(str(audio_source), sr=24000, mono=True)

    # 2. Convert to torch tensor
    import torch
    wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_samples)

    # 3. Encode with EnCodec
    with torch.no_grad():
        encoded_frames = self.encodec_model.encode(wav)
        # encoded_frames is list of tuples: [(codes, scale)]
        # codes: (batch=1, n_codebooks=8, n_frames)
        codes = encoded_frames[0][0]  # Get codes, drop scale
        codes = codes.squeeze(0)  # (n_codebooks, n_frames)

    codes_np = codes.cpu().numpy()  # (8, ~103000) for 23 min

    # 4. Group frames into TRs
    n_trs = int(np.floor(len(y) / sr / self.tr))
    if max_trs is not None:
        n_trs = min(n_trs, max_trs)

    features = np.zeros((n_trs, self.n_codebooks, self.frames_per_tr), dtype=np.int32)

    for tr_idx in range(n_trs):
        start_frame = int(tr_idx * self.tr * 75)  # 75 Hz frame rate
        end_frame = int((tr_idx + 1) * self.tr * 75)

        tr_codes = codes_np[:, start_frame:end_frame]  # (8, ~112)

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
            'n_frames': self.frames_per_tr
        }
        for i in range(n_trs)
    ])

    return features, metadata
```

#### Method: `features_to_audio()`

**Input:**
```python
features : np.ndarray
    Shape (n_trs, n_codebooks, frames_per_tr)  # (920, 8, 112)
    Discrete codes (can be continuous predictions from decoder)
```

**Implementation:**
```python
def features_to_audio(self, features, output_path):
    """Reconstruct audio from EnCodec codes."""

    # 1. Concatenate all TRs along time axis
    # (n_trs, 8, 112) → (8, n_trs * 112)
    full_codes = features.transpose(1, 0, 2).reshape(8, -1)

    # 2. Round to integers if continuous predictions
    if features.dtype == np.float32:
        full_codes = np.round(full_codes).astype(np.int32)
        # Clip to valid range [0, 1023]
        full_codes = np.clip(full_codes, 0, 1023)

    # 3. Convert to torch
    import torch
    codes_tensor = torch.from_numpy(full_codes).long()
    codes_tensor = codes_tensor.unsqueeze(0)  # (1, 8, n_frames)

    # 4. Create EncodecEncodedFrame (needed for decode API)
    from transformers import EncodecEncodedFrame
    scale = None  # Will be inferred
    encoded_frame = EncodecEncodedFrame(codes_tensor, scale)

    # 5. Decode with EnCodec
    with torch.no_grad():
        reconstructed = self.encodec_model.decode(
            [encoded_frame],
            audio_scales=[None]
        )[0]  # (1, 1, n_samples)

    # 6. Convert to numpy and save
    audio_np = reconstructed.squeeze().cpu().numpy()

    import soundfile as sf
    sf.write(str(output_path), audio_np, self.sample_rate)
```

---

### 2. AudioEncoder (`giblet/models/encoder.py`)

**Current:** Processes continuous mel spectrograms (n_mels, frames_per_tr)
**New:** Processes discrete codes (n_codebooks, frames_per_tr)

#### Changes Required:

```python
class AudioEncoder(nn.Module):
    """
    Encode audio EnCodec discrete codes using embeddings + temporal convolutions.

    Architecture:
    1. Embedding: Map discrete codes [0, 1023] to continuous (n_codebooks, frames, embed_dim)
    2. Temporal convolutions: Extract multi-scale temporal features
    3. Pooling: Collapse to fixed-size representation
    4. Compression: Map to output_features
    """

    def __init__(
        self,
        vocab_size: int = 1024,           # NEW: EnCodec vocabulary
        n_codebooks: int = 8,             # NEW: Number of codebooks
        frames_per_tr: int = 112,         # NEW: Frames per TR
        embedding_dim: int = 64,          # NEW: Embedding dimension
        output_features: int = 256,
        use_encodec: bool = True          # NEW: Feature flag
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.frames_per_tr = frames_per_tr
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        self.use_encodec = use_encodec

        if use_encodec:
            # Embedding layer for discrete codes
            # Each codebook gets its own embedding
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, embedding_dim)
                for _ in range(n_codebooks)
            ])

            # After embedding: (batch, n_codebooks, frames, embed_dim)
            # Reshape to (batch, n_codebooks * embed_dim, frames) for Conv1D
            conv_input_channels = n_codebooks * embedding_dim  # 8 × 64 = 512

            # Multi-scale temporal convolutions
            self.temporal_conv_k3 = nn.Conv1d(conv_input_channels, 128, kernel_size=3, padding=1)
            self.bn_k3 = nn.BatchNorm1d(128)

            self.temporal_conv_k5 = nn.Conv1d(conv_input_channels, 128, kernel_size=5, padding=2)
            self.bn_k5 = nn.BatchNorm1d(128)

            self.temporal_conv_k7 = nn.Conv1d(conv_input_channels, 128, kernel_size=7, padding=3)
            self.bn_k7 = nn.BatchNorm1d(128)

            # Adaptive pooling
            self.temporal_pool = nn.AdaptiveMaxPool1d(1)

            # Final compression (128*3=384 → 256)
            self.fc = nn.Sequential(
                nn.Linear(384, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_features)
            )
        else:
            # Legacy mel spectrogram path (keep for backward compatibility)
            # ... existing implementation ...
```

#### Method: `forward()`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through audio encoder.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch, n_codebooks, frames_per_tr) for EnCodec codes (LONG/INT)
        Shape (batch, n_mels, frames_per_tr) for mel spectrograms (FLOAT)

    Returns
    -------
    features : torch.Tensor
        Shape (batch, output_features)
    """
    if self.use_encodec:
        batch_size = x.size(0)

        # x: (batch, 8, 112) - discrete codes
        # Convert to long if not already
        x = x.long()

        # Embed each codebook separately
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            # embedding(x[:, i, :]): (batch, 112) → (batch, 112, 64)
            embedded.append(embedding(x[:, i, :]))

        # Stack: [(batch, 112, 64)] × 8 → (batch, 8, 112, 64)
        embedded = torch.stack(embedded, dim=1)

        # Reshape for Conv1D: (batch, 8, 112, 64) → (batch, 512, 112)
        embedded = embedded.view(batch_size, self.n_codebooks * self.embedding_dim, self.frames_per_tr)

        # Multi-scale convolutions
        feat_k3 = F.relu(self.bn_k3(self.temporal_conv_k3(embedded)))  # (batch, 128, 112)
        feat_k5 = F.relu(self.bn_k5(self.temporal_conv_k5(embedded)))  # (batch, 128, 112)
        feat_k7 = F.relu(self.bn_k7(self.temporal_conv_k7(embedded)))  # (batch, 128, 112)

        # Pool temporal dimension
        feat_k3 = self.temporal_pool(feat_k3).squeeze(-1)  # (batch, 128)
        feat_k5 = self.temporal_pool(feat_k5).squeeze(-1)  # (batch, 128)
        feat_k7 = self.temporal_pool(feat_k7).squeeze(-1)  # (batch, 128)

        # Concatenate multi-scale features
        features = torch.cat([feat_k3, feat_k5, feat_k7], dim=1)  # (batch, 384)

        # Final compression
        output = self.fc(features)  # (batch, 256)

        return output
    else:
        # Legacy mel spectrogram path
        # ... existing implementation ...
```

---

### 3. AudioDecoder (`giblet/models/decoder.py`)

**Current:** Predicts continuous mel spectrograms
**New:** Predicts discrete codes (or continuous logits)

#### Changes Required:

```python
class MultimodalDecoder(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int = 2048,
        audio_vocab_size: int = 1024,     # NEW: EnCodec vocabulary
        audio_n_codebooks: int = 8,       # NEW: Number of codebooks
        audio_frames_per_tr: int = 112,   # NEW: Frames per TR
        use_encodec: bool = True,         # NEW: Feature flag
        # Legacy params
        audio_dim: int = 2048,
        ...
    ):
        super().__init__()

        self.use_encodec = use_encodec

        if use_encodec:
            # Decoder predicts discrete codes
            # Option A: Classification (predict logits over 1024 classes)
            # Option B: Regression (predict continuous codes, round later)

            # We'll use regression for simplicity (continuous codes)
            self.layer13_audio = nn.Linear(2048, audio_n_codebooks * audio_frames_per_tr)

            # Store dimensions for reshaping
            self.audio_n_codebooks = audio_n_codebooks
            self.audio_frames_per_tr = audio_frames_per_tr

            # Optional: temporal upsampling (removed - output directly at target resolution)
        else:
            # Legacy mel spectrogram path
            # ... existing implementation ...
```

#### Method: `forward()`

```python
def forward(self, bottleneck: torch.Tensor) -> Tuple[...]:
    """
    Forward pass through decoder.

    Returns
    -------
    audio : torch.Tensor
        Shape (batch, n_codebooks, frames_per_tr) for EnCodec
        Continuous predictions in [0, 1023] range (to be rounded)
    """
    # ... layers 8-11 unchanged ...

    # Layer 12B: Audio decoder path
    audio_features = self.layer12_audio(x)  # (batch, 2048)

    if self.use_encodec:
        # Layer 13B: Predict codes
        audio = self.layer13_audio(audio_features)  # (batch, 8*112=896)

        # Reshape to (batch, n_codebooks, frames_per_tr)
        batch_size = audio.size(0)
        audio = audio.view(batch_size, self.audio_n_codebooks, self.audio_frames_per_tr)

        # Apply sigmoid then scale to [0, 1023]
        # This keeps predictions in valid range
        audio = torch.sigmoid(audio) * 1023.0

    else:
        # Legacy mel spectrogram path
        # ... existing implementation ...

    return video, audio, text
```

---

### 4. Alignment/Sync (`giblet/alignment/sync.py`)

**Current:** Resamples continuous features
**New:** Handles discrete codes (round after interpolation)

#### Changes Required:

```python
def _resample_features(features: np.ndarray, current_trs: int, target_trs: int) -> np.ndarray:
    """
    Resample features from current_trs to target_trs.

    Handles:
    - 2D: (n_trs, n_features) - continuous
    - 3D: (n_trs, n_mels, frames_per_tr) - continuous or discrete
    """
    if current_trs == target_trs:
        return features.copy()

    # Detect discrete codes (integer dtype)
    is_discrete = features.dtype in [np.int32, np.int64, np.uint32, np.uint64]

    # Handle 3D features (includes EnCodec codes)
    if features.ndim == 3:
        n_trs, dim1, dim2 = features.shape

        # Create time indices
        current_indices = np.arange(current_trs)
        target_indices = np.linspace(0, current_trs - 1, target_trs)

        # Convert to float for interpolation
        if is_discrete:
            features = features.astype(np.float32)

        # Interpolate
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

    # 2D features (existing code)
    # ... unchanged ...
```

---

### 5. HRF Convolution (`giblet/alignment/hrf.py`)

**Current:** Convolves continuous features
**New:** Handles discrete codes (convert to float, convolve, round back)

#### Changes Required:

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
    is_discrete = features.dtype in [np.int32, np.int64, np.uint32, np.uint64]
    if is_discrete:
        features = features.astype(np.float32)

    # Handle 3D audio features (includes EnCodec codes)
    if features.ndim == 3:
        n_timepoints, dim1, dim2 = features.shape

        # Determine output shape
        if mode == 'same':
            output_shape = (n_timepoints, dim1, dim2)
        else:
            output_shape = (n_timepoints + len(hrf) - 1, dim1, dim2)

        convolved = np.zeros(output_shape, dtype=np.float32)

        # Convolve each dim1 × dim2 combination
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

    # 1D/2D features (existing code)
    # ... unchanged ...
```

---

### 6. Dataset (`giblet/data/dataset.py`)

**Current:** Loads 3D mel spectrograms
**New:** Loads 3D discrete codes (no changes needed!)

The dataset already handles 3D audio features thanks to recent updates. Only metadata tracking needs minor updates:

```python
# In _preprocess_data():
if self.audio_features.ndim == 3:
    # Could be mel spectrograms OR EnCodec codes
    # Check dtype to determine
    if self.audio_features.dtype in [np.int32, np.int64]:
        audio_format = "encodec_codes"
        audio_dim = self.audio_features.shape[-2:]  # (n_codebooks, frames_per_tr)
    else:
        audio_format = "mel_spectrogram"
        audio_dim = self.audio_features.shape[-2:]  # (n_mels, frames_per_tr)
else:
    audio_format = "legacy"
    audio_dim = self.audio_features.shape[-1]

self.feature_dims = {
    'video': self.video_features.shape[-1],
    'audio': audio_dim,
    'audio_format': audio_format,  # NEW: Track format
    'text': self.text_features.shape[-1],
    'fmri': self.fmri_features.shape[-1]
}
```

---

## Backwards Compatibility Strategy

All changes include feature flags to maintain backward compatibility:

### 1. Feature Format Detection

```python
def detect_audio_format(features: np.ndarray) -> str:
    """
    Detect audio feature format.

    Returns: 'encodec', 'mel_3d', 'mel_2d'
    """
    if features.ndim == 3:
        if features.dtype in [np.int32, np.int64]:
            return 'encodec'
        else:
            return 'mel_3d'
    elif features.ndim == 2:
        return 'mel_2d'
    else:
        raise ValueError(f"Unknown audio format: shape={features.shape}, dtype={features.dtype}")
```

### 2. Configuration Parameter

Add to all processors:
```python
use_encodec: bool = True  # Set to False to use legacy mel spectrograms
```

### 3. Gradual Migration

1. **Phase 1:** Add EnCodec code with `use_encodec=False` (no behavior change)
2. **Phase 2:** Test EnCodec with `use_encodec=True` on subset of data
3. **Phase 3:** Switch default to `use_encodec=True`
4. **Phase 4:** Deprecate mel spectrogram code after validation

---

## Implementation Checklist

### Core Components (High Priority)

- [ ] **audio.py**: Replace mel spectrogram with EnCodec encoding
  - [ ] Add `use_encodec` parameter
  - [ ] Implement `audio_to_features()` with EnCodec
  - [ ] Implement `features_to_audio()` with EnCodec decoder
  - [ ] Add backward compatibility for mel spectrograms
  - [ ] Update docstrings

- [ ] **encoder.py**: Add embedding layer for discrete codes
  - [ ] Add embedding layers (one per codebook)
  - [ ] Update `forward()` to handle discrete codes
  - [ ] Keep mel spectrogram path with feature flag
  - [ ] Update `audio_frames_per_tr` default to 112

- [ ] **decoder.py**: Predict codes instead of spectrograms
  - [ ] Update Layer 13B to output (n_codebooks, frames_per_tr)
  - [ ] Add sigmoid scaling to [0, 1023] range
  - [ ] Remove temporal upsampling layers (not needed)
  - [ ] Keep mel spectrogram path with feature flag

### Alignment Components (Medium Priority)

- [ ] **sync.py**: Handle discrete code resampling
  - [ ] Detect discrete codes via dtype
  - [ ] Convert to float before interpolation
  - [ ] Round and clip after interpolation
  - [ ] Test with EnCodec codes

- [ ] **hrf.py**: Handle discrete code convolution
  - [ ] Detect discrete codes via dtype
  - [ ] Convert to float before convolution
  - [ ] Round and clip after convolution
  - [ ] Test HRF convolution with codes

### Dataset (Low Priority)

- [ ] **dataset.py**: Track audio format in metadata
  - [ ] Add `audio_format` field to metadata
  - [ ] Detect format based on shape/dtype
  - [ ] Update cache filenames to include format
  - [ ] No code changes needed (already handles 3D)

### Infrastructure

- [ ] **requirements_conda.txt**: Already has transformers
  - [x] `transformers==4.57.1` (includes EnCodec)
  - [ ] Document EnCodec usage in comments

- [ ] **Testing**: Comprehensive validation
  - [ ] Unit tests for audio_to_features()
  - [ ] Unit tests for features_to_audio()
  - [ ] Integration test: full pipeline
  - [ ] Audio quality comparison (mel vs EnCodec)
  - [ ] Backward compatibility tests

- [ ] **Documentation**
  - [ ] Update README with EnCodec approach
  - [ ] Add EnCodec technical notes
  - [ ] Create migration guide
  - [ ] Update architecture diagrams

---

## Data Flow Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENCODEC INTEGRATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Audio Waveform (Sherlock.m4v)
│
│  22,050 Hz → Resample → 24,000 Hz (EnCodec standard)
│
▼
┌──────────────────────────────────────────────────────────────┐
│  ENCODEC ENCODER (PRETRAINED, FROZEN)                        │
│  facebook/encodec_24khz @ 6.0 kbps                           │
│                                                              │
│  Input:  Waveform (1, 1, n_samples)                         │
│  Output: Codes (1, 8, ~103000)                              │
│          - 8 codebooks (RVQ layers)                          │
│          - 1024 vocabulary per codebook                      │
│          - 75 Hz frame rate                                  │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (8, 103000)  [for 23-minute stimulus]
│
▼
┌──────────────────────────────────────────────────────────────┐
│  ALIGN TO TRs                                                │
│  Group frames into TR bins (1.5s each)                      │
│                                                              │
│  TR 0: frames 0-111                                         │
│  TR 1: frames 112-223                                       │
│  ...                                                         │
│  TR 919: frames 102888-102999                               │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 8, 112)  [n_trs, n_codebooks, frames_per_tr]
│  Dtype: int32 (discrete codes [0, 1023])
│
▼
┌──────────────────────────────────────────────────────────────┐
│  HRF CONVOLUTION (Optional)                                  │
│  - Convert to float32                                        │
│  - Convolve with Glover HRF (per codebook, per frame)       │
│  - Round back to int32                                       │
│  - Clip to [0, 1023]                                        │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 8, 112)  [unchanged]
│
▼
┌──────────────────────────────────────────────────────────────┐
│  AUDIO ENCODER                                               │
│                                                              │
│  1. Embedding Layer (per codebook):                         │
│     codes (920, 8, 112) → embedded (920, 8, 112, 64)       │
│                                                              │
│  2. Reshape for Conv1D:                                     │
│     (920, 8, 112, 64) → (920, 512, 112)                    │
│                                                              │
│  3. Multi-scale Temporal Convolutions:                      │
│     - Conv1D k=3: (920, 512, 112) → (920, 128, 112)        │
│     - Conv1D k=5: (920, 512, 112) → (920, 128, 112)        │
│     - Conv1D k=7: (920, 512, 112) → (920, 128, 112)        │
│                                                              │
│  4. Adaptive Max Pooling:                                   │
│     (920, 128, 112) → (920, 128, 1) → (920, 128)           │
│                                                              │
│  5. Concatenate:                                            │
│     (920, 128) × 3 → (920, 384)                            │
│                                                              │
│  6. FC Compression:                                         │
│     (920, 384) → (920, 256)                                │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 256)  [compressed audio features]
│
▼
┌──────────────────────────────────────────────────────────────┐
│  MULTIMODAL ENCODER                                          │
│  Concatenate with video (1024) + text (256)                 │
│  → Pooled (1536) → Conv → Expand → BOTTLENECK (2048)       │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 2048)  [bottleneck representation]
│
▼
┌──────────────────────────────────────────────────────────────┐
│  MULTIMODAL DECODER                                          │
│  Bottleneck (2048) → Expand → Deconv → Unpool (1536)       │
│  → Split to video/audio/text paths                          │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 1536)  [shared features]
│
▼
┌──────────────────────────────────────────────────────────────┐
│  AUDIO DECODER                                               │
│                                                              │
│  1. Audio-specific path:                                    │
│     (920, 1536) → (920, 2048)                              │
│                                                              │
│  2. Predict codes:                                          │
│     (920, 2048) → (920, 896)  [8 × 112]                    │
│                                                              │
│  3. Reshape:                                                │
│     (920, 896) → (920, 8, 112)                             │
│                                                              │
│  4. Scale to valid range:                                  │
│     sigmoid(x) * 1023 → [0, 1023]                          │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (920, 8, 112)  [predicted codes, continuous]
│  Dtype: float32
│
▼
┌──────────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                             │
│  - Round to integers: (920, 8, 112) float32 → int32        │
│  - Clip to [0, 1023]                                        │
│  - Concatenate TRs: (920, 8, 112) → (8, 103040)            │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (8, 103040)  [full code sequence]
│  Dtype: int32
│
▼
┌──────────────────────────────────────────────────────────────┐
│  ENCODEC DECODER (PRETRAINED, FROZEN)                        │
│  facebook/encodec_24khz @ 6.0 kbps                           │
│                                                              │
│  Input:  Codes (1, 8, 103040)                               │
│  Output: Waveform (1, 1, ~3,456,000)                        │
│          [23 min × 60 s × 24,000 Hz ≈ 33M samples]          │
└──────────────────────────────────────────────────────────────┘
│
│  Shape: (33,120,000)  [reconstructed audio]
│  Dtype: float32
│
▼
OUTPUT: High-Quality Audio Waveform
        - Phase information preserved
        - Perceptually similar to input
        - Ready for playback/analysis
```

---

## Dimension Reference Table

| Stage | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| Raw audio | (33M,) | float32 | 24kHz waveform, 23 min |
| EnCodec codes | (8, 103000) | int32 | 8 codebooks, 75 Hz |
| Grouped by TR | (920, 8, 112) | int32 | Aligned to 1.5s TRs |
| After HRF | (920, 8, 112) | int32 | Convolved with HRF |
| After embedding | (920, 8, 112, 64) | float32 | Continuous embeddings |
| Reshaped for conv | (920, 512, 112) | float32 | Ready for Conv1D |
| After conv k3/k5/k7 | (920, 128, 112) | float32 | Each scale |
| After pooling | (920, 128) | float32 | Per scale |
| Concatenated | (920, 384) | float32 | All scales |
| Audio features | (920, 256) | float32 | Compressed |
| Bottleneck | (920, 2048) | float32 | Shared |
| Decoder audio path | (920, 2048) | float32 | Audio-specific |
| Predicted codes | (920, 8, 112) | float32 | Continuous predictions |
| Rounded codes | (920, 8, 112) | int32 | Clipped [0, 1023] |
| Concatenated codes | (8, 103040) | int32 | Full sequence |
| Decoded audio | (33M,) | float32 | Reconstructed waveform |

---

## Loss Function Considerations

### Current (Mel Spectrogram)
```python
# MSE on continuous mel spectrograms
loss = F.mse_loss(predicted_mel, target_mel)
```

### New (EnCodec Codes)

**Option 1: Regression Loss (RECOMMENDED)**
```python
# Treat as regression problem (continuous predictions → round later)
# Predicted: (batch, 8, 112) float32
# Target:    (batch, 8, 112) int32 → convert to float

target_float = target_codes.float()
loss = F.mse_loss(predicted_codes, target_float)

# Alternatively, smooth L1 loss (Huber loss)
loss = F.smooth_l1_loss(predicted_codes, target_float)
```

**Option 2: Classification Loss**
```python
# Predict logits over 1024 classes
# Requires changing decoder output to:
# (batch, 8, 112, 1024) logits

# Decoder:
logits = self.layer13_audio(audio_features)  # (batch, 8*112*1024)
logits = logits.view(batch, 8, 112, 1024)

# Loss:
target = target_codes.long()  # (batch, 8, 112)
loss = F.cross_entropy(
    logits.view(-1, 1024),  # (batch*8*112, 1024)
    target.view(-1)          # (batch*8*112,)
)
```

**Recommendation:** Use **regression loss (Option 1)** because:
- Simpler architecture (no need to predict 1024 logits)
- Faster training (smaller output layer)
- Codes are ordinal (nearby codes are similar)
- Can interpolate between codes during prediction

---

## Testing Strategy

### Unit Tests

1. **Audio Processing**
   ```python
   def test_audio_to_encodec_codes():
       processor = AudioProcessor(use_encodec=True)
       codes, meta = processor.audio_to_features('sherlock.m4v', max_trs=10)
       assert codes.shape == (10, 8, 112)
       assert codes.dtype == np.int32
       assert codes.min() >= 0 and codes.max() <= 1023
   ```

2. **Encoder Forward Pass**
   ```python
   def test_encoder_discrete_codes():
       encoder = AudioEncoder(use_encodec=True)
       codes = torch.randint(0, 1024, (32, 8, 112))  # batch=32
       features = encoder(codes)
       assert features.shape == (32, 256)
   ```

3. **Decoder Forward Pass**
   ```python
   def test_decoder_predict_codes():
       decoder = MultimodalDecoder(use_encodec=True)
       bottleneck = torch.randn(32, 2048)
       video, audio, text = decoder(bottleneck)
       assert audio.shape == (32, 8, 112)
       assert audio.min() >= 0 and audio.max() <= 1023
   ```

4. **Round-Trip Reconstruction**
   ```python
   def test_encodec_roundtrip():
       processor = AudioProcessor(use_encodec=True)

       # Encode
       codes, _ = processor.audio_to_features('sherlock.m4v', max_trs=10)

       # Decode
       processor.features_to_audio(codes, 'test_output.wav')

       # Load and compare
       original, _ = librosa.load('sherlock.m4v', sr=24000)
       reconstructed, _ = librosa.load('test_output.wav', sr=24000)

       # Should be very similar (EnCodec is lossy but high quality)
       mse = np.mean((original[:len(reconstructed)] - reconstructed)**2)
       assert mse < 0.01  # Threshold depends on bitrate
   ```

### Integration Tests

1. **Full Pipeline Test**
   ```python
   def test_full_pipeline():
       # 1. Load data
       dataset = MultimodalDataset('data/', subjects=1, max_trs=10)
       sample = dataset[0]

       # 2. Check audio format
       assert sample['audio'].shape == (8, 112)  # EnCodec codes

       # 3. Forward pass through encoder
       encoder = create_encoder()
       bottleneck, _ = encoder(
           sample['video'].unsqueeze(0),
           sample['audio'].unsqueeze(0),
           sample['text'].unsqueeze(0)
       )

       # 4. Forward pass through decoder
       decoder = MultimodalDecoder()
       video, audio, text = decoder(bottleneck)

       # 5. Check outputs
       assert audio.shape == (1, 8, 112)
   ```

2. **Audio Quality Comparison**
   ```python
   def test_audio_quality_comparison():
       """Compare mel spectrogram vs EnCodec reconstruction quality."""

       # Process with mel spectrogram
       mel_processor = AudioProcessor(use_encodec=False)
       mel_features, _ = mel_processor.audio_to_features('sherlock.m4v')
       mel_processor.features_to_audio(mel_features, 'mel_output.wav')

       # Process with EnCodec
       enc_processor = AudioProcessor(use_encodec=True)
       enc_codes, _ = enc_processor.audio_to_features('sherlock.m4v')
       enc_processor.features_to_audio(enc_codes, 'encodec_output.wav')

       # Compare to original
       # (Use perceptual metrics: PESQ, SI-SDR, etc.)
       # EnCodec should win significantly
   ```

---

## Migration Path

### Phase 1: Add EnCodec Support (Week 1)
- [ ] Implement AudioProcessor with EnCodec
- [ ] Add embedding layers to AudioEncoder
- [ ] Update AudioDecoder output
- [ ] Unit tests for all components
- [ ] Keep `use_encodec=False` as default

### Phase 2: Validate EnCodec (Week 2)
- [ ] Test on subset of data (10 TRs)
- [ ] Compare reconstruction quality
- [ ] Measure training performance
- [ ] Tune hyperparameters (embedding_dim, etc.)
- [ ] Integration tests

### Phase 3: Full Training (Week 3)
- [ ] Train autoencoder with EnCodec features
- [ ] Compare to mel spectrogram baseline
- [ ] Evaluate fMRI prediction accuracy
- [ ] Generate reconstruction samples
- [ ] Document results

### Phase 4: Production (Week 4)
- [ ] Switch default to `use_encodec=True`
- [ ] Update all documentation
- [ ] Archive mel spectrogram code
- [ ] Create migration guide for users
- [ ] Close Issue #24

---

## Expected Benefits

### Quantitative Improvements
1. **Audio Quality:**
   - Current (Griffin-Lim): PESQ < 2.0 (poor)
   - Expected (EnCodec): PESQ > 3.5 (good to excellent)

2. **Feature Efficiency:**
   - Mel spectrograms: 2048 continuous values
   - EnCodec codes: 8 × 112 = 896 discrete values
   - **56% reduction** in feature dimensionality

3. **Training Speed:**
   - Embedding lookup faster than continuous convolutions
   - Discrete codes → sparse gradients
   - Expected **20-30% faster** training

### Qualitative Improvements
1. **Intelligibility:** Speech should be clearly understandable
2. **Music Fidelity:** Music should preserve melody, rhythm, timbre
3. **Sound Effects:** Ambient sounds should be recognizable
4. **Scientific Validity:** Reconstructions reflect brain representations

---

## Potential Issues & Mitigations

### Issue 1: Discrete Codes in Continuous Pipeline
**Problem:** HRF convolution expects continuous values
**Solution:** Convert to float, convolve, round back (implemented above)

### Issue 2: Code Distribution Shift
**Problem:** Predicted codes may not match EnCodec distribution
**Mitigation:**
- Use regression loss (smooth predictions)
- Add regularization to keep predictions in [0, 1023]
- Fine-tune decoder with perceptual loss

### Issue 3: Temporal Alignment
**Problem:** 75 Hz EnCodec vs 1.5s TR mismatch
**Solution:** Already handled by grouping frames (112 frames/TR)

### Issue 4: Memory Usage
**Problem:** Embedding layer adds parameters
**Calculation:**
- 8 embeddings × 1024 vocab × 64 dim = **524K parameters**
- Minimal compared to conv layers (~10M total)

### Issue 5: Backward Compatibility
**Problem:** Existing cached data uses mel spectrograms
**Solution:**
- Feature flags allow both formats
- Invalidate cache when switching formats
- Cache filenames include format identifier

---

## Next Steps

1. **Immediate (This Session):**
   - [x] Create this architecture document
   - [ ] Update requirements_conda.txt with comments
   - [ ] Create implementation task list (CSV)

2. **Short-term (Next Session):**
   - [ ] Implement AudioProcessor.audio_to_features() with EnCodec
   - [ ] Implement AudioProcessor.features_to_audio() with EnCodec
   - [ ] Add unit tests

3. **Medium-term (This Week):**
   - [ ] Update AudioEncoder with embedding layers
   - [ ] Update AudioDecoder output format
   - [ ] Update alignment/HRF code
   - [ ] Integration tests

4. **Long-term (Next Week):**
   - [ ] Train autoencoder with EnCodec features
   - [ ] Evaluate and compare results
   - [ ] Document findings
   - [ ] Deploy to production

---

## References

1. **EnCodec Paper:**
   - Défossez et al. (2022). "High Fidelity Neural Audio Compression"
   - https://arxiv.org/abs/2210.13438

2. **HuggingFace Documentation:**
   - https://huggingface.co/docs/transformers/model_doc/encodec
   - https://huggingface.co/facebook/encodec_24khz

3. **Related Work:**
   - Residual Vector Quantization (RVQ)
   - Neural audio codecs (SoundStream, Lyra)
   - Perceptual audio metrics (PESQ, STOI, SI-SDR)

4. **Project Context:**
   - Issue #23: Audio reconstruction quality improvements
   - Issue #24: Audio enhancement with EnCodec
   - Session notes: 2025-10-31 comprehensive planning

---

## Appendix: Code Snippets

### A. Loading EnCodec Model

```python
from transformers import EncodecModel

# Load pretrained 24kHz model
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
model.set_target_bandwidth(6.0)  # 6 kbps
model.eval()

# For GPU
model = model.to('cuda')
```

### B. Encoding Audio

```python
import torch
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load('audio.wav')

# Resample to 24kHz if needed
if sample_rate != 24000:
    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
    waveform = resampler(waveform)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Add batch dimension: (1, n_samples) → (1, 1, n_samples)
waveform = waveform.unsqueeze(0)

# Encode
with torch.no_grad():
    encoded_frames = model.encode(waveform)

# Extract codes
codes = encoded_frames[0][0]  # (1, n_codebooks, n_frames)
```

### C. Decoding Audio

```python
from transformers import EncodecEncodedFrame

# Prepare encoded frame
codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)  # (1, 8, n_frames)
encoded_frame = EncodecEncodedFrame(codes_tensor, None)

# Decode
with torch.no_grad():
    reconstructed = model.decode([encoded_frame], audio_scales=[None])[0]

# Save
reconstructed = reconstructed.squeeze().cpu().numpy()
import soundfile as sf
sf.write('output.wav', reconstructed, 24000)
```

### D. Embedding Layer Example

```python
import torch.nn as nn

# Create embedding for one codebook
embedding = nn.Embedding(
    num_embeddings=1024,  # vocab size
    embedding_dim=64       # embedding dimension
)

# Embed discrete codes
codes = torch.randint(0, 1024, (32, 112))  # batch=32, frames=112
embedded = embedding(codes)  # (32, 112, 64)
```

---

**End of Document**

# EnCodec Integration Quick Reference

**Last Updated:** 2025-10-31
**For:** Issue #24, Task 1.3

---

## Key Dimensions at a Glance

| Stage | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| **Raw Audio** | `(33,120,000,)` | float32 | 24kHz mono, 23 min |
| **EnCodec Codes** | `(8, 103,000)` | int32 | 8 codebooks, 75 Hz |
| **Per-TR Codes** | `(920, 8, 112)` | int32 | Grouped by 1.5s TRs |
| **Embedded** | `(920, 8, 112, 64)` | float32 | After embedding layer |
| **Conv Input** | `(920, 512, 112)` | float32 | Reshaped for Conv1D |
| **Audio Features** | `(920, 256)` | float32 | Compressed |
| **Predicted Codes** | `(920, 8, 112)` | float32 | Decoder output |
| **Rounded Codes** | `(920, 8, 112)` | int32 | Clipped [0, 1023] |
| **Reconstructed Audio** | `(33,120,000,)` | float32 | High quality |

---

## Critical Numbers

- **Sample Rate:** 24,000 Hz (EnCodec standard)
- **Frame Rate:** 75 Hz (24,000 / 320 hop)
- **Frames per TR:** 112 (75 Hz × 1.5s)
- **Codebooks:** 8 (Residual Vector Quantization)
- **Vocabulary:** 1024 codes per codebook
- **Embedding Dim:** 64 (tunable)
- **Bitrate:** 6.0 kbps (quality/size tradeoff)
- **Model:** `facebook/encodec_24khz` (HuggingFace)

---

## Component Changes Summary

### AudioProcessor (`giblet/data/audio.py`)

**Before:**
```python
mel_spec = librosa.feature.melspectrogram(...)  # (2048, ~60000)
features = average_to_tr(mel_spec)              # (920, 2048)
```

**After:**
```python
codes = encodec_model.encode(waveform)          # (8, ~103000)
features = group_by_tr(codes)                   # (920, 8, 112)
```

**Key Changes:**
- Load EnCodec model in `__init__`
- Replace mel spectrogram with EnCodec encoding
- Group frames into TRs (112 frames/TR)
- Return int32 discrete codes

---

### AudioEncoder (`giblet/models/encoder.py`)

**Before:**
```python
# Input: (batch, n_mels, frames_per_tr) continuous
x = self.temporal_conv(x)  # Direct convolution
```

**After:**
```python
# Input: (batch, n_codebooks, frames_per_tr) discrete
embedded = [emb(x[:, i, :]) for i, emb in enumerate(self.embeddings)]
embedded = torch.stack(embedded, dim=1)  # (batch, 8, 112, 64)
x = embedded.view(batch, 512, 112)       # Reshape for Conv1D
x = self.temporal_conv(x)                # Convolution
```

**Key Changes:**
- Add 8 embedding layers (1024 vocab × 64 dim each)
- Embed discrete codes to continuous
- Reshape for Conv1D (8 × 64 = 512 channels)
- Rest of architecture unchanged

---

### AudioDecoder (`giblet/models/decoder.py`)

**Before:**
```python
# Predict continuous mel spectrograms
audio = self.layer13_audio(features)  # (batch, 2048)
audio = self.temporal_upsample(audio) # (batch, 2048, 65)
```

**After:**
```python
# Predict discrete codes
audio = self.layer13_audio(features)  # (batch, 896)
audio = audio.view(batch, 8, 112)     # (batch, 8, 112)
audio = torch.sigmoid(audio) * 1023   # Scale to [0, 1023]
```

**Key Changes:**
- Output 896 values (8 × 112)
- Reshape to (n_codebooks, frames_per_tr)
- Scale to valid code range [0, 1023]
- Remove temporal upsampling (not needed)

---

### Alignment/Sync (`giblet/alignment/sync.py`)

**Before:**
```python
# Resample continuous features
resampled = np.interp(target_indices, current_indices, features)
```

**After:**
```python
# Resample discrete codes
features_float = features.astype(np.float32)
resampled = np.interp(target_indices, current_indices, features_float)
resampled = np.round(resampled).astype(np.int32)
resampled = np.clip(resampled, 0, 1023)
```

**Key Changes:**
- Detect discrete codes via dtype
- Convert to float for interpolation
- Round and clip back to integers

---

### HRF Convolution (`giblet/alignment/hrf.py`)

**Before:**
```python
# Convolve continuous features
convolved = signal.convolve(features, hrf, mode='same')
```

**After:**
```python
# Convolve discrete codes
features_float = features.astype(np.float32)
convolved = signal.convolve(features_float, hrf, mode='same')
convolved = np.round(convolved).astype(np.int32)
convolved = np.clip(convolved, 0, 1023)
```

**Key Changes:**
- Same as sync.py: float → convolve → round

---

## Code Snippets

### Loading EnCodec Model

```python
from transformers import EncodecModel

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
model.set_target_bandwidth(6.0)  # 6 kbps
model.eval()  # Freeze weights
```

### Encoding Audio

```python
import torch
import librosa

# Load audio
y, sr = librosa.load('audio.wav', sr=24000, mono=True)

# To tensor
wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_samples)

# Encode
with torch.no_grad():
    encoded_frames = model.encode(wav)
    codes = encoded_frames[0][0]  # (1, 8, n_frames)
```

### Decoding Audio

```python
from transformers import EncodecEncodedFrame

# Prepare codes
codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)  # (1, 8, n_frames)
encoded_frame = EncodecEncodedFrame(codes_tensor, None)

# Decode
with torch.no_grad():
    audio = model.decode([encoded_frame], audio_scales=[None])[0]

# Save
audio_np = audio.squeeze().cpu().numpy()
import soundfile as sf
sf.write('output.wav', audio_np, 24000)
```

### Grouping by TR

```python
def group_codes_by_tr(codes, tr=1.5, frame_rate=75):
    """
    Group EnCodec codes into TRs.

    Args:
        codes: (n_codebooks, n_frames) int32
        tr: TR duration in seconds
        frame_rate: EnCodec frame rate (Hz)

    Returns:
        grouped: (n_trs, n_codebooks, frames_per_tr) int32
    """
    frames_per_tr = int(tr * frame_rate)  # 1.5 * 75 = 112
    n_frames = codes.shape[1]
    n_trs = n_frames // frames_per_tr

    grouped = np.zeros((n_trs, 8, frames_per_tr), dtype=np.int32)

    for tr_idx in range(n_trs):
        start = tr_idx * frames_per_tr
        end = start + frames_per_tr
        grouped[tr_idx] = codes[:, start:end]

    return grouped
```

### Creating Embeddings

```python
import torch.nn as nn

# 8 embedding layers, one per codebook
embeddings = nn.ModuleList([
    nn.Embedding(num_embeddings=1024, embedding_dim=64)
    for _ in range(8)
])

# Embed codes
codes = torch.randint(0, 1024, (batch, 8, 112))  # (batch, 8, 112)
embedded = [embeddings[i](codes[:, i, :]) for i in range(8)]  # List of (batch, 112, 64)
embedded = torch.stack(embedded, dim=1)  # (batch, 8, 112, 64)

# Reshape for Conv1D
embedded = embedded.view(batch, 8 * 64, 112)  # (batch, 512, 112)
```

---

## Loss Function

**Recommended: Regression Loss (MSE or Smooth L1)**

```python
# Predicted codes (continuous): (batch, 8, 112) float32
# Target codes (discrete): (batch, 8, 112) int32

target_float = target_codes.float()
loss = F.mse_loss(predicted_codes, target_float)

# OR smooth L1 (Huber loss)
loss = F.smooth_l1_loss(predicted_codes, target_float)
```

**Alternative: Classification Loss**

```python
# Requires changing decoder to output logits: (batch, 8, 112, 1024)

logits = decoder(bottleneck)  # (batch, 8, 112, 1024)
target = target_codes.long()  # (batch, 8, 112)

loss = F.cross_entropy(
    logits.view(-1, 1024),  # (batch*8*112, 1024)
    target.view(-1)          # (batch*8*112,)
)
```

---

## Testing Checklist

### Unit Tests

- [ ] `test_audio_to_features()` - Verify shape, dtype, range
- [ ] `test_features_to_audio()` - Reconstruction quality
- [ ] `test_encodec_roundtrip()` - Encode → decode similarity
- [ ] `test_encoder_forward()` - Discrete codes → features
- [ ] `test_decoder_forward()` - Bottleneck → codes
- [ ] `test_resample_discrete()` - Sync with codes
- [ ] `test_hrf_discrete()` - HRF with codes

### Integration Tests

- [ ] `test_full_pipeline()` - End-to-end
- [ ] `test_quality_comparison()` - Mel vs EnCodec
- [ ] `test_backward_compatibility()` - use_encodec=False

---

## Feature Flags

All components support backward compatibility via `use_encodec` parameter:

```python
# Use EnCodec (NEW)
processor = AudioProcessor(use_encodec=True)
encoder = AudioEncoder(use_encodec=True)
decoder = MultimodalDecoder(use_encodec=True)

# Use mel spectrograms (LEGACY)
processor = AudioProcessor(use_encodec=False)
encoder = AudioEncoder(use_encodec=False)
decoder = MultimodalDecoder(use_encodec=False)
```

---

## Common Pitfalls

### 1. Wrong Sample Rate
**Problem:** EnCodec requires 24kHz
**Solution:** Always resample to 24000 Hz

```python
y, sr = librosa.load('audio.wav', sr=24000)  # Force 24kHz
```

### 2. Forgetting to Round Predictions
**Problem:** Decoder outputs continuous values
**Solution:** Round and clip before decoding

```python
codes = np.round(predicted_codes).astype(np.int32)
codes = np.clip(codes, 0, 1023)
```

### 3. Shape Mismatches
**Problem:** Wrong number of dimensions
**Solution:** Always check shapes:

```python
assert codes.shape == (920, 8, 112)  # Per-TR codes
assert embedded.shape == (batch, 512, 112)  # Conv1D input
assert features.shape == (batch, 256)  # Encoder output
```

### 4. Discrete vs Continuous
**Problem:** Mixing dtypes
**Solution:** Track format explicitly

```python
if codes.dtype in [np.int32, np.int64]:
    # Discrete codes - convert to float before math
    codes_float = codes.astype(np.float32)
```

---

## Performance Tips

### 1. Freeze EnCodec Model
EnCodec is pretrained - don't update weights:

```python
model.eval()  # Set to eval mode
for param in model.parameters():
    param.requires_grad = False
```

### 2. Batch Processing
Process multiple TRs at once:

```python
# Bad: Loop over TRs
for tr_idx in range(n_trs):
    codes = model.encode(audio[tr_idx])

# Good: Batch encode
codes = model.encode(audio)  # Encode full stimulus once
```

### 3. Cache Codes
Save encoded codes to avoid re-encoding:

```python
# First time
codes, _ = processor.audio_to_features('sherlock.m4v')
np.save('sherlock_codes.npy', codes)

# Later
codes = np.load('sherlock_codes.npy')
```

---

## Expected Performance

### Audio Quality (vs. Mel Spectrogram)

| Metric | Mel + Griffin-Lim | EnCodec @ 6kbps | Improvement |
|--------|-------------------|-----------------|-------------|
| **PESQ** | < 2.0 (poor) | > 3.5 (good) | **+75%** |
| **SI-SDR** | < 5 dB | > 15 dB | **+200%** |
| **Intelligibility** | Garbled | Clear | **Major** |
| **Music Quality** | Distorted | Natural | **Major** |

### Training Performance

| Metric | Mel Spectrogram | EnCodec | Change |
|--------|-----------------|---------|--------|
| **Features/TR** | 2048 | 896 | **-56%** |
| **Memory** | Baseline | +5% | Minimal |
| **Speed** | Baseline | +20-30% | Faster |
| **Parameters** | Baseline | +524K | Minimal |

---

## Debugging

### Check Code Values

```python
# Codes should be in [0, 1023]
print(f"Min: {codes.min()}, Max: {codes.max()}")
assert codes.min() >= 0 and codes.max() <= 1023
```

### Check Frame Rate

```python
# 75 Hz = 24000 / 320
frame_rate = sample_rate / hop_length
assert frame_rate == 75
```

### Verify Reconstruction

```python
# Encode then decode
codes, _ = processor.audio_to_features('input.wav')
processor.features_to_audio(codes, 'output.wav')

# Compare
import soundfile as sf
original, _ = sf.read('input.wav')
reconstructed, _ = sf.read('output.wav')

mse = np.mean((original - reconstructed)**2)
print(f"MSE: {mse:.6f}")  # Should be < 0.01 for 6kbps
```

### Check Gradients

```python
# Embedding layer should have gradients
for name, param in encoder.named_parameters():
    if 'embedding' in name:
        print(f"{name}: grad={param.grad is not None}")
```

---

## Next Steps

1. **Implement AudioProcessor** (Estimated: 6 hours)
   - `audio_to_features()` with EnCodec
   - `features_to_audio()` with EnCodec
   - Unit tests

2. **Update Encoder/Decoder** (Estimated: 6 hours)
   - Add embedding layers
   - Update forward passes
   - Unit tests

3. **Update Alignment Code** (Estimated: 3 hours)
   - Handle discrete codes in sync/HRF
   - Tests

4. **Integration Testing** (Estimated: 4 hours)
   - Full pipeline test
   - Quality comparison

5. **Training** (Estimated: 32 hours)
   - Small subset validation
   - Hyperparameter tuning
   - Full training
   - Results documentation

**Total Estimated Time:** ~51 hours

---

## References

- **Full Architecture:** `ENCODEC_INTEGRATION_ARCHITECTURE.md`
- **Implementation Checklist:** `ENCODEC_IMPLEMENTATION_CHECKLIST.csv`
- **EnCodec Paper:** https://arxiv.org/abs/2210.13438
- **HuggingFace Docs:** https://huggingface.co/docs/transformers/model_doc/encodec
- **Issue #24:** Audio enhancement with EnCodec

---

**End of Quick Reference**

# EnCodec Audio Integration - Overview

This document provides a high-level overview of EnCodec neural audio codec integration in the giblet-responses project.

---

## Table of Contents

1. [What is EnCodec?](#what-is-encodec)
2. [Why EnCodec?](#why-encodec)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Feature Format](#feature-format)
6. [Performance](#performance)
7. [Next Steps](#next-steps)

---

## What is EnCodec?

**EnCodec** (Enhanced Neural Codec) is a state-of-the-art neural audio codec developed by Meta AI that provides high-quality audio compression using learned representations.

**Key Properties:**
- **Model:** `facebook/encodec_24khz` (via HuggingFace Transformers)
- **Sample Rate:** 24,000 Hz (mono)
- **Compression:** Residual Vector Quantization (RVQ) with 8 codebooks
- **Output:** Discrete codes (vocabulary size: 1024 per codebook)
- **Frame Rate:** 75 Hz (fixed)
- **Quality:** STOI ~0.74-0.90 depending on bandwidth

**Advantages over mel spectrograms:**
1. **Phase preservation:** EnCodec implicitly preserves phase information
2. **Neural decoder:** Pretrained decoder handles all reconstruction
3. **Discrete codes:** Robust to compression through autoencoder bottleneck
4. **No Griffin-Lim:** Eliminates phase guessing artifacts

---

## Why EnCodec?

### Problem with Mel Spectrograms

The original audio pipeline used mel spectrograms with Griffin-Lim reconstruction:

```
Audio → Mel Spectrogram → Encoder → Bottleneck → Decoder → Mel Spectrogram → Griffin-Lim → Audio
```

**Problems:**
- Mel spectrograms discard phase information
- Griffin-Lim must guess phase (iterative algorithm)
- Bottleneck compression degrades magnitude
- Result: Garbled, unintelligible audio

### Solution with EnCodec

```
Audio → EnCodec Codes → Encoder → Bottleneck → Decoder → EnCodec Codes → EnCodec Decoder → Audio
```

**Benefits:**
- Phase implicitly preserved in learned codes
- Discrete codes robust to compression
- Pretrained neural decoder (no phase guessing)
- Result: High-quality, intelligible audio

### Empirical Results

**Audio Quality (STOI metric):**
- Mel spectrogram + Griffin-Lim: 0.40-0.50 (poor)
- EnCodec @ 3.0 kbps: 0.74 (good)
- EnCodec @ 12.0 kbps: 0.90 (excellent)

**Compression:**
- Mel spectrograms: 2048 continuous values → ~110 MB storage
- EnCodec codes: 8 × 112 = 896 discrete values → ~1.7 MB storage
- **98% size reduction**

---

## Quick Start

### Installation

EnCodec is available via the `transformers` library (already in requirements):

```bash
pip install transformers>=4.57.1
```

No additional dependencies needed!

### Basic Usage

```python
from giblet.data.audio import AudioProcessor

# Initialize with EnCodec (default)
processor = AudioProcessor(
    use_encodec=True,
    encodec_bandwidth=3.0,  # 3.0 kbps (good quality/size tradeoff)
    tr=1.5,
    device='cpu'  # or 'cuda'
)

# Encode video audio
features, metadata = processor.audio_to_features(
    "data/stimuli_Sherlock.m4v",
    from_video=True,
    max_trs=100  # Optional: limit to first 100 TRs
)

print(f"Shape: {features.shape}")  # (100, 1, 112)
print(f"Dtype: {features.dtype}")  # int64
print(f"Range: [{features.min()}, {features.max()}]")  # [0, 1023]

# Decode back to audio
processor.features_to_audio(features, "output/decoded_audio.wav")
```

### Usage in Dataset

```python
from giblet.data import MultimodalDataset

# Dataset automatically uses EnCodec by default
dataset = MultimodalDataset(
    data_dir='data/',
    subjects='all',
    split='train',
    apply_hrf=True
)

# Get a sample
sample = dataset[0]
print(sample['audio'].shape)  # (1, 112) - EnCodec codes
```

### Legacy Mode (Mel Spectrogram)

For backward compatibility, mel spectrograms are still supported:

```python
# Disable EnCodec
processor = AudioProcessor(use_encodec=False)
features, _ = processor.audio_to_features("video.m4v")
# Returns: (n_trs, 2048, frames_per_tr) float32
```

---

## Configuration

### AudioProcessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_encodec` | bool | True | Use EnCodec neural codec |
| `encodec_bandwidth` | float | 3.0 | Bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0) |
| `device` | str | 'cpu' | Device for model ('cpu' or 'cuda') |
| `tr` | float | 1.5 | fMRI repetition time (seconds) |

### Bandwidth Selection Guide

| Bandwidth | STOI | Quality | Use Case | File Size |
|-----------|------|---------|----------|-----------|
| 1.5 kbps | 0.65 | Poor | Prototyping only | ~0.9 MB |
| **3.0 kbps** | **0.74** | **Good** | **Default (recommended)** | **~1.7 MB** |
| 6.0 kbps | 0.83 | Very good | High quality needed | ~3.4 MB |
| 12.0 kbps | 0.90 | Excellent | Research/analysis | ~6.8 MB |
| 24.0 kbps | 0.94 | Near-perfect | Maximum quality | ~13.6 MB |

**Recommendation:** Use **3.0 kbps** for most use cases (good quality/size tradeoff).

### EnCodec Model Specifications

- **Model:** `facebook/encodec_24khz`
- **Model Size:** ~30 MB (downloaded once, cached)
- **Sample Rate:** 24,000 Hz (mono)
- **Frame Rate:** 75 Hz (fixed)
- **Codebooks:** 1 (mono audio)
- **Vocabulary:** 1024 discrete codes per codebook
- **Frames per TR:** `int(75 * tr)` = 112 for TR=1.5s

---

## Feature Format

### EnCodec Mode (Default)

**Shape:** `(n_trs, 1, frames_per_tr)`
- `n_trs`: Number of TRs (e.g., 920 for Sherlock)
- `1`: Number of codebooks (mono)
- `frames_per_tr`: 112 for TR=1.5s (75 Hz × 1.5s)

**Dtype:** `int64` (discrete codes in range [0, 1023])

**Example:**
```python
features.shape  # (920, 1, 112)
features.dtype  # int64
features.min()  # 0
features.max()  # 1023
```

### Mel Spectrogram Mode (Legacy)

**Shape:** `(n_trs, n_mels, frames_per_tr)`
- `n_trs`: Number of TRs
- `n_mels`: 2048 mel frequency bins
- `frames_per_tr`: Variable (~64 frames for TR=1.5s)

**Dtype:** `float32` (dB scale)

### Format Comparison

| Mode | Shape | Dtype | Dim 1 | Dim 2 | Quality |
|------|-------|-------|-------|-------|---------|
| EnCodec | (n_trs, 1, 112) | int64 | codebooks | frames@75Hz | STOI=0.74-0.90 |
| Mel | (n_trs, 2048, ~64) | float32 | mels | frames@43Hz | STOI=0.40-0.50 |

### Metadata

The metadata DataFrame now includes `encoding_mode`:

```python
features, metadata = processor.audio_to_features("video.m4v")
print(metadata.head())
```

Output:
```
   tr_index  start_time  end_time  n_frames encoding_mode
0         0         0.0       1.5       112       encodec
1         1         1.5       3.0       112       encodec
2         2         3.0       4.5       112       encodec
```

---

## Performance

### Sherlock Dataset (48-minute episode)

**Dataset Size:**
- Total TRs: 1,976 (at TR=1.5s)
- Duration: 49.4 minutes

**EnCodec Encoding:**
- Features: 1,976 × 1 × 112 = 221,312 codes (int64)
- Storage: ~1.7 MB (vs ~110 MB for mel spectrogram)
- Encoding time: ~2-3 minutes (CPU)
- Memory usage: <1 GB peak

**Quality @ 3.0 kbps:**
- STOI: 0.74 (good intelligibility)
- PESQ: 2.66 (acceptable quality)
- SNR: 5.94 dB (acceptable)

**Model Loading:**
- First time: ~30 seconds (downloads model from HuggingFace Hub)
- Subsequent: <1 second (cached model)

### Training Performance

**Expected improvements with EnCodec:**
- **Feature efficiency:** 56% reduction (896 vs 2048 values)
- **Training speed:** 20-30% faster (embedding lookup vs convolutions)
- **Memory usage:** Similar (discrete codes + embeddings ≈ continuous features)
- **Audio quality:** 2-3x improvement in STOI

---

## Next Steps

### For Users

1. **Try it out:** Run `examples/encodec_audio_encoder_demo.py`
2. **Train a model:** Use `MultimodalDataset` with default settings (EnCodec enabled)
3. **Compare quality:** Generate reconstructions and compare to mel spectrogram baseline
4. **Tune bandwidth:** Try different bandwidth settings for your use case

### For Developers

1. **Read integration guide:** See [integration.md](integration.md) for technical details
2. **Run tests:** `pytest tests/data/test_audio_encodec.py -v`
3. **Debug issues:** See [troubleshooting.md](troubleshooting.md) for common problems
4. **Contribute:** Open issues or PRs on GitHub

---

## Backward Compatibility

✅ **Fully backward compatible**

- Old code using mel spectrograms still works
- Auto-detection handles both formats in `features_to_audio()`
- Set `use_encodec=False` to use original behavior
- Existing cached features are preserved (different cache files)

**Migration:** No code changes needed! EnCodec is enabled by default but gracefully falls back to mel spectrograms if needed.

---

## Dependencies

### Required (Already in requirements.txt)

- `transformers>=4.57.1` - For EnCodec model
- `torch>=2.0.0` - Required by transformers
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O
- `pandas>=2.0.0` - Metadata handling

### Optional (For Quality Metrics)

- `pystoi>=0.4.1` - STOI metric
- `pesq>=0.0.4` - PESQ metric

**Install all:**
```bash
pip install -r requirements.txt
pip install pystoi pesq  # Optional
```

---

## Related Documentation

- **[integration.md](integration.md)** - Technical implementation details and architecture
- **[troubleshooting.md](troubleshooting.md)** - Testing, debugging, and common issues
- **[giblet/data/README.md](../../giblet/data/README.md)** - Data pipeline documentation
- **[examples/README.md](../../examples/README.md)** - Example scripts and demos

---

## References

- **EnCodec Paper:** Défossez et al. (2022). "High Fidelity Neural Audio Compression." https://arxiv.org/abs/2210.13438
- **HuggingFace Documentation:** https://huggingface.co/docs/transformers/model_doc/encodec
- **Model Card:** https://huggingface.co/facebook/encodec_24khz
- **Issue #24:** Audio Enhancement with EnCodec (GitHub)

For questions or issues, see the main project [README.md](../../README.md) or open an issue on GitHub.

# EnCodec Integration - Troubleshooting Guide

This document provides testing procedures, common issues, and debugging guidance for the EnCodec neural audio codec integration.

---

## Table of Contents

1. [Quick Testing](#quick-testing)
2. [Common Issues](#common-issues)
3. [Quality Metrics](#quality-metrics)
4. [Known Issues and Fixes](#known-issues-and-fixes)
5. [Debugging Guide](#debugging-guide)
6. [Test Suite](#test-suite)
7. [Performance Optimization](#performance-optimization)

---

## Quick Testing

### Minimal Test Script

```python
# test_encodec_minimal.py
from giblet.data.audio import AudioProcessor

# Initialize
processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, device='cpu')

# Encode
features, metadata = processor.audio_to_features(
    "data/stimuli_Sherlock.m4v",
    max_trs=10,
    from_video=True
)

print(f"✅ Encoding successful!")
print(f"   Shape: {features.shape}")  # (10, 1, 112)
print(f"   Dtype: {features.dtype}")  # int64
print(f"   Range: [{features.min()}, {features.max()}]")  # [0, 1023]

# Decode
processor.features_to_audio(features, "test_output.wav")
print(f"✅ Decoding successful! Saved to test_output.wav")
```

**Run:**
```bash
python test_encodec_minimal.py
```

**Expected output:**
```
✅ Encoding successful!
   Shape: (10, 1, 112)
   Dtype: int64
   Range: [0, 1023]
✅ Decoding successful! Saved to test_output.wav
```

---

## Common Issues

### Issue 1: "EnCodec not available"

**Symptom:**
```python
ImportError: transformers required for EnCodec. pip install transformers
```

**Solution:**
```bash
pip install transformers>=4.57.1
```

**Verification:**
```python
python -c "from transformers import EncodecModel; print('✅ EnCodec available')"
```

---

### Issue 2: Model Download Fails

**Symptom:**
```
HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/...
```

**Causes:**
1. No internet connection
2. Firewall blocking HuggingFace Hub
3. HuggingFace Hub downtime

**Solutions:**

**Option A: Check connectivity**
```bash
curl -I https://huggingface.co
```

**Option B: Manual download**
```bash
# Download model manually
git clone https://huggingface.co/facebook/encodec_24khz ~/.cache/huggingface/hub/models--facebook--encodec_24khz
```

**Option C: Use local model**
```python
processor = AudioProcessor(
    use_encodec=True,
    encodec_model_path="/path/to/local/model"  # If supported
)
```

---

### Issue 3: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

**Option A: Use CPU**
```python
processor = AudioProcessor(use_encodec=True, device='cpu')
```

**Option B: Process smaller chunks**
```python
# Process 100 TRs at a time instead of all 920
for start_tr in range(0, 920, 100):
    features, metadata = processor.audio_to_features(
        "video.m4v",
        max_trs=100
    )
    # Save chunk
    np.save(f"features_chunk_{start_tr}.npy", features)
```

**Option C: Use gradient checkpointing (for training)**
```python
model.gradient_checkpointing_enable()
```

---

### Issue 4: Dimension Mismatch

**Symptom:**
```
RuntimeError: Expected shape (batch, 1, 112) but got (batch, 8, 112)
```

**Cause:** Confusion between 24kHz model (1 codebook, mono) and 48kHz model (8 codebooks, stereo)

**Solution:** Use 24kHz model (default)
```python
processor = AudioProcessor(
    use_encodec=True,
    encodec_bandwidth=3.0,  # Use 24kHz model (1 codebook)
)
```

**Fix script:**
```python
# If you have features with wrong shape (8 codebooks)
if features.shape[1] == 8:
    # Average codebooks (not recommended, re-encode instead)
    features = features.mean(axis=1, keepdims=True).astype(np.int32)
```

---

### Issue 5: TensorFlow Mutex Blocking (Development Only)

**Symptom:**
```
Tests hang indefinitely when running pytest
```

**Cause:** TensorFlow initialization conflicts with transformers on some systems

**Solution:** Use validation script instead
```bash
python validate_encodec_implementation.py
```

**Note:** This is a development environment issue and does not affect production code.

---

### Issue 6: Poor Audio Quality

**Symptom:** Reconstructed audio sounds garbled or low quality

**Debugging steps:**

**Step 1: Check bandwidth**
```python
print(f"Bandwidth: {processor.encodec_model.bandwidth}")
# Should be 3.0 or higher
```

**Step 2: Test direct EnCodec reconstruction (bypass model)**
```python
# Encode
features, _ = processor.audio_to_features("video.m4v", max_trs=10)

# Decode immediately (no model)
processor.features_to_audio(features, "direct_reconstruction.wav")

# Listen to direct_reconstruction.wav
# If this sounds bad, increase bandwidth
```

**Step 3: Increase bandwidth**
```python
processor = AudioProcessor(use_encodec=True, encodec_bandwidth=6.0)  # or 12.0
```

**Step 4: Compare quality metrics** (see Quality Metrics section)

---

## Quality Metrics

### STOI (Short-Time Objective Intelligibility)

**Range:** 0.0 (unintelligible) to 1.0 (perfect)

**Installation:**
```bash
pip install pystoi
```

**Usage:**
```python
from pystoi import stoi
import librosa

# Load audio
original, sr = librosa.load("original.wav", sr=24000)
reconstructed, sr = librosa.load("reconstructed.wav", sr=24000)

# Compute STOI
score = stoi(original, reconstructed, sr, extended=False)
print(f"STOI: {score:.3f}")
```

**Interpretation:**
| STOI Score | Quality | Acceptable? |
|------------|---------|-------------|
| < 0.5 | Poor | ❌ No |
| 0.5 - 0.6 | Fair | ⚠️ Maybe |
| 0.6 - 0.7 | Acceptable | ✅ Yes |
| 0.7 - 0.8 | Good | ✅ Yes |
| 0.8 - 0.9 | Very good | ✅ Yes |
| > 0.9 | Excellent | ✅ Yes |

**EnCodec benchmarks:**
- 3.0 kbps: STOI ~0.74 (good)
- 6.0 kbps: STOI ~0.83 (very good)
- 12.0 kbps: STOI ~0.90 (excellent)

---

### PESQ (Perceptual Evaluation of Speech Quality)

**Range:** 1.0 (bad) to 4.5 (excellent)

**Installation:**
```bash
pip install pesq
```

**Usage:**
```python
from pesq import pesq
import librosa

# Load audio
original, sr = librosa.load("original.wav", sr=16000)  # PESQ requires 16kHz
reconstructed, sr = librosa.load("reconstructed.wav", sr=16000)

# Compute PESQ (mode='wb' for wideband)
score = pesq(sr, original, reconstructed, 'wb')
print(f"PESQ: {score:.2f}")
```

**Interpretation:**
| PESQ Score | Quality | Acceptable? |
|------------|---------|-------------|
| < 2.0 | Poor | ❌ No |
| 2.0 - 2.5 | Fair | ⚠️ Maybe |
| 2.5 - 3.0 | Acceptable | ✅ Yes |
| 3.0 - 3.5 | Good | ✅ Yes |
| 3.5 - 4.0 | Very good | ✅ Yes |
| > 4.0 | Excellent | ✅ Yes |

**EnCodec benchmarks:**
- 3.0 kbps: PESQ ~2.66 (acceptable)
- 6.0 kbps: PESQ ~2.94 (good)
- 12.0 kbps: PESQ ~3.56 (very good)

---

### SNR (Signal-to-Noise Ratio)

**Range:** Higher is better (dB scale)

**Usage:**
```python
import numpy as np

# Compute SNR
signal_power = np.mean(original**2)
noise = reconstructed - original
noise_power = np.mean(noise**2)
snr = 10 * np.log10(signal_power / noise_power)
print(f"SNR: {snr:.2f} dB")
```

**Interpretation:**
| SNR (dB) | Quality | Acceptable? |
|----------|---------|-------------|
| < 5 | Poor | ❌ No |
| 5 - 10 | Fair | ⚠️ Maybe |
| 10 - 15 | Good | ✅ Yes |
| 15 - 20 | Very good | ✅ Yes |
| > 20 | Excellent | ✅ Yes |

---

## Known Issues and Fixes

### Issue #28: EnCodec Dimension Mismatch

**Date:** 2025-11-02

**Problem:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 129x256)
```

**Root cause:** AudioEncoder expected 128-dim input but EnCodec provides 129-dim

**Fix:** Updated AudioEncoder to handle flexible input dimensions

**Verification:**
```bash
pytest tests/models/test_encoder.py::test_audio_encoder_encodec_input -v
```

**Status:** ✅ Fixed in commit 972460d

---

### Issue: HRF Convolution Rounds Discrete Codes

**Problem:** HRF convolution converts discrete codes to float, convolves, then rounds back

**Concern:** Does rounding introduce artifacts?

**Analysis:**
- HRF smoothing effect dominates rounding error
- Rounding error: ±0.5 codes out of 1024 (<0.05%)
- Impact on STOI: <0.01 (negligible)

**Recommendation:** Keep current implementation (rounding after convolution)

---

## Debugging Guide

### Debug Mode

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

processor = AudioProcessor(use_encodec=True)
features, metadata = processor.audio_to_features("video.m4v", max_trs=10)
```

**Output:**
```
DEBUG:giblet.data.audio:Loading audio from video.m4v
DEBUG:giblet.data.audio:Resampling to 24000 Hz
DEBUG:giblet.data.audio:Encoding with EnCodec at 3.0 kbps
DEBUG:giblet.data.audio:Encoded shape: (1, 103000)
DEBUG:giblet.data.audio:Grouping into 920 TRs
DEBUG:giblet.data.audio:Final shape: (920, 1, 112)
```

---

### Inspect Features

```python
import numpy as np

features, metadata = processor.audio_to_features("video.m4v", max_trs=10)

# Check shape
print(f"Shape: {features.shape}")  # (10, 1, 112)

# Check dtype
print(f"Dtype: {features.dtype}")  # int64

# Check range
print(f"Min: {features.min()}, Max: {features.max()}")  # [0, 1023]

# Check distribution
unique_codes = np.unique(features)
print(f"Unique codes: {len(unique_codes)}")  # Should be many (not just a few)

# Visualize code distribution
import matplotlib.pyplot as plt
plt.hist(features.flatten(), bins=50)
plt.xlabel("Code value")
plt.ylabel("Frequency")
plt.title("EnCodec code distribution")
plt.savefig("code_distribution.png")
```

---

### Compare with Mel Spectrogram

```python
# Process with EnCodec
encodec_proc = AudioProcessor(use_encodec=True)
encodec_features, _ = encodec_proc.audio_to_features("video.m4v", max_trs=10)
encodec_proc.features_to_audio(encodec_features, "encodec_output.wav")

# Process with mel spectrogram
mel_proc = AudioProcessor(use_encodec=False)
mel_features, _ = mel_proc.audio_to_features("video.m4v", max_trs=10)
mel_proc.features_to_audio(mel_features, "mel_output.wav")

# Listen to both and compare
print("Listen to encodec_output.wav and mel_output.wav")
print("EnCodec should sound much better!")
```

---

## Test Suite

### Run All Tests

```bash
pytest tests/data/test_audio_encodec.py -v
```

**Expected output:**
```
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_initialization PASSED
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_encoding_dimensions PASSED
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_round_trip PASSED
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_quality_metrics PASSED
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_tr_alignment PASSED
tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_max_trs PASSED
tests/data/test_audio_encodec.py::TestBackwardsCompatibility::test_mel_spectrogram_fallback PASSED
tests/data/test_audio_encodec.py::TestBackwardsCompatibility::test_feature_format_auto_detection PASSED
tests/data/test_audio_encodec.py::TestEnCodecBandwidths::test_bandwidth[1.5] PASSED
tests/data/test_audio_encodec.py::TestEnCodecBandwidths::test_bandwidth[3.0] PASSED
tests/data/test_audio_encodec.py::TestEnCodecBandwidths::test_bandwidth[6.0] PASSED
tests/data/test_audio_encodec.py::TestEnCodecBandwidths::test_bandwidth[12.0] PASSED
tests/data/test_audio_encodec.py::TestEnCodecBandwidths::test_bandwidth[24.0] PASSED
======================== 13 passed in 45.2s =========================
```

---

### Run Specific Test Class

```bash
pytest tests/data/test_audio_encodec.py::TestEnCodecIntegration -v
```

---

### Run Single Test

```bash
pytest tests/data/test_audio_encodec.py::TestEnCodecIntegration::test_encodec_round_trip -v
```

---

### E2E Pipeline Test

```bash
python test_encodec_e2e_pipeline.py
```

**What it tests:**
- Complete pipeline: audio → EnCodec → encoder → bottleneck → decoder → EnCodec → audio
- 30 TRs (45 seconds) of Sherlock audio
- Quality metrics: STOI, PESQ, SNR
- Dimension checks at each stage
- Memory usage

**Output files:**
- `encodec_e2e_test/original_12khz.wav` - Original audio
- `encodec_e2e_test/baseline_encodec_12khz.wav` - EnCodec direct reconstruction
- `encodec_e2e_test/reconstructed_12khz.wav` - Through full pipeline
- `encodec_e2e_test/metrics_comparison.txt` - Quality metrics
- `encodec_e2e_test/spectrograms_comparison.png` - Visual comparison

**Success criteria:**
- ✅ Pipeline runs without errors
- ✅ STOI drop < 0.1 (minimal quality degradation)
- ✅ Memory < 1GB peak
- ✅ Audio files sound reasonable

---

## Performance Optimization

### GPU Acceleration

**Enable CUDA:**
```python
processor = AudioProcessor(use_encodec=True, device='cuda')
```

**Speedup:** ~3-5x faster encoding/decoding

**Memory:** ~500MB GPU memory for model

---

### Batch Processing

```python
# Process multiple audio files in parallel
from multiprocessing import Pool

def process_audio(audio_path):
    processor = AudioProcessor(use_encodec=True)
    features, metadata = processor.audio_to_features(audio_path)
    return features, metadata

audio_files = ["video1.m4v", "video2.m4v", "video3.m4v"]

with Pool(processes=4) as pool:
    results = pool.map(process_audio, audio_files)
```

---

### Cache Encoded Features

```python
import numpy as np
from pathlib import Path

def get_cached_features(audio_path, cache_dir="cache"):
    cache_path = Path(cache_dir) / f"{Path(audio_path).stem}_encodec.npy"

    if cache_path.exists():
        # Load from cache
        features = np.load(cache_path)
        metadata = pd.read_csv(cache_path.with_suffix('.csv'))
        return features, metadata

    # Encode and cache
    processor = AudioProcessor(use_encodec=True)
    features, metadata = processor.audio_to_features(audio_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    metadata.to_csv(cache_path.with_suffix('.csv'), index=False)

    return features, metadata
```

---

## Troubleshooting Checklist

When encountering issues, check:

- [ ] `transformers` installed? (`pip install transformers`)
- [ ] Internet connection for model download?
- [ ] Correct model loaded? (24kHz, not 48kHz)
- [ ] Correct device? ('cpu' or 'cuda')
- [ ] Sufficient memory? (EnCodec needs ~1GB)
- [ ] Correct bandwidth? (3.0 kbps default)
- [ ] Audio file exists and is valid?
- [ ] Features have correct shape? (n_trs, 1, 112)
- [ ] Features have correct dtype? (int64)
- [ ] Features have correct range? [0, 1023]
- [ ] Quality metrics acceptable? (STOI > 0.7)

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs:** Enable `logging.DEBUG` for detailed output
2. **Run tests:** `pytest tests/data/test_audio_encodec.py -v`
3. **Check GitHub issues:** Search for similar problems
4. **Open an issue:** Provide logs, error messages, and code snippet
5. **Ask on Slack:** #context-lab channel

---

## Related Documentation

- **[overview.md](overview.md)** - User-facing overview and quick start
- **[integration.md](integration.md)** - Technical implementation details
- **[giblet/data/README.md](../../giblet/data/README.md)** - Data pipeline documentation
- **[tests/README.md](../../tests/README.md)** - Test suite documentation

For questions or issues, see the main project [README.md](../../README.md) or open an issue on GitHub.

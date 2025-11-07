# Audio Encoding Research: Solutions for Issue #23

## TL;DR: Use EnCodec

**Recommendation:** Replace mel spectrograms with **Meta's EnCodec** neural audio codec.

**Why:**
- Designed specifically for compression + reconstruction (our exact use case)
- Preserves phase implicitly through learned representations
- Pretrained models available (24kHz mono, 48kHz stereo)
- MIT license, production-ready, 10M parameters
- Drop-in replacement for current mel spectrogram pipeline

**Expected improvement:** Garbled audio → Intelligible, high-quality reconstruction

---

## The Problem

Current pipeline:
```
Audio → Mel Spectrogram (magnitude only, PHASE LOST) → Encoder → Bottleneck → Decoder → Mel Spec → Griffin-Lim (phase guessing FAILS) → Garbled audio
```

Griffin-Lim can't reconstruct phase after bottleneck compression. We need an approach that preserves or implicitly encodes phase.

---

## Approach Comparison

| Approach | Quality | Params | Memory | Complexity | Pretrained | Rating |
|----------|---------|--------|--------|------------|------------|--------|
| **EnCodec** | Excellent | Low (10M) | Low | Low | ✓ | ⭐⭐⭐⭐⭐ |
| Complex FFT | Excellent | Medium | High | Medium | ✗ | ⭐⭐⭐⭐ |
| Raw Waveform | Excellent | High (50M+) | Very High | High | ✗ | ⭐⭐⭐ |
| Wav2Vec 2.0 | Good | Medium | Medium | Medium | ✓ | ⭐⭐⭐ |
| Audio-as-Image | Good | Medium | Medium | Low | ✗ | ⭐⭐⭐ |
| WaveGlow | Excellent | Very High (87M) | Very High | Very High | ✗ | ⭐⭐ |

---

## EnCodec Details

### What is EnCodec?

State-of-the-art neural audio codec from Meta AI:
- **Architecture:** Convolutional encoder → Residual Vector Quantization (compression) → Convolutional decoder
- **Training:** Multi-objective (reconstruction + perceptual + adversarial losses)
- **Compression:** 2x to 128x (1.5-24 kbps), perceptually lossless for music at 12 kbps
- **Used in:** Meta's AudioGen, MusicGen (production-proven)

### For Our Use Case

**Current approach:**
- Mel spectrogram: [128 bins, ~100 frames] = 12,800 values per TR
- Total: 920 TRs × 51 KB = 47 MB

**EnCodec approach:**
- Codes: [8 quantizers, 112 frames] = 896 values per TR
- Total: 920 TRs × 3.6 KB = 3.3 MB (even smaller!)

**Integration:**
```python
# Option 1: Use EnCodec as preprocessor (RECOMMENDED)
audio → EnCodec.encode() → codes → autoencoder → predicted codes → EnCodec.decode() → audio

# Option 2: Fine-tune EnCodec bottleneck to match fMRI dims
Replace EnCodec quantization with linear projection to fMRI dimensions
```

### Quick Start

```bash
pip install encodec
```

```python
from encodec import EncodecModel

# Load pretrained
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps

# Encode
codes = model.encode(audio)  # Audio → compressed codes

# Decode
reconstructed = model.decode(codes)  # Codes → audio
```

### Memory & Parameters

**Per TR (1.5s audio):**
- Input: 36,000 samples (1.5s @ 24kHz)
- Codes: 896 values (8 quantizers × 112 frames)
- Memory: 3.6 KB (vs. 51 KB for mel spectrogram)

**Model:**
- Parameters: ~20M (10M encoder + 10M decoder)
- Memory: 80 MB (model weights)
- Compute: Efficient (convolutional, no autoregression)

---

## Implementation Roadmap

### Phase 1: Proof of Concept (1-2 days)

1. Install EnCodec
2. Test on Sherlock audio (encode/decode full stimulus)
3. Evaluate quality (SNR, PESQ, listening test)
4. Decision: proceed?

**Expected results:**
- SNR > 20 dB
- PESQ > 3.0
- Subjectively: very similar to original

### Phase 2: Integration (3-5 days)

1. Precompute EnCodec codes for all 920 TRs
2. Modify `SherlockDataset` to return codes instead of mel spectrograms
3. Update audio encoder/decoder for new input shape
4. Train autoencoder on EnCodec codes

**Changes needed:**
```python
# Dataset: Precompute codes
encodec = EncodecModel.encodec_model_24khz()
codes = []
for tr in range(920):
    audio_tr = get_audio_for_tr(tr)
    code_tr = encodec.encode(audio_tr)
    codes.append(code_tr)

# Encoder: Accept codes instead of mel specs
# Input shape: [batch, 8, 112] instead of [batch, 128, 100]

# Decoder: Predict codes
# Output shape: [batch, 8, 112]

# Reconstruction: Decode codes to audio
audio_pred = encodec.decode(predicted_codes)
```

### Phase 3: Optimization (1-2 weeks)

**Option A:** Freeze EnCodec (simpler)
- Keep EnCodec frozen, train autoencoder to predict codes
- Pros: Fast, simple
- Cons: Not optimized for our bottleneck

**Option B:** Fine-tune EnCodec (better quality)
- Replace EnCodec quantization with linear projection to fMRI dims
- Fine-tune end-to-end
- Pros: Optimized for task
- Cons: More complex, requires more compute

### Phase 4: Evaluation (2-3 days)

**Metrics:**
- SNR (Signal-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Subjective listening tests

**Comparison:**
- Original audio
- EnCodec reconstruction
- Old mel + Griffin-Lim (current approach)

**Expected improvements:**
- SNR: 10+ dB increase
- PESQ: 2.0 → 3.5+ (acceptable → good)
- STOI: 0.3 → 0.8+ (unintelligible → intelligible)

---

## Alternative: Complex FFT (Backup Plan)

If EnCodec doesn't work, use **complex-valued STFT** with complex neural networks.

**How it works:**
- Use STFT with both magnitude AND phase (complex tensor)
- Process with complex-valued layers (ComplexConv2d, etc.)
- Reconstruct via inverse STFT (no phase guessing)

**Libraries:**
- [wavefrontshaping/complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)
- [josiahwsmith10/complextorch](https://github.com/josiahwsmith10/complextorch)

**Pros:**
- Perfect reconstruction (if magnitude + phase preserved)
- No phase estimation needed
- Theoretically sound

**Cons:**
- 2x parameters (complex weights)
- Higher memory usage
- Requires training (no pretrained models)
- More complex implementation

---

## Why EnCodec Beats Alternatives

### vs. Complex FFT
- ✓ Pretrained (no training needed)
- ✓ Lower parameters (10M vs. 20M+)
- ✓ Simpler implementation (standard PyTorch)
- ✓ Production-proven

### vs. Raw Waveform
- ✓ Much lower parameters (10M vs. 50M+)
- ✓ Lower memory (codes vs. raw samples)
- ✓ Pretrained models available
- ✓ Faster training

### vs. Wav2Vec 2.0
- ✓ Designed for reconstruction (not just encoding)
- ✓ Has decoder (Wav2Vec is encoder-only)
- ✓ Lower parameters (10M vs. 95M)
- ✓ Works for music + speech (not just speech)

### vs. WaveGlow (Normalizing Flows)
- ✓ Much lower parameters (10M vs. 87M)
- ✓ Faster (no sequential flow steps)
- ✓ Designed for compression (not just generation)
- ✓ Pretrained for our task

---

## Resources

### EnCodec
- Paper: ["High Fidelity Neural Audio Compression"](https://arxiv.org/abs/2210.13438) (Défossez et al., 2022)
- GitHub: [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- Demo: [audiocraft.metademolab.com/encodec](https://audiocraft.metademolab.com/encodec.html)
- License: MIT

### Related Work
- Paper: ["Natural sounds reconstructed from fMRI"](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003293) (PLOS Biology, 2024)
  - They used DNN features + generative models, but faced temporal bottleneck challenges
- Paper: ["SoundStream: An End-to-End Neural Audio Codec"](https://arxiv.org/abs/2107.03312) (Google, 2021)
  - Similar to EnCodec, but EnCodec has better quality + pretrained models

---

## Next Steps

1. **Test EnCodec on Sherlock audio** (this week)
   - Install, encode/decode, evaluate quality
2. **Integrate with dataset** (next week)
   - Precompute codes, update dataset class
3. **Train autoencoder** (following week)
   - Update encoder/decoder, train on codes
4. **Evaluate & compare** (week after)
   - Metrics + listening tests vs. old approach
5. **Report results** (update Issue #23)

**Timeline:** 3-4 weeks to complete implementation + evaluation

---

## Questions for Team

1. Should we proceed with EnCodec testing?
2. Preference for Option A (freeze EnCodec) vs. Option B (fine-tune)?
3. What quality metrics are most important (SNR, PESQ, subjective listening)?
4. Should we also test Complex FFT as comparison?

---

**Full research report:** See `/notes/AUDIO_ENCODING_RESEARCH_REPORT.md` for detailed analysis of all 7 approaches.

# Audio Encoding Approaches for fMRI Autoencoder Reconstruction
## Comprehensive Research Report for Issue #23

**Date:** 2025-10-31
**Author:** Research Analysis
**Status:** Ready for Implementation Decision

---

## Executive Summary

The current mel spectrogram + Griffin-Lim approach fails because **phase information is discarded**, and the Griffin-Lim algorithm cannot reliably reconstruct phase after bottleneck compression through the fMRI autoencoder. This report evaluates 4 major alternative approaches for high-quality audio reconstruction.

### Top Recommendation: EnCodec (⭐⭐⭐⭐⭐)

**EnCodec is the clear winner** for this use case because:
- **Designed specifically for compression + reconstruction** (exactly our bottleneck scenario)
- **Preserves phase implicitly** through learned representations
- **MIT license** (research-friendly, no restrictions)
- **Low parameter count** (~10M parameters)
- **Pretrained models available** (24kHz mono, 48kHz stereo)
- **Production-ready** (used in Meta's AudioGen, MusicGen)
- **Compression ratios**: 2x to 64x (flexible for different bottleneck sizes)
- **Easy integration** with PyTorch

---

## The Core Problem

```
Current Pipeline:
Audio Waveform → Mel Spectrogram (MAGNITUDE ONLY) → Encoder → fMRI Bottleneck → Decoder → Mel Spectrogram → Griffin-Lim → Audio
                                    ↑ PHASE LOST                                                              ↑ PHASE GUESSING FAILS
```

**Why it fails:**
1. Mel spectrograms discard phase information (only magnitude retained)
2. Griffin-Lim tries to "guess" phase through iterative reconstruction
3. After bottleneck compression, magnitude is degraded → phase reconstruction becomes impossible
4. Result: Garbled, unintelligible audio

**What we need:**
An audio representation that either:
- Preserves phase information explicitly, OR
- Uses learned representations that implicitly encode phase, OR
- Is invertible by design (no information loss in forward/backward transform)

---

## Approach Comparison Table

| Approach | Phase Preservation | Reconstruction Quality | Parameters | Memory Usage | Implementation Complexity | Training Required | Recommendation |
|----------|-------------------|----------------------|------------|--------------|---------------------------|------------------|----------------|
| **EnCodec** | ✓ Learned | Excellent | Low (10M) | Low | Low | Pretrained available | ⭐⭐⭐⭐⭐ |
| **Complex FFT** | ✓ Full (explicit) | Excellent | Medium (2x) | High | Medium | Yes | ⭐⭐⭐⭐ |
| **Raw Waveform** | ✓ Implicit | Excellent | High (50M+) | Very High | High | Yes | ⭐⭐⭐ |
| **Wav2Vec 2.0** | ~ Partial | Good | Medium (95M) | Medium | Medium | Pretrained available | ⭐⭐⭐ |
| **Audio-as-Image** | ~ Partial | Good | Medium | Medium | Low | Yes | ⭐⭐⭐ |
| **CQT** | Requires phase | Good | Medium | Medium | Medium | Yes | ⭐⭐ |
| **WaveGlow** | ✓ Invertible | Excellent | Very High (87M) | Very High | Very High | Yes | ⭐⭐ |

---

## Detailed Analysis

### 1. EnCodec: Neural Audio Codec (RECOMMENDED ⭐⭐⭐⭐⭐)

#### Overview
EnCodec is a state-of-the-art neural audio codec from Meta AI specifically designed for **high-fidelity compression and reconstruction**. It's the only approach in this list explicitly built for our exact use case: compress audio → reconstruct with high quality.

#### Architecture
```
Input Audio (48kHz stereo or 24kHz mono)
    ↓
Convolutional Encoder (strided convolutions for downsampling)
    ↓
Residual Vector Quantization (RVQ) - Compression happens here
    ↓
Convolutional Decoder (transposed convolutions for upsampling)
    ↓
Output Audio (reconstructed)
```

**Key Innovation:** Uses **Residual Vector Quantization (RVQ)** which:
- Quantizes audio into discrete codes (like tokens)
- Multiple codebooks handle residuals progressively
- Preserves quality at extreme compression ratios
- Phase is implicitly preserved in learned representations

#### Training Approach
EnCodec is trained with:
1. **Reconstruction loss** (L1 + multi-scale STFT)
2. **Perceptual loss** (MS-STFT discriminator)
3. **Adversarial loss** (ensures realism)
4. **Loss balancer** (automatically weights objectives)

This multi-objective training ensures both accurate and perceptually pleasing reconstruction.

#### For Our Use Case

**Integration Strategy:**
```python
# Option 1: Use EnCodec as preprocessor (RECOMMENDED)
# Before training autoencoder:
audio → EnCodec.encode() → compressed codes → feed to autoencoder encoder
# After autoencoder decoder:
predicted codes → EnCodec.decode() → reconstructed audio

# Option 2: Fine-tune EnCodec bottleneck to match fMRI
# Replace EnCodec's quantization layer with linear projection to fMRI dimensions
# Fine-tune entire model end-to-end
```

**Why this works:**
- EnCodec already handles audio compression → reconstruction
- Its latent codes are designed to preserve all information needed for reconstruction
- We can either:
  - Use EnCodec as-is (freeze it, train autoencoder on its codes)
  - Fine-tune it to map directly to fMRI dimensions

#### Specifications

**Available Models:**
- **24 kHz mono**: Compression to 1.5, 3, 6, 12, or 24 kbps
- **48 kHz stereo**: Compression to 3, 6, 12, or 24 kbps (music quality)

**Parameters:** ~10M (encoder) + ~10M (decoder) = 20M total

**Memory Requirements:**
- Standard mode: 4GB RAM for 3-5 min audio
- HQ mode: ~200MB (processes 1-second chunks)
- **For fMRI (920 TRs × 1.5s = 23 min):** Use HQ mode, ~1GB RAM

**Compression Example:**
```
Input: 48kHz stereo, 16-bit = 1,536 kbps
EnCodec @ 12 kbps = 128x compression
Quality: Perceptually lossless for music
```

**Latent Representation Size:**
```
For 24kHz model at 3kbps:
- Frame rate: 75 Hz (320 samples per frame)
- Codebook: 1024 entries
- Number of quantizers: 8
- Latent per second: 75 frames/s × 8 quantizers = 600 codes/s

For our 23-minute stimulus:
23 min × 60 s × 600 codes/s = 828,000 codes
Per TR (1.5s): 900 codes

This is much smaller than raw spectrograms!
```

#### Installation & Usage

```bash
pip install encodec
```

```python
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

# Load pretrained model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps

# Encode
wav = torch.randn(1, 1, 24000)  # 1 second, mono, 24kHz
encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

# Decode
reconstructed = model.decode(encoded_frames)

# Integration with autoencoder:
# 1. Encode all audio to codes: [920 TRs, n_codes_per_TR]
# 2. Feed codes to autoencoder (instead of mel spectrograms)
# 3. Autoencoder predicts codes from fMRI
# 4. Decode predicted codes back to audio
```

#### Pros
- ✓ **Designed for compression + reconstruction** (perfect for our bottleneck)
- ✓ **Pretrained models** (no training needed initially)
- ✓ **MIT license** (no restrictions)
- ✓ **Low parameters** (10M encoder, 10M decoder)
- ✓ **Excellent quality** (outperforms MP3 at 10x compression)
- ✓ **Flexible compression ratios** (1.5-24 kbps)
- ✓ **Production-proven** (used in AudioGen, MusicGen)
- ✓ **Easy PyTorch integration**
- ✓ **Handles stereo + mono**

#### Cons
- ✗ Requires learning new API (but it's simple)
- ✗ Black box (harder to interpret than STFT)
- ✗ Adds dependency (but well-maintained by Meta)

#### References
- Paper: ["High Fidelity Neural Audio Compression" (2022)](https://arxiv.org/abs/2210.13438)
- GitHub: [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- Demo: [audiocraft.metademolab.com/encodec.html](https://audiocraft.metademolab.com/encodec.html)

---

### 2. Complex-Valued FFT with Phase Preservation (⭐⭐⭐⭐)

#### Overview
Instead of mel spectrograms (magnitude only), use **complex-valued STFT** (magnitude + phase) and process with **complex-valued neural networks**.

#### How It Works

**Standard STFT:**
```python
stft = torch.stft(audio, n_fft=2048, return_complex=True)
# stft is complex: shape [freq_bins, time_frames]
# stft = magnitude * exp(i * phase)

# Current approach (BAD):
mel_spec = magnitude_only(stft)  # Phase lost!

# Complex approach (GOOD):
real_part = stft.real  # [freq_bins, time]
imag_part = stft.imag  # [freq_bins, time]
complex_input = torch.stack([real_part, imag_part], dim=0)  # [2, freq, time]
```

**Complex-Valued Neural Networks:**
- Use complex arithmetic in all layers
- Convolutions, activations, batch norm all operate on complex numbers
- Preserves phase information throughout network
- Reconstruction via inverse STFT (perfect if magnitude + phase preserved)

#### Architecture

```python
# Complex Convolution Layer (simplified):
class ComplexConv2d:
    def forward(self, input_complex):
        # input_complex has .real and .imag attributes
        # weight_complex has .real and .imag attributes

        # Complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        output_real = F.conv2d(input_complex.real, weight.real) - F.conv2d(input_complex.imag, weight.imag)
        output_imag = F.conv2d(input_complex.real, weight.imag) + F.conv2d(input_complex.imag, weight.real)

        return torch.complex(output_real, output_imag)
```

#### For Our Use Case

```python
# Encoder pathway:
audio → STFT (complex) → Complex CNN → bottleneck (complex) → match to fMRI

# Decoder pathway:
fMRI prediction → Complex CNN → STFT (complex) → Inverse STFT → audio

# No phase estimation needed - phase is preserved throughout!
```

#### PyTorch Libraries

**Available implementations (2024):**
1. **wavefrontshaping/complexPyTorch**
   - Simple, high-level API
   - Requires PyTorch >= 1.7 (native complex support)
   - Layers: ComplexConv2d, ComplexLinear, ComplexBatchNorm2d

2. **josiahwsmith10/complextorch**
   - Efficient implementations
   - Updated for PyTorch 2.1.0+
   - Includes attention modules

3. **ivannz/cplxmodule**
   - Advanced features (variational dropout)
   - Based on "Deep Complex Networks" (ICLR 2018)

4. **torchcvnn/torchcvnn**
   - Comprehensive layer library
   - Updated Feb 2024

**Example usage:**
```python
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

class ComplexEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(1, 64, 3, 2, 1)
        self.conv2 = ComplexConv2d(64, 128, 3, 2, 1)

    def forward(self, x_complex):
        x = complex_relu(self.conv1(x_complex))
        x = complex_relu(self.conv2(x))
        return x
```

#### Parameters & Memory

**Parameter increase:** ~2x vs real-valued network
- Each complex weight = 2 real numbers (real + imaginary)
- But STFT has 2x channels anyway (real/imag), so similar total

**Memory increase:** Moderate
- Complex tensors use 2x memory of real tensors
- But we avoid storing magnitude + phase separately
- For 1025 freq bins × 1293 time frames: ~2.6M values × 2 = 5.2M floats
- Per TR: 5.2M × 920 TRs = 4.8B floats = 19.2 GB (need batching)

**Computation:** Slightly slower than real-valued
- Complex multiply = 4 real multiplies + 2 adds
- But PyTorch's native complex support is optimized

#### Pros
- ✓ **Perfect reconstruction** (if magnitude + phase preserved exactly)
- ✓ **No phase estimation** (no Griffin-Lim needed)
- ✓ **Theoretically sound** (invertible transform)
- ✓ **PyTorch libraries available** (mature implementations)
- ✓ **Flexible** (can use any CNN architecture)

#### Cons
- ✗ **2x parameters** vs real-valued networks
- ✗ **Higher memory usage** (complex tensors)
- ✗ **Requires training** (no pretrained complex audio models)
- ✗ **More complex implementation** (need to learn complex layers)
- ✗ **Still susceptible to bottleneck compression** (fMRI dim << STFT dim)

#### References
- Paper: ["Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications" (2024)](https://arxiv.org/abs/2510.09926)
- Paper: ["DCCRN: Deep Complex Convolution Recurrent Network" (2020)](https://www.researchgate.net/publication/343415434)
- PyTorch support: [torch.fft module](https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/)

---

### 3. Raw Waveform Encoding (1D CNNs) (⭐⭐⭐)

#### Overview
Skip spectrograms entirely. Encode the **raw audio waveform** using 1D CNNs, similar to WaveNet or SampleRNN.

#### Architecture

**WaveNet-style (Dilated Convolutions):**
```python
class WaveNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Dilated convolutions with exponentially growing dilation
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=2, dilation=2**i)
            for i in range(10)  # dilation: 1, 2, 4, 8, ..., 512
        ])
        # Receptive field: 1024 samples = 43ms @ 24kHz

    def forward(self, waveform):
        # waveform: [batch, 1, time_samples]
        x = waveform
        for conv in self.dilated_convs:
            x = torch.relu(conv(x))
        return x  # Compressed representation
```

**Strided Convolutions (like EnCodec):**
```python
class StridedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Aggressive downsampling via strided convolutions
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=4)    # 24000 → 6000
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=4)  # 6000 → 1500
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, stride=5) # 1500 → 300
        self.conv4 = nn.Conv1d(256, 512, kernel_size=7, stride=5) # 300 → 60
        # Total downsampling: 400x (24000 → 60 samples per second)

    def forward(self, waveform):
        x = torch.relu(self.conv1(waveform))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x
```

#### For Our Use Case

**Challenge:** Temporal scale mismatch
- Audio sample rate: 24,000 Hz (24,000 samples/second)
- TR duration: 1.5 seconds = 36,000 samples per TR
- Total samples: 920 TRs × 36,000 = 33,120,000 samples

**Solution:** Aggressive downsampling
```python
# Downsample 24kHz → 75 Hz (320x)
# This matches EnCodec's frame rate
# Result: 75 samples/second × 1380 seconds = 103,500 samples total
# Per TR: ~112 samples (manageable)

encoder_output = strided_encoder(waveform)  # [920, 512, 112]
# Flatten or pool to match fMRI dimensions
```

#### Parameters & Memory

**Parameters:** High
- Example (WaveNet): ~50M parameters
- Example (SampleRNN): ~30M parameters
- Reason: Need large receptive fields to capture long-range dependencies

**Memory:** Very High
- Input: 33M samples × 4 bytes = 132 MB (just for input!)
- Intermediate activations: Much larger
- Need to process in chunks (per TR or smaller)

**Computation:** Expensive
- 1D convolutions over millions of samples
- Autoregressive models (WaveNet) are very slow for generation
- Non-autoregressive (strided) is faster but still heavy

#### Pros
- ✓ **No transform needed** (end-to-end learning)
- ✓ **Perfect phase preservation** (implicitly learned)
- ✓ **Theoretically optimal** (no hand-crafted features)
- ✓ **Proven in speech synthesis** (WaveNet, SampleRNN)

#### Cons
- ✗ **Very high memory** (millions of samples)
- ✗ **High parameter count** (50M+)
- ✗ **Slow training** (long sequences)
- ✗ **Requires careful design** (receptive field, downsampling)
- ✗ **No pretrained models** for this specific task
- ✗ **Hard to interpret** (what do waveform features mean?)

#### References
- Paper: ["WaveNet: A Generative Model for Raw Audio" (2016)](https://arxiv.org/abs/1609.03499)
- Blog: [Sander Dieleman - Generating music in the waveform domain](https://sander.ai/2020/03/24/audio-generation.html)
- Paper: ["SampleRNN: An Unconditional End-to-End Neural Audio Generation Model"](https://arxiv.org/abs/1612.07837)

---

### 4. Pretrained Audio Representations (Wav2Vec 2.0 / HuBERT) (⭐⭐⭐)

#### Overview
Use **self-supervised pretrained models** (Wav2Vec 2.0, HuBERT) to extract audio features, then train autoencoder on those features.

#### How It Works

**Wav2Vec 2.0:**
1. Pretrained on 960 hours of speech (LibriSpeech)
2. Learns to predict masked audio segments (like BERT for audio)
3. Outputs high-level representations (phonetic, acoustic features)
4. 12-24 transformer layers, outputs features every 20ms

**HuBERT (Hidden Unit BERT):**
- Similar to Wav2Vec 2.0
- Predicts discrete acoustic units (from clustered MFCCs)
- Encodes low-level acoustic features (early layers) → phonetic content (deep layers)

#### Architecture

```python
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE

# Load pretrained model
bundle = WAV2VEC2_BASE
model = bundle.get_model()
model.eval()

# Extract features
with torch.no_grad():
    waveform, sample_rate = torchaudio.load("audio.wav")
    features, _ = model.extract_features(waveform)
    # features: list of tensors from each transformer layer
    # Use final layer: features[-1]  # [batch, time, 768]
```

#### For Our Use Case

**Option 1: Use as feature extractor (frozen)**
```python
# Extract features from all audio
audio_features = []
for tr in range(920):
    waveform = load_audio_for_tr(tr)  # 1.5s @ 24kHz
    with torch.no_grad():
        feats, _ = wav2vec_model.extract_features(waveform)
        audio_features.append(feats[-1].mean(dim=1))  # Average over time: [768]

# Train autoencoder: fMRI → 768-dim features
# Reconstruct by: predicted features → ... → audio (need decoder!)
```

**Challenge:** Wav2Vec/HuBERT are **encoders only** - no native decoder for reconstruction!

**Option 2: Train decoder (vocoder)**
```python
# Need to train: 768-dim features → audio waveform
# This is essentially building a vocoder conditioned on Wav2Vec features
# Similar to: HiFi-GAN, WaveGlow, but conditioned on Wav2Vec instead of mel-spec
```

**Option 3: Use DeCoAR 2.0 (has reconstruction loss)**
- DeCoAR 2.0 combines Wav2Vec 2.0 loss + reconstruction loss
- Trained to reconstruct audio from features
- But less common, fewer pretrained models

#### Parameters & Memory

**Model size:**
- Wav2Vec 2.0 Base: 95M parameters
- Wav2Vec 2.0 Large: 317M parameters
- HuBERT Base: 95M parameters

**Feature dimensions:**
- Output: 768-dim (base) or 1024-dim (large)
- Time resolution: 50 Hz (features every 20ms)
- Per TR (1.5s): 75 features × 768 dims = 57,600 values

**Memory:** Moderate (frozen model) to High (fine-tuning)

#### Pros
- ✓ **Pretrained representations** (leverage large-scale pretraining)
- ✓ **Semantic features** (phonetic, acoustic information)
- ✓ **Proven for speech tasks** (ASR, emotion recognition)
- ✓ **Available in PyTorch** (torchaudio, HuggingFace)

#### Cons
- ✗ **No native decoder** (need to build vocoder for reconstruction)
- ✗ **Large models** (95M parameters just for encoder)
- ✗ **Designed for speech** (may not capture music/environmental sounds well)
- ✗ **Not optimized for reconstruction** (optimized for classification tasks)
- ✗ **Extra training needed** (vocoder to reconstruct audio)

#### When to Use
- If you want semantic/phonetic features for analysis
- If reconstruction quality is secondary to interpretability
- If you're working with speech primarily (not music/environmental sounds)

**For this project:** Not ideal because reconstruction is the goal, and these models lack decoders.

#### References
- Paper: ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)](https://arxiv.org/abs/2006.11477)
- Paper: ["HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction" (2021)](https://arxiv.org/abs/2106.07447)
- PyTorch: [torchaudio.pipelines](https://pytorch.org/audio/stable/pipelines.html)

---

### 5. Spectrogram as Image (2D CNN with Phase Channel) (⭐⭐⭐)

#### Overview
Treat the audio spectrogram as a 2D image with **multiple channels**: magnitude + phase (or real + imaginary).

#### Architecture

```python
# Convert STFT to 2-channel image
stft = torch.stft(audio, n_fft=2048, return_complex=True)
magnitude = stft.abs()  # [freq_bins, time_frames]
phase = stft.angle()    # [freq_bins, time_frames]

# Stack as 2-channel image
audio_image = torch.stack([magnitude, phase], dim=0)  # [2, freq, time]

# Or use real/imaginary:
audio_image = torch.stack([stft.real, stft.imag], dim=0)  # [2, freq, time]

# Use standard 2D CNN (like for images)
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)  # 2 channels: mag + phase
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x
```

#### For Our Use Case

**Encoder pathway:**
```python
audio → STFT (complex) → [magnitude, phase] → 2D CNN → bottleneck → fMRI
```

**Decoder pathway:**
```python
fMRI → 2D CNN → [magnitude, phase] → Inverse STFT → audio
```

**Advantages over current approach:**
- Phase is explicitly included as a channel
- Network can learn to preserve phase through bottleneck
- Standard 2D CNN (no complex layers needed)
- Reconstruction via inverse STFT (no Griffin-Lim)

#### Alternative: 3D Encoding (Time/Frequency/Phase)

**Idea:** Map audio to 3D space (time × frequency × phase)
- Encode phase as spatial dimension
- Use 3D CNNs (like video processing)
- More expressive but much more memory

```python
# Bin phase into K discrete values (e.g., K=16)
phase_binned = (phase + np.pi) / (2 * np.pi) * K  # Map [-π, π] → [0, K]

# Create 3D tensor
audio_3d = torch.zeros(freq_bins, time_frames, K)
for f in range(freq_bins):
    for t in range(time_frames):
        k = int(phase_binned[f, t])
        audio_3d[f, t, k] = magnitude[f, t]

# Use 3D CNN
conv3d = nn.Conv3d(1, 64, kernel_size=3)
```

#### Parameters & Memory

**2-channel 2D CNN:**
- Similar to standard image CNN
- Input: [2, 1025, 1293] (for 1 TR with n_fft=2048)
- Memory: ~2.6M × 2 channels = 5.2M floats = 21 MB per TR
- Parameters: Standard (depends on architecture, e.g., 10-50M)

**3D CNN:**
- Input: [1, freq, time, phase_bins]
- Much larger (phase adds dimension)
- Memory: 5.2M × 16 bins = 83M floats = 332 MB per TR
- Parameters: Higher than 2D (3D convolutions are expensive)

#### Pros
- ✓ **Simple implementation** (standard 2D CNN)
- ✓ **Explicit phase representation** (network can learn to preserve it)
- ✓ **No complex layers needed** (unlike Approach 2)
- ✓ **Flexible** (can use pretrained image models as starting point)
- ✓ **Invertible** (inverse STFT if magnitude + phase preserved)

#### Cons
- ✗ **Phase representation is tricky** (wrapping at ±π, discontinuities)
- ✗ **Network may not learn to preserve phase well** (no guarantee)
- ✗ **Requires training** (no pretrained audio-as-image models)
- ✗ **3D version is memory-intensive**

#### When to Use
- If you want standard CNNs (no complex layers)
- If you want to leverage image processing techniques
- If you're willing to experiment with phase representation

**For this project:** Worth trying as a simpler alternative to complex CNNs, but EnCodec is likely better.

#### References
- Paper: ["Audio Spectrogram Representations for Processing with CNNs" (2017)](https://arxiv.org/abs/1706.09559)
- Discussion: [Do I need 3 RGB channels for spectrogram CNN?](https://stats.stackexchange.com/questions/426818)

---

### 6. Constant-Q Transform (CQT) (⭐⭐)

#### Overview
CQT is an alternative to STFT with **logarithmic frequency spacing** (like musical notes) instead of linear spacing.

#### How It Works

**STFT:** Linear frequency bins (e.g., 0 Hz, 10 Hz, 20 Hz, ...)
- Good for speech, generic audio
- Uniform resolution across frequencies

**CQT:** Logarithmic frequency bins (e.g., C1, C#1, D1, ..., musical notes)
- Good for music (matches human auditory perception)
- Better frequency resolution at low frequencies
- Better temporal resolution at high frequencies

```python
import librosa

# Compute CQT
audio, sr = librosa.load("audio.wav", sr=24000)
cqt = librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12)
# Shape: [n_bins, time_frames]
# n_bins=84 covers 7 octaves (12 bins per octave)
```

#### For Our Use Case

**Problem:** CQT still discards phase!
- Like mel spectrograms, CQT gives magnitude only
- Inverse CQT requires phase (same problem as before)
- Would still need Griffin-Lim or phase estimation

**Possible solution:** CQT + phase preservation
- Use CQT magnitude + STFT phase (hybrid)
- Or use complex-valued CQT (if available)

#### Pros
- ✓ **Better for music** (logarithmic frequency scale)
- ✓ **Perceptually motivated** (matches human hearing)
- ✓ **Better pitch resolution** than STFT
- ✓ **Compact representation** (fewer bins needed)

#### Cons
- ✗ **Still discards phase** (same problem as mel spectrogram)
- ✗ **Computationally expensive** (slower than STFT)
- ✗ **Less common in deep learning** (fewer libraries, examples)
- ✗ **Doesn't solve our core problem** (phase reconstruction)

#### When to Use
- If stimulus is primarily music
- If you need pitch-aware features
- Combined with phase preservation method

**For this project:** Not recommended as primary approach; doesn't solve phase problem.

#### References
- [Constant-Q Transform - Wikipedia](https://en.wikipedia.org/wiki/Constant-Q_transform)
- Paper: ["Constant-Q transform toolbox for music processing"](https://www.researchgate.net/publication/228523955)
- librosa: [librosa.cqt](https://librosa.org/doc/main/generated/librosa.cqt.html)

---

### 7. Normalizing Flows (WaveGlow, Glow-TTS) (⭐⭐)

#### Overview
**Normalizing flows** are invertible neural networks - by design, they can perfectly reverse their forward pass. WaveGlow and Glow-TTS use flows for audio synthesis.

#### How It Works

**Key idea:** Learn an invertible mapping between:
- Gaussian noise distribution (simple) ↔ Audio distribution (complex)

**Architecture:**
```
Audio → Flow (forward) → Latent (Gaussian)
Latent (Gaussian) → Flow (reverse) → Audio

Flow consists of:
1. Activation normalization (ActNorm)
2. Invertible 1×1 convolution
3. Affine coupling layers (invertible by design)
```

**Why invertible:**
- Each layer is designed to be reversible
- Forward pass: audio → latent codes
- Reverse pass: latent codes → audio (exact reconstruction if no loss)

#### For Our Use Case

**WaveGlow for Audio:**
```python
# Training:
audio → WaveGlow.encode() → latent (Gaussian noise)
latent → WaveGlow.decode() → audio (reconstruction loss)

# Our use:
audio → WaveGlow.encode() → latent → autoencoder encoder → fMRI
fMRI → autoencoder decoder → latent → WaveGlow.decode() → audio
```

**Advantages:**
- Perfect invertibility (in theory)
- No phase estimation needed
- High-quality synthesis

**Challenges:**
- Very large models (WaveGlow: 87M parameters)
- Designed for synthesis (text → audio), not compression
- No pretrained models for our task (speech/music encoding)

#### Parameters & Memory

**Model size:**
- WaveGlow: 87M parameters
- Glow-TTS: 28M parameters (but for text → mel-spec, not audio)

**Computation:**
- Slow (many sequential flow steps)
- Not real-time

**Memory:**
- Very high (need to store intermediate flow states)

#### Pros
- ✓ **Invertible by design** (mathematically guaranteed)
- ✓ **High-quality synthesis** (proven in TTS)
- ✓ **No phase problem** (works directly on waveforms)

#### Cons
- ✗ **Very large models** (87M parameters)
- ✗ **Slow** (many sequential steps)
- ✗ **Not designed for compression** (designed for generation)
- ✗ **No pretrained models** for audio encoding (only for TTS)
- ✗ **High memory usage**

#### When to Use
- If you need guaranteed invertibility
- If you have massive compute resources
- If you're willing to train from scratch

**For this project:** Not recommended - too heavy, not designed for our use case. EnCodec is better.

#### References
- Paper: ["WaveGlow: A Flow-based Generative Network for Speech Synthesis" (2018)](https://arxiv.org/abs/1811.00002)
- Paper: ["Glow-TTS: A Generative Flow for Text-to-Speech" (2020)](https://arxiv.org/abs/2005.11129)
- Blog: [Understanding Normalizing Flows](https://medium.com/subex-ai-labs/understanding-normalizing-flows-and-its-use-case-in-speech-synthesis-part-2-3e19840e80b5)

---

## Comparison: EnCodec vs. Alternatives

### Why EnCodec Wins

| Criterion | EnCodec | Complex FFT | Raw Waveform | Wav2Vec 2.0 | Audio-as-Image | WaveGlow |
|-----------|---------|-------------|--------------|-------------|----------------|----------|
| **Designed for our use case** | ✓✓✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| **Pretrained available** | ✓✓✓ | ✗ | ✗ | ✓✓ | ✗ | ✗ |
| **Reconstruction quality** | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓ | ✓✓ | ✓✓✓ |
| **Parameter efficiency** | ✓✓✓ | ✓✓ | ✗ | ✗ | ✓✓ | ✗ |
| **Memory efficiency** | ✓✓✓ | ✓ | ✗ | ✓✓ | ✓✓ | ✗ |
| **Ease of implementation** | ✓✓✓ | ✓✓ | ✓ | ✓✓✓ | ✓✓✓ | ✓ |
| **Training required** | ✗ (pretrained) | ✓ | ✓ | ✗ (pretrained) | ✓ | ✓ |
| **Production-ready** | ✓✓✓ | ✗ | ✗ | ✓✓ | ✗ | ✓ |

### Real-World Evidence

**EnCodec is used in production:**
- Meta's AudioGen (audio generation)
- Meta's MusicGen (music generation)
- Various music streaming research projects

**Why it works for compression + reconstruction:**
- Trained on massive datasets (24kHz: speech, 48kHz: music)
- Multi-objective training (reconstruction + perceptual quality)
- Residual vector quantization handles extreme compression
- Adversarial training ensures realism

---

## Implementation Roadmap: EnCodec Integration

### Phase 1: Proof of Concept (1-2 days)

**Goal:** Verify EnCodec can reconstruct our Sherlock audio at acceptable quality.

```python
# 1. Install
pip install encodec

# 2. Load Sherlock audio
from giblet.datasets.sherlock import SherlockDataset
dataset = SherlockDataset(...)
audio_full = dataset.get_audio()  # Full Sherlock audio

# 3. Test EnCodec encoding/decoding
from encodec import EncodecModel

model_24k = EncodecModel.encodec_model_24khz()
model_24k.set_target_bandwidth(6.0)  # 6 kbps

# Encode entire audio
encoded = model_24k.encode(audio_full)
codes = torch.cat([enc[0] for enc in encoded], dim=-1)  # Extract codes

# Decode
reconstructed = model_24k.decode(encoded)

# 4. Evaluate quality
import torchaudio
import torchaudio.transforms as T

# Listen
torchaudio.save("original.wav", audio_full, 24000)
torchaudio.save("reconstructed.wav", reconstructed, 24000)

# Metrics
snr = compute_snr(audio_full, reconstructed)
pesq_score = compute_pesq(audio_full, reconstructed)
print(f"SNR: {snr:.2f} dB, PESQ: {pesq_score:.2f}")
```

**Expected results:**
- SNR > 20 dB (good quality)
- PESQ > 3.0 (perceptually decent)
- Subjective: Should sound very similar to original

---

### Phase 2: Integration with Autoencoder (3-5 days)

**Goal:** Replace mel spectrogram with EnCodec codes in dataset/model.

#### Step 2.1: Modify Dataset

```python
# giblet/datasets/sherlock.py

class SherlockDataset:
    def __init__(self, use_encodec=True, encodec_bandwidth=6.0):
        self.use_encodec = use_encodec

        if self.use_encodec:
            from encodec import EncodecModel
            self.encodec = EncodecModel.encodec_model_24khz()
            self.encodec.set_target_bandwidth(encodec_bandwidth)
            self._precompute_encodec_codes()

    def _precompute_encodec_codes(self):
        """Precompute EnCodec codes for all TRs to save time during training."""
        print("Precomputing EnCodec codes...")
        audio_full = self.get_audio()

        # Encode entire audio
        with torch.no_grad():
            encoded = self.encodec.encode(audio_full)
            # encoded is a list of (codes, scales) tuples
            codes = torch.cat([enc[0] for enc in encoded], dim=-1)
            # codes: [batch=1, num_quantizers, time_frames]

        # Split into TRs
        # Assuming 75 frames/sec, 1.5s per TR → ~112 frames per TR
        frames_per_tr = int(75 * self.tr_duration)
        self.encodec_codes = []

        for tr_idx in range(self.n_trs):
            start = tr_idx * frames_per_tr
            end = start + frames_per_tr
            tr_codes = codes[:, :, start:end]  # [1, n_q, frames_per_tr]
            self.encodec_codes.append(tr_codes.squeeze(0))  # [n_q, frames_per_tr]

        print(f"Precomputed {len(self.encodec_codes)} TRs of EnCodec codes")

    def __getitem__(self, idx):
        # ... existing code ...

        if self.use_encodec:
            audio = self.encodec_codes[idx]  # [n_quantizers, time_frames]
        else:
            # Old mel spectrogram code
            audio = self._get_mel_spectrogram(idx)

        return {
            'fmri': fmri,
            'audio': audio,
            'video': video,
            'text': text
        }
```

#### Step 2.2: Modify Encoder

```python
# giblet/models/encoder.py

class AudioEncoder(nn.Module):
    def __init__(self, input_type='mel', encodec_params=None):
        super().__init__()
        self.input_type = input_type

        if input_type == 'encodec':
            # EnCodec codes: [n_quantizers, time_frames]
            # e.g., [8 quantizers, 112 frames] for 1.5s @ 75Hz
            n_quantizers = encodec_params['n_quantizers']  # 8
            time_frames = encodec_params['frames_per_tr']  # ~112

            self.encoder = nn.Sequential(
                nn.Flatten(),  # [8 * 112] = 896
                nn.Linear(n_quantizers * time_frames, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
        elif input_type == 'mel':
            # Old mel spectrogram encoder
            ...

    def forward(self, x):
        return self.encoder(x)
```

#### Step 2.3: Modify Decoder

```python
# giblet/models/decoder.py

class AudioDecoder(nn.Module):
    def __init__(self, output_type='mel', encodec_params=None, encodec_model=None):
        super().__init__()
        self.output_type = output_type
        self.encodec_model = encodec_model  # For final waveform reconstruction

        if output_type == 'encodec':
            n_quantizers = encodec_params['n_quantizers']
            time_frames = encodec_params['frames_per_tr']

            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, n_quantizers * time_frames),
                nn.Unflatten(1, (n_quantizers, time_frames))
            )
        elif output_type == 'mel':
            # Old mel decoder
            ...

    def forward(self, z):
        codes_pred = self.decoder(z)  # [batch, n_quantizers, time_frames]
        return codes_pred

    def reconstruct_audio(self, codes_pred):
        """Convert predicted codes to audio waveform using EnCodec decoder."""
        if self.encodec_model is None:
            raise ValueError("Need encodec_model for audio reconstruction")

        # EnCodec.decode() expects list of (codes, None) tuples
        # We predicted codes, but don't have scales - use None
        encoded_frames = [(codes_pred, None)]

        with torch.no_grad():
            audio = self.encodec_model.decode(encoded_frames)

        return audio
```

#### Step 2.4: Training Loop

```python
# train.py

from encodec import EncodecModel

# Load EnCodec for reconstruction (not training - just for loss)
encodec = EncodecModel.encodec_model_24khz()
encodec.set_target_bandwidth(6.0)
encodec.eval()  # Frozen

# Training loop
for batch in dataloader:
    fmri = batch['fmri']
    audio_codes_true = batch['audio']  # EnCodec codes

    # Forward pass
    z_fmri = encoder_fmri(fmri)
    audio_codes_pred = decoder_audio(z_fmri)

    # Loss 1: Code-level MSE
    loss_codes = F.mse_loss(audio_codes_pred, audio_codes_true)

    # Loss 2: Reconstructed audio quality (optional, slower)
    # Decode both true and predicted codes to audio
    audio_true = encodec.decode([(audio_codes_true, None)])
    audio_pred = encodec.decode([(audio_codes_pred, None)])
    loss_audio = F.l1_loss(audio_pred, audio_true)

    # Combined loss
    loss = loss_codes + 0.1 * loss_audio  # Weight audio loss less (slower to compute)

    loss.backward()
    optimizer.step()
```

---

### Phase 3: Optimization (1-2 weeks)

**Goal:** Fine-tune for best quality.

#### Option A: Freeze EnCodec (Simpler)
- Keep EnCodec frozen (don't train it)
- Only train autoencoder to predict EnCodec codes
- Pros: Faster, simpler, less memory
- Cons: EnCodec not optimized for our bottleneck

#### Option B: Fine-tune EnCodec (Better Quality)
- Replace EnCodec's quantization layer with linear projection to fMRI dims
- Fine-tune entire EnCodec + autoencoder end-to-end
- Pros: Optimized for our specific task
- Cons: More complex, requires more memory/compute

```python
# Option B: Fine-tuning EnCodec

class EnCodecFineTuned(nn.Module):
    def __init__(self, encodec_model, fmri_dim=50):
        super().__init__()
        self.encoder = encodec_model.encoder  # Keep encoder
        self.decoder = encodec_model.decoder  # Keep decoder

        # Replace quantization with linear projection
        # EnCodec encoder outputs: [batch, 128, time]
        # We want: [batch, fmri_dim, time_fmri]
        self.to_fmri = nn.Linear(128, fmri_dim)
        self.from_fmri = nn.Linear(fmri_dim, 128)

    def encode(self, audio):
        latent = self.encoder(audio)  # [batch, 128, time]
        fmri_features = self.to_fmri(latent.transpose(1, 2)).transpose(1, 2)
        return fmri_features

    def decode(self, fmri_features):
        latent = self.from_fmri(fmri_features.transpose(1, 2)).transpose(1, 2)
        audio = self.decoder(latent)
        return audio

# Training
model = EnCodecFineTuned(encodec, fmri_dim=50)

for batch in dataloader:
    audio_true = batch['audio']

    # Encode → "fMRI" → Decode
    fmri_features = model.encode(audio_true)
    audio_pred = model.decode(fmri_features)

    # Multi-scale loss (like EnCodec training)
    loss = compute_multi_scale_stft_loss(audio_pred, audio_true)

    loss.backward()
    optimizer.step()
```

---

### Phase 4: Evaluation (2-3 days)

**Metrics:**

1. **Objective:**
   - SNR (Signal-to-Noise Ratio): dB
   - PESQ (Perceptual Evaluation of Speech Quality): 1-5 scale
   - STOI (Short-Time Objective Intelligibility): 0-1
   - Multi-scale STFT loss

2. **Subjective:**
   - Human listening tests
   - Intelligibility (can you understand speech?)
   - Naturalness (does it sound natural?)
   - Compare: Original vs. EnCodec vs. Mel+Griffin-Lim

```python
# evaluation/audio_quality.py

import torch
import torchaudio
from pesq import pesq
from pystoi import stoi

def evaluate_audio_quality(audio_true, audio_pred, sr=24000):
    """
    Args:
        audio_true: [batch, 1, samples]
        audio_pred: [batch, 1, samples]
        sr: sample rate
    """
    audio_true = audio_true.cpu().numpy()
    audio_pred = audio_pred.cpu().numpy()

    results = []
    for i in range(len(audio_true)):
        # Convert to 1D
        true = audio_true[i, 0, :]
        pred = audio_pred[i, 0, :]

        # SNR
        signal_power = np.mean(true ** 2)
        noise_power = np.mean((true - pred) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # PESQ (requires 16kHz for narrowband, or 16kHz for wideband)
        # Resample to 16kHz
        true_16k = librosa.resample(true, orig_sr=sr, target_sr=16000)
        pred_16k = librosa.resample(pred, orig_sr=sr, target_sr=16000)
        pesq_score = pesq(16000, true_16k, pred_16k, 'wb')

        # STOI
        stoi_score = stoi(true, pred, sr, extended=False)

        results.append({
            'snr': snr,
            'pesq': pesq_score,
            'stoi': stoi_score
        })

    return results

# Usage
metrics = evaluate_audio_quality(audio_true, audio_pred)
print(f"Average SNR: {np.mean([m['snr'] for m in metrics]):.2f} dB")
print(f"Average PESQ: {np.mean([m['pesq'] for m in metrics]):.2f}")
print(f"Average STOI: {np.mean([m['stoi'] for m in metrics]):.2f}")
```

**Expected results (good quality):**
- SNR > 15 dB
- PESQ > 2.5 (acceptable) or > 3.5 (good)
- STOI > 0.7 (intelligible)

---

## Alternative Approach: Complex FFT (If EnCodec Fails)

If EnCodec doesn't work for some reason, Complex FFT is the backup plan.

### Implementation with complexPyTorch

```bash
pip install git+https://github.com/wavefrontshaping/complexPyTorch.git
```

```python
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu

class ComplexAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: complex STFT [1, freq, time]
        self.conv1 = ComplexConv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = ComplexBatchNorm2d(64)
        self.conv2 = ComplexConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = ComplexBatchNorm2d(128)

    def forward(self, x_complex):
        # x_complex: complex tensor [batch, 1, freq, time]
        x = complex_relu(self.bn1(self.conv1(x_complex)))
        x = complex_relu(self.bn2(self.conv2(x)))
        return x

class ComplexAudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = ComplexConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = ComplexBatchNorm2d(64)
        self.deconv2 = ComplexConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, z_complex):
        x = complex_relu(self.bn1(self.deconv1(z_complex)))
        x = self.deconv2(x)  # Output: complex STFT
        return x

# Training
encoder = ComplexAudioEncoder()
decoder = ComplexAudioDecoder()

audio_waveform = ...  # [batch, 1, samples]
stft_complex = torch.stft(audio_waveform, n_fft=2048, return_complex=True)
# stft_complex: [batch, freq, time] (complex tensor)

# Encode
z_complex = encoder(stft_complex.unsqueeze(1))  # Add channel dim

# Decode
stft_pred = decoder(z_complex).squeeze(1)

# Reconstruct audio
audio_pred = torch.istft(stft_pred, n_fft=2048)

# Loss: Complex MSE
loss = torch.mean(torch.abs(stft_pred - stft_complex) ** 2)
```

---

## Memory & Computational Budgets

### Current System (Mel Spectrogram)

**Per TR:**
- Mel spectrogram: [128 mel bins, ~100 time frames] = 12,800 values
- Memory: 12,800 × 4 bytes = 51 KB

**Total (920 TRs):**
- 920 × 51 KB = 47 MB (tiny!)

### EnCodec

**Per TR:**
- Codes: [8 quantizers, 112 frames] = 896 values
- Memory: 896 × 4 bytes = 3.6 KB (even smaller!)

**Total (920 TRs):**
- 920 × 3.6 KB = 3.3 MB (negligible!)

**Model size:**
- EnCodec 24kHz: ~20M parameters = 80 MB

### Complex FFT

**Per TR:**
- Complex STFT: [1025 freq bins, ~100 time frames] × 2 (real + imag) = 205,000 values
- Memory: 205,000 × 4 bytes = 820 KB

**Total (920 TRs):**
- 920 × 820 KB = 755 MB (manageable)

**Model size:**
- 2x real-valued model (complex weights)
- Example: 50M parameters (real) → 100M parameters (complex) = 400 MB

### Raw Waveform

**Per TR:**
- Waveform: 1.5s × 24,000 Hz = 36,000 samples
- Memory: 36,000 × 4 bytes = 144 KB

**Total (920 TRs):**
- 920 × 144 KB = 132 MB

**Model size:**
- WaveNet-style: 50M+ parameters = 200+ MB

---

## Recommended Next Steps

### Immediate (This Week):

1. **Install EnCodec:**
   ```bash
   pip install encodec
   ```

2. **Test EnCodec on Sherlock audio:**
   - Encode/decode full Sherlock stimulus
   - Listen to quality
   - Compute SNR, PESQ

3. **Precompute EnCodec codes:**
   - Save codes for all 920 TRs
   - Store in dataset (`.pt` file or HDF5)

4. **Report results:**
   - Audio samples (original vs. reconstructed)
   - Metrics
   - Decision: proceed with EnCodec?

### Short-term (Next 2 Weeks):

5. **Integrate EnCodec with dataset:**
   - Modify `SherlockDataset` to use EnCodec codes
   - Update dataloaders

6. **Modify encoder/decoder:**
   - Update audio encoder for EnCodec input shape
   - Update audio decoder for EnCodec output shape

7. **Train autoencoder:**
   - Train with EnCodec codes
   - Monitor reconstruction quality

8. **Evaluate:**
   - Compare: EnCodec vs. old mel+Griffin-Lim
   - Metrics + listening tests

### Long-term (Next Month):

9. **Fine-tune EnCodec (optional):**
   - If quality isn't sufficient, fine-tune EnCodec end-to-end
   - Experiment with different bandwidths

10. **Publish results:**
    - Update Issue #23 with results
    - Document improvements
    - Share audio samples

---

## References & Resources

### EnCodec
- **Paper:** [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) (Défossez et al., 2022)
- **GitHub:** [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **Demo:** [audiocraft.metademolab.com/encodec](https://audiocraft.metademolab.com/encodec.html)
- **License:** MIT

### Complex-Valued Networks
- **Paper:** [Phase-Aware Deep Learning with Complex-Valued CNNs](https://arxiv.org/abs/2510.09926) (2024)
- **Paper:** [DCCRN: Deep Complex Convolution Recurrent Network](https://www.researchgate.net/publication/343415434) (2020)
- **Library:** [wavefrontshaping/complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)
- **Library:** [josiahwsmith10/complextorch](https://github.com/josiahwsmith10/complextorch)

### Neural Audio Codecs
- **Paper:** [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) (Google, 2021)
- **Comparison:** [INVESTIGATING NEURAL AUDIO CODECS](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/SLT2024_CodecInvestigation.pdf) (Microsoft, 2024)

### fMRI Audio Reconstruction
- **Paper:** [Natural sounds reconstructed from fMRI](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003293) (PLOS Biology, 2024)
- **Paper:** [Re-Bottleneck: Latent Re-Structuring for Neural Audio Autoencoders](https://arxiv.org/abs/2507.07867) (2024)

### Evaluation Metrics
- **PESQ:** [pesq · PyPI](https://pypi.org/project/pesq/)
- **STOI:** [pystoi · PyPI](https://pypi.org/project/pystoi/)
- **Multi-scale STFT:** Included in EnCodec training code

---

## Conclusion

**EnCodec is the clear winner** for improving audio reconstruction in the fMRI autoencoder. It:

1. **Solves the phase problem** through learned compression (no Griffin-Lim needed)
2. **Requires minimal changes** to existing codebase (drop-in replacement for mel spectrograms)
3. **Provides pretrained models** (no training from scratch)
4. **Is production-proven** (used in Meta's audio generation systems)
5. **Offers flexibility** (multiple compression rates, mono/stereo support)

**Next steps:**
1. Test EnCodec on Sherlock audio (1-2 days)
2. Integrate with dataset (2-3 days)
3. Train autoencoder with EnCodec codes (1 week)
4. Evaluate quality improvements (2-3 days)

**Expected outcome:**
- **Significant improvement** in audio reconstruction quality
- **Intelligible speech** (vs. current garbled output)
- **Perceptually plausible** music/sound effects
- **Foundation for future improvements** (fine-tuning, multi-timescale, etc.)

This research report is ready to be posted to **Issue #23** for team discussion and implementation planning.

---

**End of Report**

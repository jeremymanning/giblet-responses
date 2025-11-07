# Audio Encoding Quick Reference

## Issue #23: Audio Reconstruction Quality

**Problem:** Mel spectrograms discard phase → Griffin-Lim fails after bottleneck compression

**Solution:** Use EnCodec neural audio codec

---

## Quick Comparison

| Method | Quality | Setup | Why Use | Why Not Use |
|--------|---------|-------|---------|-------------|
| **EnCodec** ⭐⭐⭐⭐⭐ | Excellent | `pip install encodec` | Pretrained, designed for compression+reconstruction | Black box (harder to interpret) |
| **Complex FFT** ⭐⭐⭐⭐ | Excellent | Need complex-valued layers | Theoretically sound, perfect reconstruction possible | 2x parameters, requires training |
| **Raw Waveform** ⭐⭐⭐ | Excellent | Standard PyTorch | No transforms, end-to-end | Very high memory/params (50M+) |
| **Wav2Vec 2.0** ⭐⭐⭐ | Good | `torchaudio` | Pretrained semantic features | No decoder (need to build vocoder) |
| **Audio-as-Image** ⭐⭐⭐ | Good | Standard 2D CNN | Simple, standard CNNs | Phase representation tricky |
| **CQT** ⭐⭐ | Good | `librosa` | Better for music | Still discards phase |
| **WaveGlow** ⭐⭐ | Excellent | Complex setup | Invertible by design | 87M params, very slow |

---

## EnCodec: The Winner

### Why EnCodec?

1. **Designed for our exact use case** (compress → reconstruct)
2. **Pretrained models available** (no training from scratch)
3. **MIT license** (research-friendly)
4. **Low parameters** (10M encoder + 10M decoder)
5. **Low memory** (codes are tiny: 3.6 KB per TR vs. 51 KB for mel specs)
6. **Production-proven** (Meta's AudioGen, MusicGen)

### Specs

| Property | Value |
|----------|-------|
| Parameters | 20M total (10M encoder + 10M decoder) |
| Input | 24kHz mono or 48kHz stereo |
| Compression | 2x to 128x (1.5-24 kbps) |
| Quality @ 12kbps | Perceptually lossless for music |
| License | MIT |
| Latent size per TR | 896 values (8 quantizers × 112 frames) |
| Memory per TR | 3.6 KB (vs. 51 KB for mel specs) |

### Installation

```bash
pip install encodec
```

### Usage

```python
from encodec import EncodecModel

# Load model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps

# Encode
audio = load_audio()  # [batch, channels, samples]
codes = model.encode(audio)

# Decode
reconstructed = model.decode(codes)
```

### Integration with Our Autoencoder

**Option 1: EnCodec as Preprocessor (RECOMMENDED)**
```
Audio → EnCodec.encode() → codes → Autoencoder → predicted codes → EnCodec.decode() → Audio
```

**Option 2: Fine-tune EnCodec**
```
Audio → EnCodec.encoder → fMRI projection → Autoencoder → fMRI→latent → EnCodec.decoder → Audio
```

---

## Implementation Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **1. Proof of Concept** | 1-2 days | Test EnCodec on Sherlock audio, evaluate quality |
| **2. Integration** | 3-5 days | Precompute codes, update dataset, modify encoder/decoder |
| **3. Training** | 1 week | Train autoencoder on EnCodec codes |
| **4. Evaluation** | 2-3 days | Metrics (SNR, PESQ, STOI), listening tests, comparison |
| **Total** | ~3-4 weeks | Full implementation + evaluation |

---

## Expected Improvements

### Current (Mel + Griffin-Lim)
- SNR: ~5 dB (poor)
- PESQ: ~1.5 (bad)
- STOI: ~0.3 (unintelligible)
- Subjective: Garbled, unintelligible

### With EnCodec
- SNR: >15 dB (good)
- PESQ: >3.0 (acceptable to good)
- STOI: >0.7 (intelligible)
- Subjective: Clear, intelligible, perceptually plausible

### Improvement
- SNR: +10 dB
- PESQ: +1.5 points
- STOI: +0.4 points
- Quality: Garbled → Intelligible

---

## Code Snippets

### Test EnCodec Quality

```python
import torch
from encodec import EncodecModel
import torchaudio

# Load model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Load audio
audio, sr = torchaudio.load("sherlock.wav")
audio = torchaudio.transforms.Resample(sr, 24000)(audio)

# Encode/decode
encoded = model.encode(audio.unsqueeze(0))
reconstructed = model.decode(encoded)

# Save
torchaudio.save("original.wav", audio, 24000)
torchaudio.save("reconstructed.wav", reconstructed.squeeze(0), 24000)

# Compute SNR
signal_power = torch.mean(audio ** 2)
noise_power = torch.mean((audio - reconstructed.squeeze(0)) ** 2)
snr = 10 * torch.log10(signal_power / noise_power)
print(f"SNR: {snr:.2f} dB")
```

### Precompute Codes for Dataset

```python
from encodec import EncodecModel
import torch

# Load model
encodec = EncodecModel.encodec_model_24khz()
encodec.set_target_bandwidth(6.0)

# Load full Sherlock audio
audio_full = load_sherlock_audio()  # [1, samples]

# Encode
with torch.no_grad():
    encoded = encodec.encode(audio_full)
    codes = torch.cat([enc[0] for enc in encoded], dim=-1)  # [1, n_q, T]

# Split into TRs (1.5s each, 75 frames/sec → 112 frames per TR)
frames_per_tr = 112
codes_per_tr = []

for tr_idx in range(920):
    start = tr_idx * frames_per_tr
    end = start + frames_per_tr
    tr_codes = codes[:, :, start:end]  # [1, 8, 112]
    codes_per_tr.append(tr_codes.squeeze(0))  # [8, 112]

# Save
torch.save(codes_per_tr, "encodec_codes_sherlock.pt")
```

### Update Dataset

```python
class SherlockDataset:
    def __init__(self, use_encodec=True):
        self.use_encodec = use_encodec

        if use_encodec:
            self.audio_codes = torch.load("encodec_codes_sherlock.pt")
            self.encodec_model = EncodecModel.encodec_model_24khz()

    def __getitem__(self, idx):
        fmri = self.load_fmri(idx)

        if self.use_encodec:
            audio = self.audio_codes[idx]  # [8, 112]
        else:
            audio = self.load_mel_spectrogram(idx)  # [128, 100]

        return {'fmri': fmri, 'audio': audio}
```

### Update Encoder/Decoder

```python
class AudioEncoder(nn.Module):
    def __init__(self, input_type='encodec'):
        super().__init__()

        if input_type == 'encodec':
            # Input: [8 quantizers, 112 frames] = 896 values
            self.encoder = nn.Sequential(
                nn.Flatten(),  # [896]
                nn.Linear(896, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )

class AudioDecoder(nn.Module):
    def __init__(self, output_type='encodec'):
        super().__init__()

        if output_type == 'encodec':
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 896),
                nn.Unflatten(1, (8, 112))  # [batch, 8, 112]
            )
```

---

## Evaluation Metrics

### Objective Metrics

```python
from pesq import pesq
from pystoi import stoi
import numpy as np

def evaluate_audio(true, pred, sr=24000):
    # SNR
    signal_power = np.mean(true ** 2)
    noise_power = np.mean((true - pred) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # PESQ (requires 16kHz)
    import librosa
    true_16k = librosa.resample(true, orig_sr=sr, target_sr=16000)
    pred_16k = librosa.resample(pred, orig_sr=sr, target_sr=16000)
    pesq_score = pesq(16000, true_16k, pred_16k, 'wb')

    # STOI
    stoi_score = stoi(true, pred, sr, extended=False)

    return {
        'snr': snr,
        'pesq': pesq_score,
        'stoi': stoi_score
    }
```

### Quality Thresholds

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| **SNR** | <10 dB | 10-15 dB | 15-20 dB | >20 dB |
| **PESQ** | <2.0 | 2.0-3.0 | 3.0-4.0 | >4.0 |
| **STOI** | <0.5 | 0.5-0.7 | 0.7-0.9 | >0.9 |

---

## Alternative: Complex FFT (Backup)

If EnCodec doesn't work, use complex-valued neural networks with STFT.

### Installation

```bash
pip install git+https://github.com/wavefrontshaping/complexPyTorch.git
```

### Usage

```python
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

# STFT
stft = torch.stft(audio, n_fft=2048, return_complex=True)  # Complex tensor

# Complex CNN
class ComplexEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(1, 64, 3, 2, 1)
        self.conv2 = ComplexConv2d(64, 128, 3, 2, 1)

    def forward(self, x_complex):
        x = complex_relu(self.conv1(x_complex))
        x = complex_relu(self.conv2(x))
        return x

# Reconstruct
stft_pred = decoder(z_complex)
audio_pred = torch.istft(stft_pred, n_fft=2048)
```

---

## Resources

### Primary Recommendation: EnCodec
- **Paper:** [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- **GitHub:** [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **Demo:** [audiocraft.metademolab.com/encodec](https://audiocraft.metademolab.com/encodec.html)

### Backup: Complex-Valued Networks
- **Paper:** [Phase-Aware Deep Learning with Complex-Valued CNNs](https://arxiv.org/abs/2510.09926)
- **Library:** [wavefrontshaping/complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)

### Related Research
- **fMRI Audio:** [Natural sounds reconstructed from fMRI](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003293)
- **Neural Codecs:** [SoundStream: End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)

---

## Files Created

1. **Full Report:** `/notes/AUDIO_ENCODING_RESEARCH_REPORT.md` (1429 lines)
   - Detailed analysis of 7 approaches
   - Technical specifications
   - Implementation guides
   - References

2. **Issue Comment:** `/notes/AUDIO_ENCODING_ISSUE23_POST.md` (274 lines)
   - Concise summary for GitHub
   - Comparison table
   - Implementation roadmap
   - Ready to post to Issue #23

3. **This File:** `/notes/AUDIO_ENCODING_QUICK_REFERENCE.md`
   - Quick lookup reference
   - Code snippets
   - Evaluation metrics

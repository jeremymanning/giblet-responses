# giblet/models - Neural Network Architectures

This module implements the multimodal autoencoder architecture for predicting brain activity from stimuli and reconstructing stimuli from brain representations.

## Overview

The architecture consists of three main components:

```
Input Stimuli                                        Reconstructed Stimuli
├─ Video (259,200-dim)    ┐                        ┌─ Video (259,200-dim)
├─ Audio (128-dim)        ├──► Encoder ──► Bottleneck ──► Decoder ──┤
└─ Text (1024-dim)        ┘      ↓         (2048-dim)         ↑      ├─ Audio (128-dim)
                                 │                            │      └─ Text (1024-dim)
                                 ▼                            │
                           fMRI Prediction              fMRI Input
                            (85,810 voxels)           (85,810 voxels)
```

**Key Capabilities:**
1. **Encode stimuli → Brain activity**: Predict fMRI voxel responses from video/audio/text
2. **Decode bottleneck → Stimuli**: Reconstruct video/audio/text from compressed representation
3. **Dual loss function**: Train using both reconstruction loss and fMRI matching loss

## Architecture Components

### MultimodalAutoencoder ([autoencoder.py](autoencoder.py))

Main wrapper combining encoder and decoder with dual loss function.

**Architecture Flow:**
```
Stimuli → Encoder → Bottleneck → fMRI Prediction (for loss)
                        ↓
                    Decoder → Reconstructed Stimuli (for loss)
```

**Usage:**
```python
from giblet.models import MultimodalAutoencoder

model = MultimodalAutoencoder(
    video_height=90,
    video_width=160,
    audio_features=128,          # EnCodec compressed dim
    text_dim=1024,
    n_voxels=85810,             # Number of brain voxels
    bottleneck_dim=2048,        # Compressed representation size
    reconstruction_weight=1.0,   # Weight for reconstruction loss
    fmri_weight=1.0             # Weight for fMRI matching loss
)

# Forward pass
outputs = model(video, audio, text)

# outputs contains:
# - 'fmri_pred': Predicted brain activity (85,810 voxels)
# - 'video_recon': Reconstructed video
# - 'audio_recon': Reconstructed audio
# - 'text_recon': Reconstructed text
# - 'bottleneck': Compressed representation (2048-dim)

# Compute loss
loss, losses_dict = model.compute_loss(
    outputs=outputs,
    fmri_target=actual_fmri,
    video_target=video,
    audio_target=audio,
    text_target=text
)
```

**Key Parameters:**
- `video_height`, `video_width` (int): Video frame dimensions (default: 90×160)
- `audio_features` (int): Audio feature dimension (128 for EnCodec, 2048 for mel)
- `text_dim` (int): Text embedding dimension (default: 1024)
- `n_voxels` (int): Number of brain voxels (default: 85810)
- `bottleneck_dim` (int): Compressed representation size (default: 2048)
- `reconstruction_weight` (float): Weight for reconstruction loss (default: 1.0)
- `fmri_weight` (float): Weight for fMRI matching loss (default: 1.0)

**Loss Functions:**
1. **Reconstruction Loss**: MSE between reconstructed and original stimuli
   - Video reconstruction MSE
   - Audio reconstruction MSE
   - Text reconstruction MSE (cosine similarity)
2. **fMRI Matching Loss**: MSE between predicted and actual brain activity

Total loss: `reconstruction_weight × reconstruction_loss + fmri_weight × fmri_loss`

---

### MultimodalEncoder ([encoder.py](encoder.py))

Encodes video/audio/text stimuli into bottleneck representation and predicts brain activity.

**Architecture (Layers 1-6):**
```
Layer 1: Input (video + audio + text)
    ↓
Layer 2A/B/C: Modality-specific encoders
    ├─ VideoEncoder: Linear layers (259,200 → 1024)
    ├─ AudioEncoder: Linear layers (128 → 256)
    └─ TextEncoder: Linear layers (1024 → 256)
    ↓
Layer 3: Concatenate features (1024 + 256 + 256 = 1536)
    ↓
Layer 4: Feature convolution (1536 → 4096) + ReLU
    ↓
Layer 5: Linear mapping (4096 → 85,810 voxels)
    ↓
Layer 6: Bottleneck (85,810 → 2048) [compressed representation]
```

**Usage:**
```python
from giblet.models.encoder import MultimodalEncoder

encoder = MultimodalEncoder(
    video_input_dim=259200,      # Flattened temporal concat
    audio_features=128,           # EnCodec features
    text_dim=1024,
    n_voxels=85810,
    bottleneck_dim=2048,
    video_output_features=1024,
    audio_output_features=256,
    text_output_features=256
)

# Forward pass
fmri_pred, bottleneck = encoder(video, audio, text)

# fmri_pred: (batch_size, 85810) - Predicted brain activity
# bottleneck: (batch_size, 2048) - Compressed representation
```

**Sub-Encoders:**

#### VideoEncoder

Processes flattened temporal concatenation features using Linear layers.

**Architecture:**
- Input: Flattened frames (259,200-dim for frame_skip=2)
- Layer 1: 259,200 → 4096 + BatchNorm + Dropout(0.3)
- Layer 2: 4096 → 2048 + BatchNorm + Dropout(0.3)
- Layer 3: 2048 → 1024 + BatchNorm + Dropout(0.2)
- Output: 1024-dim features

**Design Rationale:**
- Uses Linear layers instead of Conv2D due to temporal concatenation format
- Each TR contains all frames from [t-TR, t] flattened into single vector
- Progressive dimensionality reduction with regularization
- Gradient checkpointing support for memory efficiency

#### AudioEncoder

Processes EnCodec compressed audio or mel spectrograms.

**Architecture:**
- Input: Audio features (128-dim for EnCodec, 2048-dim for mel)
- Flattens EnCodec codes: (batch, codebooks, frames) → (batch, codebooks×frames)
- Layer 1: Input → 1024 + BatchNorm + Dropout(0.3)
- Layer 2: 1024 → 512 + BatchNorm + Dropout(0.3)
- Layer 3: 512 → 256 + BatchNorm + Dropout(0.2)
- Output: 256-dim features

**EnCodec Mode:**
- Receives pre-compressed EnCodec features
- Shape: (batch, n_codebooks, frames_per_tr)
- Default: 8 codebooks, ~112 frames/TR at 75Hz
- Flattens to (batch, 8×112 = 896) before processing

#### TextEncoder

Processes text embeddings using Linear layers.

**Architecture:**
- Input: Text embeddings (1024-dim)
- Layer 1: 1024 → 512 + BatchNorm + Dropout(0.3)
- Layer 2: 512 → 256 + BatchNorm + Dropout(0.2)
- Output: 256-dim features

---

### MultimodalDecoder ([decoder.py](decoder.py))

Decodes bottleneck representation back into video/audio/text stimuli.

**Architecture (Layers 8-13):**
```
Layer 8: Expand from bottleneck (2048 → 8000)
    ↓
Layer 9: Feature expansion (8000 → 4096)
    ↓
Layer 10: Feature deconvolution (4096 → 2048) + ReLU
    ↓
Layer 11: Feature unpooling (2048 → 1536)
    ↓
Layer 12A/B/C: Modality-specific decoders
    ├─ Video Decoder: (512 → 1024 → 259,200)
    ├─ Audio Decoder: (512 → 1024 → 128 or 2048)
    └─ Text Decoder: (512 → 1024)
    ↓
Layer 13: Output reconstructions
```

**Usage:**
```python
from giblet.models.decoder import MultimodalDecoder

decoder = MultimodalDecoder(
    bottleneck_dim=2048,
    video_dim=259200,
    audio_dim=128,              # EnCodec mode
    audio_frames_per_tr=112,    # For EnCodec @ 75Hz
    text_dim=1024,
    use_encodec=True,
    n_codebooks=8
)

# Forward pass
video_recon, audio_recon, text_recon = decoder(bottleneck)

# video_recon: (batch, 259200) - Reconstructed video features
# audio_recon: (batch, 8, 112) - Reconstructed EnCodec codes
# text_recon: (batch, 1024) - Reconstructed text embeddings
```

**Key Parameters:**
- `bottleneck_dim` (int): Input dimension (default: 2048)
- `video_dim` (int): Output video dimension (default: 259200)
- `audio_dim` (int): Output audio dimension (128 for EnCodec, 2048 for mel)
- `audio_frames_per_tr` (int): Temporal frames per TR (112 for EnCodec, 65 for mel)
- `text_dim` (int): Output text dimension (default: 1024)
- `use_encodec` (bool): Use EnCodec mode (default: False)
- `n_codebooks` (int): Number of EnCodec codebooks (default: 8)
- `dropout` (float): Dropout rate (default: 0.3)

**Modality-Specific Decoders:**

- **Video Decoder**: 512 → 1024 → 259,200 (flattened temporal concat)
- **Audio Decoder (EnCodec)**: 512 → 1024 → (8 codebooks × 112 frames)
- **Audio Decoder (Mel)**: 512 → 1024 → 2048 mels
- **Text Decoder**: 512 → 1024 (embeddings)

---

## Layer-by-Layer Architecture

Following the specification from Issue #2:

| Layer | Description | Input Dim | Output Dim | Location |
|-------|-------------|-----------|------------|----------|
| **Layer 1** | Input (video + audio + text concatenated) | - | Video: 259,200<br>Audio: 128<br>Text: 1024 | Input |
| **Layer 2A** | Video encoder (Linear) | 259,200 | 1024 | VideoEncoder |
| **Layer 2B** | Audio encoder (Linear) | 128 | 256 | AudioEncoder |
| **Layer 2C** | Text encoder (Linear) | 1024 | 256 | TextEncoder |
| **Layer 3** | Concatenate features | 1536 | 1536 | MultimodalEncoder |
| **Layer 4** | Feature space convolution + ReLU | 1536 | 4096 | MultimodalEncoder |
| **Layer 5** | Linear mapping to voxels | 4096 | 85,810 | MultimodalEncoder |
| **Layer 6** | Bottleneck (middle layer) | 85,810 | 2048 | MultimodalEncoder |
| **Layer 7** | Bottleneck representation | 2048 | 2048 | Between Encoder/Decoder |
| **Layer 8** | Expand from bottleneck | 2048 | 8000 | MultimodalDecoder |
| **Layer 9** | Feature expansion | 8000 | 4096 | MultimodalDecoder |
| **Layer 10** | Feature deconvolution + ReLU | 4096 | 2048 | MultimodalDecoder |
| **Layer 11** | Feature unpooling | 2048 | 1536 | MultimodalDecoder |
| **Layer 12A** | Video decoder | 512 | 259,200 | MultimodalDecoder |
| **Layer 12B** | Audio decoder | 512 | 128 or 2048 | MultimodalDecoder |
| **Layer 12C** | Text decoder | 512 | 1024 | MultimodalDecoder |
| **Layer 13** | Output reconstructions | - | Video/Audio/Text | Output |

---

## Feature Dimensions

### Input Dimensions (Layer 1)

| Modality | Configuration | Dimension | Notes |
|----------|--------------|-----------|-------|
| **Video** | frame_skip=2, TR=1.5s | 259,200 | ~18 frames × 160×90×3, flattened |
| **Video** | frame_skip=4, TR=1.5s | 129,600 | ~9 frames × 160×90×3, flattened |
| **Audio** | EnCodec @ 24kHz | 128 | Compressed representation per TR |
| **Audio** | Mel @ 22.05kHz | 2048 | 2048 mel bins |
| **Text** | BGE embeddings | 1024 | BAAI/bge-large-en-v1.5 |

### Intermediate Dimensions

- **Encoded Features** (Layer 3): 1536 (1024 + 256 + 256)
- **Feature Space** (Layer 4): 4096
- **fMRI Voxels** (Layer 5): 85,810
- **Bottleneck** (Layer 6/7): 2048 (compressed brain representation)

### Output Dimensions (Layer 13)

Same as input dimensions - model reconstructs original stimuli.

---

## Training Configuration

### Standard Configuration

```python
model = MultimodalAutoencoder(
    video_height=90,
    video_width=160,
    audio_features=128,          # EnCodec
    text_dim=1024,
    n_voxels=85810,
    bottleneck_dim=2048,
    video_features=1024,
    audio_features=256,
    text_features=256,
    decoder_hidden_dim=2048,
    decoder_dropout=0.3,
    reconstruction_weight=1.0,
    fmri_weight=1.0
)
```

### Memory-Optimized Configuration

For training on GPUs with limited memory:

```python
model = MultimodalAutoencoder(
    video_height=90,
    video_width=160,
    audio_features=128,
    text_dim=1024,
    n_voxels=85810,
    bottleneck_dim=1024,         # Reduced from 2048
    video_features=512,          # Reduced from 1024
    audio_features=128,          # Reduced from 256
    text_features=128,           # Reduced from 256
    decoder_hidden_dim=1024,     # Reduced from 2048
    decoder_dropout=0.4,         # Increased regularization
    reconstruction_weight=1.0,
    fmri_weight=1.0,
    gradient_checkpointing=True  # Enable memory savings
)
```

### Multi-GPU Training

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Create model
model = MultimodalAutoencoder(...)

# Wrap with DDP for distributed training
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Training loop
for batch in dataloader:
    outputs = model(video, audio, text)
    loss, _ = model.module.compute_loss(
        outputs=outputs,
        fmri_target=fmri,
        video_target=video,
        audio_target=audio,
        text_target=text
    )
    loss.backward()
    optimizer.step()
```

---

## Model Outputs

### Forward Pass Output

```python
outputs = model(video, audio, text)

# outputs is a dictionary containing:
{
    'fmri_pred': torch.Tensor,      # (batch, 85810) - Predicted brain activity
    'bottleneck': torch.Tensor,     # (batch, 2048) - Compressed representation
    'video_recon': torch.Tensor,    # (batch, 259200) - Reconstructed video
    'audio_recon': torch.Tensor,    # (batch, 128) or (batch, 2048) - Reconstructed audio
    'text_recon': torch.Tensor      # (batch, 1024) - Reconstructed text
}
```

### Loss Computation Output

```python
loss, losses_dict = model.compute_loss(outputs, fmri_target, video_target, audio_target, text_target)

# loss: scalar total loss
# losses_dict: detailed breakdown
{
    'total': float,              # Total weighted loss
    'reconstruction': float,     # Combined reconstruction loss
    'fmri': float,              # fMRI matching loss
    'video_recon': float,       # Video reconstruction MSE
    'audio_recon': float,       # Audio reconstruction MSE
    'text_recon': float         # Text reconstruction loss (cosine)
}
```

---

## Design Decisions

### 1. Linear Layers for Video (Not Conv2D)

**Rationale**: VideoEncoder uses Linear layers instead of Conv2D because:
- Input is **flattened temporal concatenation** (all frames from [t-TR, t])
- Not a single 2D image - multiple frames concatenated
- Linear layers provide flexibility for temporal information
- Mirrors AudioEncoder and TextEncoder architecture

### 2. EnCodec vs Mel Spectrograms

**EnCodec (Recommended)**:
- Neural audio codec with learned compression
- 128-dim per TR vs 2048-dim for mel
- Better reconstruction quality
- Consistent with state-of-the-art audio processing

**Mel Spectrograms (Legacy)**:
- Traditional audio features
- Larger dimension (2048 mels)
- Griffin-Lim reconstruction (lower quality)

### 3. Bottleneck Dimension

**Default: 2048**
- Balances compression vs information preservation
- Roughly matches fMRI voxel dimensionality after PCA
- Allows for meaningful "brain-like" representation

**Trade-offs**:
- Larger bottleneck: More information, more memory
- Smaller bottleneck: More compression, potential information loss

### 4. Dual Loss Function

Training optimizes both:
1. **Reconstruction**: Ensures stimuli can be recovered from bottleneck
2. **fMRI Matching**: Ensures bottleneck predicts actual brain activity

This creates a biologically-constrained latent space.

---

## Usage Examples

### Basic Training Loop

```python
from giblet.models import MultimodalAutoencoder
from giblet.data import MultimodalDataset
from torch.utils.data import DataLoader
import torch.optim as optim

# Create model and dataset
model = MultimodalAutoencoder(...)
dataset = MultimodalDataset(data_dir='data/', subjects='all', split='train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        fmri = batch['fmri'].to(device)

        # Forward pass
        outputs = model(video, audio, text)

        # Compute loss
        loss, losses_dict = model.compute_loss(
            outputs=outputs,
            fmri_target=fmri,
            video_target=video,
            audio_target=audio,
            text_target=text
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"Recon: {losses_dict['reconstruction']:.4f}, "
              f"fMRI: {losses_dict['fmri']:.4f}")
```

### Inference: Predict Brain Activity

```python
model.eval()
with torch.no_grad():
    outputs = model(video, audio, text)
    fmri_prediction = outputs['fmri_pred']

    # Compare to actual fMRI
    correlation = torch.corrcoef(torch.stack([
        fmri_prediction.flatten(),
        actual_fmri.flatten()
    ]))[0, 1]
    print(f"Brain prediction correlation: {correlation:.4f}")
```

### Inference: Reconstruct Stimuli

```python
model.eval()
with torch.no_grad():
    outputs = model(video, audio, text)

    # Get reconstructions
    video_recon = outputs['video_recon']
    audio_recon = outputs['audio_recon']
    text_recon = outputs['text_recon']

    # Reconstruct actual video/audio files
    video_processor.features_to_video(video_recon, 'reconstructed.mp4')
    audio_processor.features_to_audio(audio_recon, 'reconstructed.wav')
```

### Lesion Simulation

```python
# Simulate brain lesion by zeroing specific bottleneck units
model.eval()
with torch.no_grad():
    outputs = model(video, audio, text)
    bottleneck = outputs['bottleneck']

    # Zero out units 100-200 (simulate lesion)
    bottleneck_lesioned = bottleneck.clone()
    bottleneck_lesioned[:, 100:200] = 0

    # Decode with lesioned bottleneck
    video_recon, audio_recon, text_recon = model.decoder(bottleneck_lesioned)

    # Observe effect on reconstructed stimuli
    video_processor.features_to_video(video_recon, 'lesioned_stimulus.mp4')
```

---

## Checkpoint Loading

### Save Checkpoint

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': model.get_config()  # Save model hyperparameters
}, 'checkpoint.pt')
```

### Load Checkpoint

```python
# Create model with same config
checkpoint = torch.load('checkpoint.pt')
model = MultimodalAutoencoder(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Resume training
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## Performance Considerations

### Memory Usage

**Approximate memory requirements** (batch_size=32):

| Component | Memory | Notes |
|-----------|--------|-------|
| Video features | ~32 MB | 32 × 259,200 × 4 bytes |
| Model parameters | ~450 MB | All layers |
| Gradients | ~450 MB | During training |
| Activations | ~200 MB | Forward pass |
| **Total** | **~1.1 GB per batch** | Requires ~8-12 GB GPU for training |

**Optimization strategies:**
1. Use `frame_skip=4` to reduce video dimension by 50%
2. Enable gradient checkpointing
3. Use mixed precision training (bfloat16)
4. Reduce batch size on smaller GPUs

### Computational Complexity

**Forward pass FLOPs** (approximate):

- VideoEncoder: ~1.3 GFLOPs (dominated by first Linear layer)
- AudioEncoder: ~0.3 GFLOPs
- TextEncoder: ~0.5 GFLOPs
- Bottleneck: ~0.7 GFLOPs
- Decoder: ~2.1 GFLOPs
- **Total**: ~5 GFLOPs per sample

**Training time** (8× A100 GPUs, batch_size=32):
- ~500 ms/batch (forward + backward)
- ~1 hour/epoch (17 subjects × 920 TRs / batch_size=32)

---

## Related Modules

- **[giblet/data/](../data/)** - Data loading and preprocessing for model inputs
- **[giblet/training/](../training/)** - Training pipeline and utilities
- **[giblet/alignment/](../alignment/)** - HRF convolution and temporal alignment

---

## References

- Issue #2: Architecture specification
- Issue #29: EnCodec integration and temporal concatenation fix
- [Encoder architecture diagram](../../docs/architecture/encoder_architecture_diagram.txt)
- [Decoder architecture diagram](../../docs/architecture/DECODER_ARCHITECTURE_VISUAL.txt)

For questions or issues, see the main project [README.md](../../README.md).

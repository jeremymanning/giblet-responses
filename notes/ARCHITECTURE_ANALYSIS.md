# Autoencoder Architecture Analysis for Sherlock fMRI Project

## Actual Dataset Dimensions

### Video Stimulus
- **Resolution**: 640 × 360 pixels
- **Frame rate**: 25 fps
- **Duration**: 23:46 (1,426 seconds)
- **Total frames**: ~35,650 frames
- **Audio**: 44,100 Hz stereo

### fMRI Data
- **Spatial dimensions**: 61 × 73 × 61 voxels
- **Total voxels in 3D grid**: 271,633
- **Actual brain voxels**: ~83,300 (30.7% of volume)
  - 185,691 voxels are outside brain (always zero)
  - Only using voxels with signal in >95% of timepoints
- **Voxel size**: 3 × 3 × 3 mm
- **Temporal resolution (TR)**: 1.0 second
- **Timepoints**: 1,976 TRs (~33 minutes)
- **Note**: Duration mismatch suggests multiple presentations or recall period

---

## Input Layer Design

### Option 1: Raw Video Input (Full Resolution)

**Per frame:**
- Video: 640 × 360 × 3 (RGB) = **691,200 features**
- Audio: At 44.1 kHz for 0.04s (1 frame @ 25fps) = 1,764 samples × 2 channels = **3,528 features**
- **Total per frame**: 694,728 features

**Issues:**
- Extremely high dimensional
- Temporal mismatch: 25 fps video vs 1 Hz fMRI
- Need to aggregate ~25 video frames per fMRI TR

### Option 2: Downsampled Video (Recommended)

**Downsampling strategy:**
- Spatial: 640×360 → 160×90 (4× reduction) = 14,400 pixels × 3 = **43,200 features**
- Temporal: Average 25 frames → 1 per TR
- Audio: MFCC features (40 coefficients typical) or mel spectrogram
  - 40 MFCC features × 2 channels = **80 features**
  - OR: Average audio power in frequency bands (20 bands × 2 channels) = **40 features**

**Total per TR (Option 2a - with MFCCs):**
- Video: 43,200
- Audio: 80
- **Total: 43,280 features**

**Total per TR (Option 2b - simpler audio):**
- Video: 43,200
- Audio: 40
- **Total: 43,240 features** ✓ **RECOMMENDED**

### Option 3: Pre-extracted Features (Alternative)

Use pretrained network (e.g., ResNet, VGG) to extract features:
- Video: ResNet-50 final layer = **2,048 features per frame**
- Average over 25 frames → 2,048 per TR
- Audio: VGGish embeddings = **128 features**
- **Total: 2,176 features**

**Advantage**: Much smaller, captures semantic content
**Disadvantage**: Can't reconstruct original video

---

## Middle Layer (fMRI Representation)

**Actual brain voxels**: **83,300** (not 271,633!)

This is already much more reasonable since we're excluding the 69% of voxels outside the brain.

**Further reduction options:**
1. **Use all brain voxels**: 83,300 voxels
   - Pro: Maximum spatial resolution within brain
   - Con: Still quite large

2. **Spatial downsampling (2× each dimension)**:
   - Reduces to ~10,000-15,000 voxels
   - Still captures major brain structures
   - 5-8× reduction in parameters

3. **ROI-based (cortical only)**:
   - Gray matter only: ~50,000-60,000 voxels
   - Removes white matter, subcortical structures
   - More interpretable for cortical processing

4. **PCA compression**:
   - 83,300 → 5,000 components (captures ~95% variance)
   - Most efficient for reducing parameters

**RECOMMENDED**: Use all **83,300 brain voxels** directly (3× smaller than originally thought!)

---

## Autoencoder Architecture

### Architecture 1: Simple Dense Layers

```
Input: 43,240 features (downsampled video + audio)
  ↓
Encoder:
  Dense(10,000) + ReLU
  Dense(5,000) + ReLU
  Dense(1,000) + ReLU
  Dense(500) + ReLU
  Dense(271,633) → fMRI representation
  ↓
Decoder:
  Dense(500) + ReLU
  Dense(1,000) + ReLU
  Dense(5,000) + ReLU
  Dense(10,000) + ReLU
  Dense(43,240) → Reconstructed video/audio
```

**Parameters (UPDATED with 83,300 voxels):**
- Encoder:
  - 43,240 → 10,000: 432.4M
  - 10,000 → 5,000: 50M
  - 5,000 → 1,000: 5M
  - 1,000 → 500: 0.5M
  - 500 → 83,300: **41.7M**
- Decoder (symmetric): 530M
- **Total: ~575 MILLION parameters**

**Still large but NOW FEASIBLE!** 3× reduction from original estimate.

---

### Architecture 2: Convolutional Encoder/Decoder (RECOMMENDED)

Treat video frames as images and use convolutional layers:

```
Video Input: 160×90×3
  ↓
Encoder (Video path):
  Conv2D(32, 3×3, stride=2) → 80×45×32
  Conv2D(64, 3×3, stride=2) → 40×23×64
  Conv2D(128, 3×3, stride=2) → 20×12×128
  Conv2D(256, 3×3, stride=2) → 10×6×256
  Flatten → 15,360 features
  Dense(1,024)

Audio Input: 40 features
  Dense(256)

Combined:
  Concat(1,024 + 256) → 1,280
  Dense(512)
  Dense(271,633) → fMRI representation

Decoder:
  Dense(512)
  Dense(1,280)
  Split into video (1,024) and audio (256) paths

  Video path:
    Dense(15,360)
    Reshape(10, 6, 256)
    ConvTranspose2D(128, 3×3, stride=2) → 20×12×128
    ConvTranspose2D(64, 3×3, stride=2) → 40×23×64
    ConvTranspose2D(32, 3×3, stride=2) → 80×45×32
    ConvTranspose2D(3, 3×3, stride=2) → 160×90×3

  Audio path:
    Dense(256)
    Dense(40)
```

**Parameter Count (UPDATED with 83,300 voxels):**

**Encoder:**
- Conv layers: (32×3×3×3) + (64×3×3×32) + (128×3×3×64) + (256×3×3×128) ≈ 0.5M
- Dense(15,360 → 1,024): 15.7M
- Dense(40 → 256): 10K
- Dense(1,280 → 512): 0.66M
- Dense(512 → 83,300): **42.7M**

**Decoder:**
- Dense(83,300 → 512): **42.7M**
- Dense(512 → 1,280): 0.66M
- Dense(1,024 → 15,360): 15.7M
- Dense(256 → 40): 10K
- ConvTranspose layers: ≈ 0.5M

**Total: ~119 MILLION parameters**

**Excellent!** 2.6× reduction makes this very feasible. The bottleneck layers are now manageable.

---

### Architecture 3: Spatially-Aware fMRI Middle Layer (BEST APPROACH)

Instead of flattening fMRI, keep it 3D and use 3D convolutions:

```
Video Input: 160×90×3
  ↓
Encoder (Video):
  Conv2D(32, 3×3, stride=2) → 80×45×32
  Conv2D(64, 3×3, stride=2) → 40×23×64
  Conv2D(128, 3×3, stride=2) → 20×12×128
  Flatten → 30,720
  Dense(8,000)

Audio Input: 40
  Dense(256)

Combined:
  Concat → 8,256
  Dense(4,000)
  Dense(2,000)

Reshape to match fMRI grid:
  Dense(61×73×61×2 = 544,906)
  Reshape(61, 73, 61, 2)  ← Compressed fMRI representation

fMRI Processing:
  Conv3D(8, 3×3×3) → 61×73×61×8
  Conv3D(16, 3×3×3) → 61×73×61×16
  Conv3D(1, 1×1×1) → 61×73×61×1  ← Match actual fMRI

Decoder (reverse):
  Conv3D(16, 3×3×3) → 61×73×61×16
  Conv3D(8, 3×3×3) → 61×73×61×8
  Conv3D(2, 3×3×3) → 61×73×61×2
  Flatten → 544,906
  Dense(2,000)
  Dense(4,000)
  Dense(8,256)
  Split → video (8,000) + audio (256)

  Video: Dense(30,720) → Reshape → ConvTranspose layers
  Audio: Dense(40)
```

**Parameter Count:**

**Encoder:**
- Conv2D layers: 0.5M
- Dense(30,720 → 8,000): 246M
- Dense(40 → 256): 10K
- Dense(8,256 → 4,000): 33M
- Dense(4,000 → 2,000): 8M
- Dense(2,000 → 544,906): **1.09 BILLION** ← Still too big!

**Problem persists**: Dense layer to/from flattened fMRI is the bottleneck.

---

## Practical Solution: Hierarchical Compression

### Architecture 4: Multi-Stage Compression (FEASIBLE)

**Stage 1: Compress fMRI spatially first**
- Use PCA or learned compression on fMRI
- 271,633 voxels → 5,000 PCA components (captures ~95% variance)
- OR: Use brain parcellation (e.g., Schaefer 400 ROIs)

**Stage 2: Autoencoder with compressed middle layer**

```
Input: 43,240 (video/audio)
  ↓
Encoder:
  Dense(10,000)
  Dense(5,000)
  Dense(5,000) → Match PCA-compressed fMRI
  ↓
Decoder:
  Dense(5,000)
  Dense(10,000)
  Dense(43,240)
```

**Parameters:**
- 43,240 → 10,000: 432M
- 10,000 → 5,000: 50M
- 5,000 → 5,000: 25M
- (Decoder symmetric): 507M
- **Total: ~1 BILLION parameters**

Still large but feasible with modern GPUs.

---

## Recommended Approach

### Two-Stage Training:

**Stage 1: Learn fMRI Compression**
```
fMRI (271,633) → Encoder → Compressed (5,000) → Decoder → Reconstructed fMRI
```
- Parameters: ~700M
- Train to reconstruct fMRI with minimal loss

**Stage 2: Learn Stimulus-to-fMRI Mapping**
```
Video/Audio (43,240) → Encoder → Compressed fMRI (5,000) → Decoder → Reconstructed Video/Audio
```
- Parameters: ~1B
- Middle layer constrained to match compressed fMRI from Stage 1

**Stage 3: Lesion Studies**
- Fix specific compressed fMRI features to 0
- Observe changes in decoded video/audio
- Interpret which features correspond to which brain regions

---

## Computational Requirements

### Memory (for training one timepoint):
- Input batch (batch_size=32): 43,240 × 32 × 4 bytes ≈ 5.5 MB
- Activations through network: ~500 MB per batch
- Gradients: ~2× activation memory ≈ 1 GB
- Model parameters (1B): 4 GB (float32)
- **Total GPU memory needed: ~8-16 GB**

**Feasible on**:
- Single NVIDIA A100 (40GB)
- Single NVIDIA RTX 3090/4090 (24GB)
- Multiple V100s (16GB each)

### Training time estimate:
- ~2,000 TRs × 16 subjects = 32,000 samples
- 100 epochs × 32,000 / 32 batch_size = 100K iterations
- ~1 second per iteration
- **Total: ~28 hours on single GPU**

---

## Summary: Recommended Configuration

### Key Finding: Only 83,300 brain voxels (not 271,633!)

**Impact**: 3× reduction in middle layer size dramatically reduces model complexity.

### Three Viable Approaches:

#### Option A: Direct Dense Network (Simple)
| Component | Specification |
|-----------|--------------|
| **Input** | 160×90×3 video + 40 audio features = 43,240 |
| **Middle Layer** | 83,300 brain voxels (direct) |
| **Architecture** | Dense encoder/decoder with 4-5 layers |
| **Parameters** | ~575 million |
| **GPU Memory** | 12-16 GB |
| **Training Time** | ~1 day on single GPU |

#### Option B: Convolutional Network (RECOMMENDED)
| Component | Specification |
|-----------|--------------|
| **Input** | 160×90×3 video + 40 audio features = 43,240 |
| **Middle Layer** | 83,300 brain voxels |
| **Architecture** | Conv2D encoder/decoder for video + dense for fMRI |
| **Parameters** | ~119 million |
| **GPU Memory** | 8-12 GB |
| **Training Time** | ~12 hours on single GPU |

#### Option C: PCA-Compressed (Most Efficient)
| Component | Specification |
|-----------|--------------|
| **Input** | 160×90×3 video + 40 audio features = 43,240 |
| **Middle Layer** | 5,000 PCA components (from 83,300 voxels) |
| **Architecture** | Dense or convolutional |
| **Parameters** | ~50-100 million |
| **GPU Memory** | 8 GB |
| **Training Time** | ~6-8 hours on single GPU |

### Training Data
- **Samples**: 1,976 TRs × 16 subjects = 31,616 samples
- **Validation split**: 80/20 → 25,293 train / 6,323 validation

### Hardware Requirements
**Minimum**:
- GPU: NVIDIA RTX 3060 (12GB) or better
- RAM: 32 GB system memory
- Storage: 50 GB for data + 10 GB for models

**Recommended**:
- GPU: NVIDIA RTX 4090 (24GB) or A100 (40GB)
- RAM: 64 GB
- Storage: 100 GB SSD

**Your Setup (8× A6000 48GB)**:
- **EXCELLENT!** Each A6000 has 48GB - you can easily fit any of the proposed architectures
- Can train with much larger batch sizes (batch_size=128-256 instead of 32)
- Can use **data parallelism** across all 8 GPUs → **8× faster training**
- Expected training time with distributed training:
  - Option A (575M params): ~1.5 hours (vs 1 day on single GPU)
  - Option B (119M params): ~90 minutes (vs 12 hours on single GPU)
  - Option C (50-100M params): ~45 minutes (vs 6-8 hours on single GPU)
- You could even train multiple architectural variants simultaneously for comparison!

### Key Insight
By excluding non-brain voxels (69% of the volume!), we reduce the middle layer from 271,633 to 83,300 voxels. This makes **Option B (Convolutional Network with 119M parameters) highly feasible** without needing aggressive PCA compression. You can train directly on the full-resolution brain data!

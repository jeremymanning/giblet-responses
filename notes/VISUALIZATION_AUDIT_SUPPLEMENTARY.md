# Visualization Audit - Supplementary Analysis & Diagrams

---

## Model Architecture Comparison

### Issue #2 Logical Architecture (Intended)

```
INPUT
  ↓
LAYER 1: Modality preprocessing (implicit in encoder setup)
  ├─ LAYER 2A: Video → Conv2D layers → 1024-dim
  ├─ LAYER 2B: Audio → Conv1D layers → 256-dim
  └─ LAYER 2C: Text  → Linear layers → 256-dim
  ↓
LAYER 3: Pooling/Concatenation (1024+256+256 = 1536-dim)
  ↓
LAYER 4: Feature convolution + ReLU
  ↓
LAYER 5: Linear to brain voxels (85,810-dim)
  ↓
LAYER 6: Bottleneck compression (8000-dim) ← MIDDLE LAYER
  ↓
LAYER 7: Expand from bottleneck
  ↓
LAYER 8: Feature deconvolution + ReLU
  ↓
LAYER 9: Unpool features
  ├─ LAYER 10A: Video path → 43,200-dim
  ├─ LAYER 10B: Audio path → 2,048-dim
  └─ LAYER 10C: Text path → 1,024-dim
  ↓
LAYER 11: Reconstruction output

Total: 11 logical layers
```

### Actual PyTorch Module Structure (What exists)

```
MultimodalAutoencoder (1,983,999,154 params)
├─ Encoder (1,603,873,778 params)
│  ├─ VideoEncoder
│  │  ├─ conv1 (Conv2d): 3→32, 896 params
│  │  ├─ bn1 (BatchNorm2d): 64 params
│  │  ├─ conv2 (Conv2d): 32→64, 18,496 params
│  │  ├─ bn2 (BatchNorm2d): 128 params
│  │  ├─ conv3 (Conv2d): 64→128, 73,856 params
│  │  ├─ bn3 (BatchNorm2d): 256 params
│  │  ├─ conv4 (Conv2d): 128→256, 295,168 params
│  │  ├─ bn4 (BatchNorm2d): 512 params
│  │  └─ fc (Linear): 15360→1024, 15,729,664 params
│  │     [SUBTOTAL: 16,118,640 params]
│  │
│  ├─ AudioEncoder
│  │  ├─ conv1 (Conv1d): 1→32, 128 params
│  │  ├─ bn1 (BatchNorm1d): 64 params
│  │  ├─ conv2 (Conv1d): 32→64, 6,208 params
│  │  ├─ bn2 (BatchNorm1d): 128 params
│  │  ├─ conv3 (Conv1d): 64→128, 24,704 params
│  │  ├─ bn3 (BatchNorm1d): 256 params
│  │  ├─ conv4 (Conv1d): 128→256, 98,560 params
│  │  ├─ bn4 (BatchNorm1d): 512 params
│  │  └─ fc (Linear): 32768→256, 8,388,864 params
│  │     [SUBTOTAL: 8,519,424 params]
│  │
│  ├─ TextEncoder
│  │  ├─ fc1 (Linear): 1024→512, 524,800 params
│  │  ├─ bn1 (BatchNorm1d): 1,024 params
│  │  ├─ fc2 (Linear): 512→256, 131,328 params
│  │  └─ bn2 (BatchNorm1d): 512 params
│  │     [SUBTOTAL: 657,664 params]
│  │
│  ├─ feature_conv (Sequential)
│  │  ├─ [0] Linear: 1536→1536, 2,360,832 params
│  │  ├─ [1] BatchNorm1d: 3,072 params
│  │  ├─ [2] ReLU: 0 params
│  │  └─ [3] Dropout: 0 params
│  │     [SUBTOTAL: 2,363,904 params]
│  │
│  ├─ to_bottleneck (Sequential)
│  │  ├─ [0] Linear: 1536→4096, 6,295,552 params
│  │  ├─ [1] BatchNorm1d: 8,192 params
│  │  ├─ [2] ReLU: 0 params
│  │  ├─ [3] Dropout: 0 params
│  │  ├─ [4] Linear: 4096→8000, 32,776,000 params
│  │  ├─ [5] BatchNorm1d: 16,000 params
│  │  ├─ [6] ReLU: 0 params
│  │  └─ [7] Dropout: 0 params
│  │     [SUBTOTAL: 39,095,744 params]
│  │
│  └─ bottleneck_to_voxels (Sequential)
│     ├─ [0] Linear: 8000→16384, 131,088,384 params
│     ├─ [1] BatchNorm1d: 32,768 params
│     ├─ [2] ReLU: 0 params
│     ├─ [3] Dropout: 0 params
│     ├─ [4] Linear: 16384→85810, 1,405,996,850 params
│     └─ [5] BatchNorm1d: ... params
│        [SUBTOTAL: 1,537,118,146 params]
│
└─ Decoder (380,125,376 params)
   ├─ layer7 (Sequential)
   │  ├─ [0] Linear: 8000→2048, 16,386,048 params
   │  ├─ [1] BatchNorm1d: 4,096 params
   │  ├─ [2] ReLU: 0 params
   │  └─ [3] Dropout: 0 params
   │     [SUBTOTAL: 16,390,144 params]
   │
   ├─ layer8 (Sequential)
   │  ├─ [0] Linear: 2048→4096, 8,392,704 params
   │  ├─ [1] BatchNorm1d: 8,192 params
   │  ├─ [2] ReLU: 0 params
   │  └─ [3] Dropout: 0 params
   │     [SUBTOTAL: 8,400,896 params]
   │
   ├─ layer9 (Sequential)
   │  ├─ [0] Linear: 4096→8192, 33,562,624 params
   │  ├─ [1] BatchNorm1d: 16,384 params
   │  ├─ [2] ReLU: 0 params
   │  └─ [3] Dropout: 0 params
   │     [SUBTOTAL: 8,192,384 params]
   │
   ├─ layer10_video (Sequential)
   │  ├─ [0] Linear: 8192→8192, 67,117,056 params
   │  ├─ [1] BatchNorm1d: 16,384 params
   │  ├─ [2] ReLU: 0 params
   │  ├─ [3] Dropout: 0 params
   │  ├─ [4] Linear: 8192→4096, 33,558,528 params
   │  ├─ [5] BatchNorm1d: 8,192 params
   │  ├─ [6] ReLU: 0 params
   │  └─ [7] Dropout: 0 params
   │     [SUBTOTAL: 100,700,160 params]
   │
   ├─ layer10_audio (Sequential)
   │  ├─ [0] Linear: 8192→2048, 16,779,264 params
   │  ├─ [1] BatchNorm1d: 4,096 params
   │  ├─ [2] ReLU: 0 params
   │  ├─ [3] Dropout: 0 params
   │  ├─ [4] Linear: 2048→1024, 2,098,176 params
   │  ├─ [5] BatchNorm1d: 2,048 params
   │  ├─ [6] ReLU: 0 params
   │  └─ [7] Dropout: 0 params
   │     [SUBTOTAL: 18,883,584 params]
   │
   ├─ layer10_text (Sequential)
   │  ├─ [0] Linear: 8192→2048, 16,779,264 params
   │  ├─ [1] BatchNorm1d: 4,096 params
   │  ├─ [2] ReLU: 0 params
   │  ├─ [3] Dropout: 0 params
   │  ├─ [4] Linear: 2048→2048, 4,196,352 params
   │  ├─ [5] BatchNorm1d: 4,096 params
   │  ├─ [6] ReLU: 0 params
   │  └─ [7] Dropout: 0 params
   │     [SUBTOTAL: 20,987,808 params]
   │
   ├─ layer11_video (Sequential)
   │  ├─ [0] Linear: 4096→43200, 176,990,400 params
   │  └─ [1] Sigmoid: 0 params
   │     [SUBTOTAL: 176,990,400 params]
   │
   ├─ layer11_audio (Linear): 1024→2048, 2,099,200 params
   │
   └─ layer11_text (Linear): 2048→1024, 2,098,176 params
```

**Total: 99 named modules, 52 with parameters**

---

## Visualization Layer Extraction Results

### What Gets Extracted (52 layers)

| # | Path | Type | Params | Input→Output |
|---|------|------|--------|--------------|
| 1 | encoder.video_encoder.conv1 | Conv2d | 896 | 3→32 channels |
| 2 | encoder.video_encoder.bn1 | BatchNorm2d | 64 | - |
| 3 | encoder.video_encoder.conv2 | Conv2d | 18,496 | 32→64 channels |
| 4 | encoder.video_encoder.bn2 | BatchNorm2d | 128 | - |
| 5 | encoder.video_encoder.conv3 | Conv2d | 73,856 | 64→128 channels |
| 6 | encoder.video_encoder.bn3 | BatchNorm2d | 256 | - |
| 7 | encoder.video_encoder.conv4 | Conv2d | 295,168 | 128→256 channels |
| 8 | encoder.video_encoder.bn4 | BatchNorm2d | 512 | - |
| 9 | encoder.video_encoder.fc | Linear | 15,729,664 | 15360→1024 |
| 10 | encoder.audio_encoder.conv1 | Conv1d | 128 | 1→32 channels |
| ... | [8 more audio layers] | ... | ... | ... |
| 19 | encoder.text_encoder.fc1 | Linear | 524,800 | 1024→512 |
| ... | [3 more text layers] | ... | ... | ... |
| 23 | encoder.feature_conv.0 | Linear | 2,360,832 | 1536→1536 |
| 24 | encoder.feature_conv.1 | BatchNorm1d | 3,072 | - |
| 25 | encoder.to_bottleneck.0 | Linear | 6,295,552 | 1536→4096 |
| ... | [9 bottleneck layers] | ... | ... | ... |
| 32-52 | decoder.* | Various | Varies | See structure |

**Key Observation**: Every layer (conv→bn→relu sequence) is shown as individual nodes

---

## Parallel Structure Analysis

### What IS Parallel:

```
ENCODER:
                    Video Path              Audio Path              Text Path
                        |                       |                        |
Input: [video]      Input: [audio]         Input: [text]
       ↓                   ↓                      ↓
  [Conv2d blocks]    [Conv1d blocks]       [Linear blocks]
     (16.1M)            (8.5M)               (657K)
       ↓                   ↓                      ↓
  Output: 1024-dim   Output: 256-dim       Output: 256-dim
       |___________________|__________________|
                           |
                    Concatenate (1536-dim)
                           |
                    Feature Conv
                           |
                    Bottleneck (8000-dim)

DECODER:
                    Bottleneck (8000-dim)
                           |
                    Layer 7: 2048-dim
                           |
                    Layer 8: 4096-dim
                           |
                    Layer 9: 8192-dim
                           |
            _________________|_________________
            |               |                 |
    Layer 10 Video     Layer 10 Audio    Layer 10 Text
    8192→4096         8192→2048→1024     8192→2048
    (100.7M)          (18.9M)            (20.9M)
    (parallel!)       (parallel!)        (parallel!)
            |               |                 |
    Layer 11 Video    Layer 11 Audio    Layer 11 Text
    4096→43200         1024→2048         2048→1024
    (176.9M)           (2.1M)            (2.1M)
            |               |                 |
            |_______________|_________________|
                            |
                      RECONSTRUCTION
```

### Current Visualization Shows:

```
52 Layers in vertical stack:
[Layer 1] encoder.video_encoder.conv1
[Layer 2] encoder.video_encoder.bn1
...
[Layer 9] encoder.video_encoder.fc
[Layer 10] encoder.audio_encoder.conv1  ← Appears sequential, not parallel!
...
[Layer 18] encoder.audio_encoder.fc
[Layer 19] encoder.text_encoder.fc1
...
```

**Problem**: User sees layers 2A, 2B, 2C rendered one after another (sequential) rather than side-by-side (parallel)

---

## Orientation Issue Deep Dive

### Current Implementation (Vertical Only)

```python
# Figure setup
figsize=(16, 24)  # Width=16", Height=24" → Portrait
ax.set_xlim(0, 10)
ax.set_ylim(0, len(layers) + 2)  # 0 to 54

# Rendering loop
y_pos = len(layers)  # Start at top
for layer in layers:
    # Draw at (x=~5, y=y_pos)
    rect = FancyBboxPatch(
        (5 - size/40, y_pos - 0.4),  # X-width control
        size/20, 0.8,                  # Width × Height
    )
    ax.add_patch(rect)

    y_pos -= 1  # Next layer below
```

**Result**:
```
Y-Axis: 54 layers (top to bottom)
X-Axis: 10 units total, layers centered at x=5, width ∝ log(params)

        x=0    x=5    x=10
        |------|------|
        |      [L1]   |  y=52
        |      [L2]   |  y=51
        |      [L3]   |  y=50
        |      [L4]   |  y=49
        |     [L5]    |  y=48
        |      ...    |
        |     [L52]   |  y=0
```

### How Horizontal Would Look

```
X-Axis: 52 layers (left to right)
Y-Axis: 10 units total, layers centered at y=5, height ∝ log(params)

       y=10 |—————————————————————————————————
            |
       y=5  |[L1][L2][L3][L4]...[L52]
            |
       y=0  |—————————————————————————————————
            |x=0        x=20         x=52
```

**Required Changes**:
1. Swap X and Y axis limits
2. Change layer loop to increment x_pos instead of y_pos
3. Update all FancyBboxPatch coordinates
4. Rotate text positioning 90°
5. Adjust figsize to landscape

**Code Locations Needing Modification**:
- Line 161: Add `orientation='vertical'` parameter
- Line 228-229: Conditional axis setup
- Line 240: Conditional position initialization
- Lines 250-286: Coordinate swap in rectangles/polygons
- Lines 295-311: Text rotation/positioning
- Line 343: Conditional tight_layout

---

## Parameter Distribution Analysis

### By Component

```
Total: 1,983,999,154 parameters

Encoder: 1,603,873,778 (80.8%)
├─ bottleneck_to_voxels:  1,537,118,146 (77.4% of encoder)  [8000→85810]
├─ to_bottleneck:           39,095,744 (2.4%)               [1536→8000]
├─ video_encoder:           16,118,640 (1.0%)               [video processing]
├─ feature_conv:             2,363,904 (0.1%)               [1536→1536]
├─ audio_encoder:            8,519,424 (0.5%)               [audio processing]
└─ text_encoder:               657,664 (0.0%)               [text processing]

Decoder: 380,125,376 (19.2%)
├─ layer11_video:          176,990,400 (46.5% of decoder)   [4096→43200]
├─ layer10_video:          100,700,160 (26.5%)              [8192→4096]
├─ layer10_text:            20,987,808 (5.5%)               [parallel path]
├─ layer10_audio:           18,883,584 (5.0%)               [parallel path]
├─ layer9:                   8,192,384 (2.2%)               [unpool]
├─ layer8:                   8,400,896 (2.2%)               [deconvolution]
├─ layer11_audio:            2,099,200 (0.6%)               [1024→2048]
├─ layer11_text:             2,098,176 (0.6%)               [2048→1024]
└─ layer7:                  16,390,144 (4.3%)               [expand]
```

### Visual Representation (Logarithmic)

```
Layer sizes (log10 scale):
10.0 ▓                    ← 131M+ (bottleneck_to_voxels.0)
9.5  ▓                    ← 1.4B (bottleneck_to_voxels.4) ← LARGEST!
9.0  ▓▓
8.5  ▓▓▓▓
8.0  ▓▓▓▓▓▓▓
7.5  ▓▓▓▓▓▓▓▓▓▓▓▓▓
7.0  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
6.5  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
6.0  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
5.5  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
5.0  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     0  5  10  15  20  25  30  35  40  45  50
```

**Key Observation**: Huge variance in layer sizes!
- 3 orders of magnitude between biggest (1.4B) and smallest (64)
- Logarithmic sizing makes sense, but still produces large size ranges (20-200 units in visualization)

---

## Extraction Logic Deep Dive

### What Gets Skipped

```python
for name, module in model.named_modules():
    if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
        continue  # ← SKIPPED ITEMS

Skipped (11 items):
1. '' (the root MultimodalAutoencoder itself)
2. 'encoder' (MultimodalEncoder container)
3. 'encoder.video_encoder' (VideoEncoder container)
4. 'encoder.audio_encoder' (AudioEncoder container)
5. 'encoder.text_encoder' (TextEncoder container)
6. 'encoder.feature_conv' (Sequential container)
7. 'encoder.to_bottleneck' (Sequential container)
8. 'encoder.bottleneck_to_voxels' (Sequential container)
9. 'decoder' (MultimodalDecoder container)
10. 'decoder.layer7' (Sequential container)
11-16. 'decoder.layer8/9/10_video/audio/text/layer11_video' (Sequential/container)
```

**Why**: The code intentionally skips containers to get only leaf modules (actual parameter holders).

**Consequence**: No way to group related layers back together without additional logic.

---

## Comparison with Alternative Approaches

### Option 1: Current Implementation (52 leaf layers)
✓ Accurate parameter counting
✓ Simple extraction logic
✓ Color coding helps identify modalities
✗ Parallel structure invisible
✗ Only vertical orientation
✗ No architectural abstraction

### Option 2: Hierarchical Visualization (11 logical layers)
✓ Matches Issue #2 specification exactly
✓ Parallel paths clearly visible
✓ Could support horizontal layout
✗ Requires custom architecture parser
✗ Not generalizable to other models
✗ Would lose some detail

### Option 3: Hybrid Approach (11 layers with expandable detail)
✓ Shows logical architecture
✓ Can expand to show leaf layers if needed
✓ Flexible layout
✗ Complex implementation
✗ Requires interactive visualization (HTML/D3/Plotly)
✗ PDF export becomes problematic

### Option 4: Use Netron or Similar
✓ Professional-grade visualization
✓ Interactive exploration
✓ Handles arbitrary architectures
✗ Not PyTorch-native
✗ May not highlight parameter counts
✗ Separate tool (not integrated)

---

## Summary Table

| Issue | Location | Root Cause | Fix Difficulty | Impact |
|-------|----------|-----------|---------------|----|
| Parallel structure invisible | Lines 239-313 | No X-offset for parallel paths | MODERATE | Major - affects understanding |
| Only vertical orientation | Lines 161, 228-229, 240+ | Hardcoded Y-axis progression | HIGH | Minor - workaround exists |
| 52 vs 11 layers shown | Lines 21-77 | Extraction focuses on leaf modules | VERY HIGH | Medium - needs architectural parser |
| No grouping of Sequential layers | Line 44 | Sequential containers explicitly skipped | MODERATE | Medium - loss of structure |
| Layer names too long | Lines 291-293 | Names include full path | LOW | Low - truncation works |

---

## File Audit Summary

**File**: `/Users/jmanning/giblet-responses/giblet/utils/visualization.py`
- **Lines**: 417 total
- **Functions**: 5 main functions + 2 helpers
- **Issues**: 3 major, 1 moderate
- **Code Quality**: Good for current scope, limited extensibility

**Testing Notes**:
- Tested with real MultimodalAutoencoder model
- All 52 layers extracted correctly
- Total parameters match: 1,983,999,154
- Color coding works as expected
- PDF/PNG export functional

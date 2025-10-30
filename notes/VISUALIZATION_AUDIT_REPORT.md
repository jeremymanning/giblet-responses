# Visualization Implementation Audit Report

**Date:** 2025-10-29
**File:** `/Users/jmanning/giblet-responses/giblet/utils/visualization.py`
**Model:** MultimodalAutoencoder with 1,983,999,154 parameters

---

## Executive Summary

The current visualization implementation has **three critical issues**:

1. **Parallel Structure Lost**: Encoder/decoder parallel paths (video/audio/text) are flattened into a linear sequence
2. **Orientation Limitation**: Only vertical layout supported; no horizontal option
3. **Architectural Mismatch**: Visualization shows 52 leaf layers instead of the logical 11-layer architecture from Issue #2

**Assessment**: The current approach can be **partially fixed** for parallel structure visualization, but would require substantial refactoring for proper horizontal layout and architectural abstraction.

---

## Detailed Findings

### 1. How Layers Are Extracted From PyTorch Model

**Location**: `_get_layer_info()` function (lines 21-77)

**Current Logic**:
```python
for name, module in model.named_modules():
    # Skip container modules and the model itself
    if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
        continue

    # Count parameters
    params = sum(p.numel() for p in module.parameters(recurse=False))

    if params > 0:  # Only include layers with parameters
        # Extract layer info
```

**What This Extracts**:
- **52 layers total** (all leaf modules with parameters)
- Each Conv2d, Linear, BatchNorm included separately
- Sequential containers are skipped, preventing architectural grouping

**Example Output Structure**:
```
1. encoder.video_encoder.conv1      | Conv2d     | 896 params
2. encoder.video_encoder.bn1        | BatchNorm2d | 64 params
3. encoder.video_encoder.conv2      | Conv2d     | 18,496 params
...
9. encoder.video_encoder.fc         | Linear     | 15,729,664 params
10. encoder.audio_encoder.conv1     | Conv1d     | 128 params
...
52. decoder.layer11_text            | Linear     | 2,098,176 params
```

**Root Cause of Issue**: The extraction focuses on leaf layers (smallest units) rather than logical architectural units. This is technically correct but loses semantic meaning.

---

### 2. Parallel Layer Problem: Why 2A/B/C Are Not Shown

**Issue Specification**: Issue #2 defines architecture as:
- Layer 2A: Video encoder path (convolutions on video frames)
- Layer 2B: Audio encoder path (convolutions on mel spectrogram)
- Layer 2C: Text encoder path (linear on embeddings)
- These should process in **parallel** and then merge

**Current Visualization Behavior**:
The 27 layers from the three encoder paths appear consecutively:
- Layers 1-9: video_encoder (conv1→bn1→conv2→bn2→conv3→bn3→conv4→bn4→fc)
- Layers 10-18: audio_encoder (conv1→bn1→conv2→bn2→conv3→bn3→conv4→bn4→fc)
- Layers 19-22: text_encoder (fc1→bn1→fc2→bn2)

**Why They're Not Shown Parallel**:

1. **Named Module Structure**: Parallel paths ARE present in model.named_modules():
   ```python
   encoder.video_encoder    # VideoEncoder class
   encoder.audio_encoder    # AudioEncoder class
   encoder.text_encoder     # TextEncoder class
   ```

2. **Extraction Doesn't Group Them**: The code extracts leaf layers individually
   - No tracking of which layers belong to which encoder
   - No hierarchical grouping logic
   - Result: All 52 layers treated as a flat sequence

3. **Visualization Renders Vertically**: Lines 239-313 render layers top-to-bottom
   - Each layer gets a `y_pos` that decreases by 1
   - No X-axis variation for parallel paths
   - All layers stack vertically regardless of logical structure

**The Parallel Structure IS Detectable**:
```
Encoder video paths extracted: 9 layers (names contain 'video_encoder')
Encoder audio paths extracted: 9 layers (names contain 'audio_encoder')
Encoder text paths extracted: 4 layers (names contain 'text_encoder')
Decoder video paths extracted: 4 layers (names contain 'layer10_video')
Decoder audio paths extracted: 4 layers (names contain 'layer10_audio')
Decoder text paths extracted: 4 layers (names contain 'layer10_text')
```

**Solution Difficulty**: **MODERATE** - Would need to:
- Parse layer names to identify modality (`'video_encoder' in name`, etc.)
- Calculate X positions for parallel lanes
- Draw layers side-by-side with connecting lines to merge points
- Update the matplotlib drawing logic (lines 240-313)

---

### 3. Diagram Rendering: Vertical vs Horizontal Orientation

**Location**: `create_network_diagram()` function (lines 154-354)

**Current Hardcoded Orientation**:
- Figure size: `figsize=(16, 24)` - width=16", height=24" (vertical portrait)
- Axis limits: `ax.set_xlim(0, 10)` and `ax.set_ylim(0, len(layers) + 2)`
  - X-axis tiny (0-10)
  - Y-axis large (0-52+)
- Layer rendering: `y_pos = len(layers); y_pos -= 1` after each layer
  - Decreases Y position (top to bottom)
  - X position fixed (lines 250, 263, 277, etc. use `5 - size/40` to `5 + size/40`)

**Why Horizontal Layout Doesn't Work**:

1. **Figure Dimensions**: Portrait orientation built into figsize default
2. **Layout Calculations**: All layer positions use Y-axis for progression
3. **Text Positioning**: Layer names/dimensions positioned relative to Y changes (lines 295-311)
4. **Axis Configuration**: Hard X-limit of 10 can't accommodate 52 layers horizontally

**Code Locations Needing Changes**:
- Line 161: `figsize=(16, 24)` - Parameter hint only
- Line 228-229: `ax.set_xlim(0, 10)` and `ax.set_ylim(0, len(layers) + 2)`
- Line 240: `y_pos = len(layers)` - Loop counter
- Lines 250-286: Rectangle and 3D effect drawing uses Y for positioning
- Lines 295-311: Text positioning tied to Y coordinates

**Solution Difficulty**: **HIGH** - Would require:
- Adding `orientation` parameter to function
- Conditional logic for X vs Y axis progression
- Recalculating all rectangle/polygon coordinates
- Adjusting text positioning for rotated layout
- Potential matplotlib projection for true side-by-side rendering

---

### 4. Architecture Representation vs. Issue #2 Specification

**Issue #2 Specification** (from encoder.py comments):
```
Layer 1: Input (video + audio + text concatenated after processing)
Layer 2A: Video convolutions (Conv2D)
Layer 2B: Audio convolutions (Conv1D)
Layer 2C: Text linear mapping
Layer 3: Pool all features
Layer 4: Feature space convolution + ReLU
Layer 5: Linear mapping to brain voxels
Layer 6: Bottleneck convolution (middle layer ~5000-10000 dims)
```

**Current Visualization Shows** (52 layers):
```
Actual Model Structure:
- encoder.video_encoder: 9 leaf layers (conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4, fc)
- encoder.audio_encoder: 9 leaf layers (conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4, fc)
- encoder.text_encoder: 4 leaf layers (fc1, bn1, fc2, bn2)
- encoder.feature_conv: 3 leaf layers (linear, bn, [ReLU dropped])
- encoder.to_bottleneck: 5 leaf layers
- encoder.bottleneck_to_voxels: 5 leaf layers
- decoder.layer7-9: 11 leaf layers
- decoder.layer10_*: 12 leaf layers (parallel video/audio/text)
- decoder.layer11_*: 3 layers (final outputs)
```

**The Mismatch**:

| Issue #2 Logical Layer | Actual Modules | Visualization Shows |
|---|---|---|
| Layer 1 (Input) | `video+audio+text` concatenation | No explicit node |
| Layer 2A (Video) | `VideoEncoder` class (9 layers) | 9 sequential layers |
| Layer 2B (Audio) | `AudioEncoder` class (9 layers) | 9 sequential layers |
| Layer 2C (Text) | `TextEncoder` class (4 layers) | 4 sequential layers |
| Layer 3 (Pool) | `torch.cat()` operation | Not shown |
| Layer 4 (Feature conv) | `feature_conv` Sequential (3 layers) | 3 sequential layers |
| Layer 5+6 (Bottleneck) | `to_bottleneck` + `bottleneck_to_voxels` | 10 sequential layers |

**Root Cause**: PyTorch `named_modules()` doesn't distinguish between:
- Logical "layers" as in architecture papers
- Actual module instances (which include BatchNorm after every Conv)
- Functional operations (concatenation, pooling) that create no parameters

**Solution Difficulty**: **VERY HIGH** - Would require:
- Custom architecture parser (not standard PyTorch)
- Manual layer grouping logic
- Handling non-parameter operations
- Semantic understanding of model intent
- Possible model-specific code (less general)

---

## What's Working Correctly

1. **Parameter Counting**: `_get_layer_info()` correctly counts all parameters (1,983,999,154 total)
2. **Color Coding**: `_get_layer_color()` properly identifies modalities by name pattern
3. **Size Scaling**: `_calculate_layer_size()` logarithmic scaling prevents huge variance
4. **3D Effects**: Matplotlib 3D-style rectangles render cleanly (lines 248-286)
5. **PDF/PNG Export**: Both formats work correctly (lines 345-350)
6. **Dimension Extraction**: Input/output dimensions extracted for Conv/Linear/BatchNorm (lines 59-73)

---

## Root Cause Analysis

### Issue 1: Parallel Structure Not Shown
**Root Cause**: Extraction focuses on leaf modules, not logical groupings
- `named_modules()` returns all modules, flat hierarchy
- No grouping logic to identify "this video_encoder" vs "this audio_encoder"
- Rendering loop treats all 52 layers identically

**Fix Difficulty**: **MODERATE**
- Can parse layer names for 'video'/'audio'/'text'
- Can calculate X offsets for parallel lanes
- Requires ~50-100 lines of additional code

### Issue 2: No Horizontal Orientation
**Root Cause**: Hardcoded Y-axis progression, fixed axis limits
- Entire drawing logic assumes Y increases per layer
- X-axis limit of 10 is arbitrary (visualizes ~8 units wide)
- Would need significant refactoring

**Fix Difficulty**: **HIGH**
- Requires conditional logic throughout rendering code
- Would need parameter propagation through helper functions
- Text positioning especially complex

### Issue 3: Architectural Mismatch
**Root Cause**: Fundamental difference between PyTorch module representation and paper architecture
- PyTorch 52 leaf layers vs. Issue #2 logical 11 layers
- Non-parameter operations (concatenation) aren't modules
- Would need custom parsing

**Fix Difficulty**: **VERY HIGH**
- Requires semantic understanding of model
- No generic solution across different architectures
- Would become model-specific code

---

## Recommendations

### Short Term (Easy Fixes)
1. **Update Documentation**: Add comment explaining 52 layers vs. 11 logical layers
2. **Enhanced Title**: Show encoder/decoder parameter breakdown in title
3. **Better Color Coding**: Add more specific colors for sequential layers within encoders

### Medium Term (Moderate Effort)
1. **Parallel Layer Detection**: Parse names to identify video/audio/text paths
2. **Side-by-Side Visualization**: Render parallel paths in adjacent X lanes
3. **Merge Point Visualization**: Show where paths join (layer 3 pooling, decoder layer 10)

### Long Term (High Effort)
1. **Horizontal Layout Option**: Add `orientation='vertical'|'horizontal'` parameter
2. **Hierarchical View**: Group Sequential containers as single logical units
3. **Custom Layer Abstraction**: Create architecture-specific visualization classes
4. **Alternative Libraries**: Consider `netron` for architecture visualization instead

---

## Code Quality Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Parameter Counting | Excellent | Correct enumeration and summing |
| Color Scheme | Good | Clear modality identification |
| PDF/PNG Export | Good | Works reliably |
| Documentation | Fair | Docstrings present but don't explain 52 vs 11 issue |
| Maintainability | Fair | Tight coupling between extraction and rendering |
| Extensibility | Poor | Hard to add parallel path support or change orientation |
| Performance | Good | Handles 52 layers smoothly |

---

## Specific Code Locations

**Critical Issues**:
1. Layer extraction (lines 21-77): Treats all leaf modules equally
2. Visualization loop (lines 239-313): Assumes Y-axis progression
3. Rendering setup (lines 225-229): Hardcoded aspect ratio

**Supporting Logic**:
- Color detection (lines 115-151): Works well
- Size calculation (lines 80-112): Appropriate logarithmic scaling
- File export (lines 345-350): No issues

---

## Conclusion

The visualization implementation is **functionally sound** for its current scope (showing all layers, color-coded by modality, with 3D styling). However, it has **architectural limitations** that prevent showing the true logical structure of the model.

**Current State**: Shows 52 individual leaf layers in vertical sequence
**Desired State** (Issue #2): Shows 11 logical layers with parallel paths 2A/B/C clearly visible

**Feasibility of Fixes**:
- Parallel paths: ✓ Moderate effort, significant improvement
- Horizontal orientation: ~ High effort, nice-to-have
- Logical layer abstraction: ✗ Very high effort, would require custom architecture parser

**Recommended Next Step**: Implement parallel path detection and side-by-side rendering (medium effort, high impact on clarity).

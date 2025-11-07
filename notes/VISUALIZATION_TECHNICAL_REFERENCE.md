# Visualization Technical Reference

Quick lookup guide for visualization code issues and fixes.

---

## Issue 1: Parallel Structure Not Shown

### Problem Code Location
**File**: `giblet/utils/visualization.py`
**Function**: `create_network_diagram()` (lines 239-313)
**Loop**: Layer rendering loop

```python
# Line 239-313: Current loop treats all layers identically
for layer in layers:
    size = _calculate_layer_size(layer['params'], sizing_mode)
    color = _get_layer_color(layer['name'], layer['type'])

    # Draw at fixed x=5
    rect = mpatches.FancyBboxPatch(
        (5 - size/40, y_pos - 0.4),  # X-position hardcoded!
        ...
    )
    y_pos -= 1  # Y decrements for each layer
```

### Solution Architecture

```python
# Pseudocode for parallel path detection and rendering

def detect_modality(layer_name: str) -> str:
    """Detect if layer is video/audio/text/other."""
    if 'video_encoder' in layer_name or 'layer10_video' in layer_name:
        return 'video'
    elif 'audio_encoder' in layer_name or 'layer10_audio' in layer_name:
        return 'audio'
    elif 'text_encoder' in layer_name or 'layer10_text' in layer_name:
        return 'text'
    else:
        return 'main'  # Non-modality-specific (encoder/decoder stem)

# Modified rendering loop
modality_x_offsets = {
    'main': 2.5,      # Center
    'video': 0.5,     # Left lane
    'audio': 2.5,     # Center lane
    'text': 4.5       # Right lane
}

for layer in layers:
    modality = detect_modality(layer['name'])
    x_base = modality_x_offsets[modality]

    # Draw at appropriate X position for modality
    rect = mpatches.FancyBboxPatch(
        (x_base - size/40, y_pos - 0.4),  # Use modality-specific X
        size/20, 0.8,
        ...
    )
```

### Implementation Steps

1. Add `detect_modality()` helper function (~10 lines)
2. Create `modality_x_offsets` dictionary in `create_network_diagram()` (~5 lines)
3. Extract modality in layer loop (~3 lines)
4. Use modality-specific X offset in FancyBboxPatch (~1 line change)
5. Add connecting lines at merge/split points (~20 lines)
6. Update X-axis limits from 10 to account for lanes (~1 line)

**Estimated Total**: 40-60 lines added/modified

---

## Issue 2: Orientation Support

### Problem Code Locations

**Location 1: Function signature and parameter**
```python
# Line 154-162
def create_network_diagram(
    model: nn.Module,
    output_path: str,
    legend: bool = True,
    sizing_mode: str = 'logarithmic',
    show_dimension: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 24),  # ← Hardcoded
    dpi: int = 300
) -> Path:
```

Need to add: `orientation: str = 'vertical'` parameter

**Location 2: Axis setup**
```python
# Lines 225-229
fig = plt.figure(figsize=figsize, facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, len(layers) + 2)
ax.axis('off')
```

Need conditional logic:
```python
if orientation == 'vertical':
    figsize = figsize  # (16, 24)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 2)
else:  # horizontal
    figsize = (figsize[1], figsize[0])  # Swap to (24, 16)
    ax.set_xlim(0, len(layers) + 2)
    ax.set_ylim(0, 10)
```

**Location 3: Layer position tracking**
```python
# Line 240
y_pos = len(layers)
```

Need conditional:
```python
if orientation == 'vertical':
    y_pos = len(layers)
    x_pos = None
else:
    x_pos = 0
    y_pos = None
```

**Location 4: Rectangle coordinates (8 places)**
```python
# Lines 250, 263, 277 (FancyBboxPatch)
# Lines 262-266, 276-280 (Polygon vertices)
```

All use `(5 - size/40, y_pos - 0.4)` format.

Need conditional coordinate generation:
```python
if orientation == 'vertical':
    rect_x = 5 - size/40
    rect_y = y_pos - 0.4
else:
    rect_x = x_pos - 0.4
    rect_y = 5 - size/40
```

**Location 5: Position increment**
```python
# Line 313
y_pos -= 1
```

Need conditional:
```python
if orientation == 'vertical':
    y_pos -= 1
else:
    x_pos += 1
```

**Location 6: Text positioning (2 places)**
```python
# Lines 295, 308
ax.text(5 + size/40 + offset + 0.2, y_pos, ...)
ax.text(5 + size/40 + offset + 0.2, y_pos - 0.2, ...)
```

Need conditional:
```python
if orientation == 'vertical':
    text_x = 5 + size/40 + offset + 0.2
    text_y = y_pos
else:
    text_x = x_pos
    text_y = 5 + size/40 + offset + 0.2
```

**Estimated Total**: 150-200 lines modified/added

---

## Issue 3: Logical Layer Abstraction

### The Problem

PyTorch modules ≠ Logical layers from Issue #2

```
Issue #2 Layer 2 (Video encoder):
  INPUT: 3×160×90 video frame
  OUTPUT: 1024-dim features
  Contains: Conv2d → BatchNorm → ReLU → Conv2d → ... → Linear

PyTorch representation:
  MultimodalEncoder.video_encoder (class: VideoEncoder)
    - conv1 (Conv2d)
    - bn1 (BatchNorm2d)
    - conv2 (Conv2d)
    - ... (9 modules total)
    - fc (Linear)

Current visualization:
  Shows 9 separate rectangles, one for each module
  User must understand they're all part of "Layer 2A"
```

### Why It's Hard

1. **No semantic grouping**: PyTorch doesn't distinguish between:
   - A layer block: Conv→BatchNorm→ReLU (3 modules = 1 logical layer)
   - A standalone layer: Linear (1 module = 1 logical layer)

2. **Non-parameter operations**: Concatenation, ReLU, Dropout, Pooling
   - Don't create modules (no parameters)
   - But are important to show logically
   - Can't be extracted from `named_modules()`

3. **Model-specific**: Would need to know:
   - Which Conv2d layers group with which BatchNorm
   - Where pooling/concatenation happens
   - What the "intended" architecture is
   - This varies by model

### Potential Solutions

**Option A: Heuristic Grouping** (~100 lines)
```python
def group_layers_heuristically(layers):
    """Group layers based on common patterns."""
    groups = []
    i = 0
    while i < len(layers):
        # If Conv→BN→next is the pattern, group them
        if (i + 1 < len(layers) and
            'conv' in layers[i]['type'].lower() and
            'batchnorm' in layers[i+1]['type'].lower()):
            groups.append([layers[i], layers[i+1]])
            i += 2
        else:
            groups.append([layers[i]])
            i += 1
    return groups
```

**Issue**: Fragile, breaks with variations

**Option B: Name-based Grouping** (~80 lines)
```python
def group_layers_by_name(layers):
    """Group layers within same named module."""
    groups_dict = {}
    for layer in layers:
        # Extract parent module name
        parent = layer['name'].rsplit('.', 1)[0]
        if parent not in groups_dict:
            groups_dict[parent] = []
        groups_dict[parent].append(layer)
    return list(groups_dict.values())
```

**Issue**: Groups too aggressively (all video_encoder layers together)

**Option C: Custom Architecture Parser** (~200+ lines)
```python
# For MultimodalAutoencoder specifically
def parse_giblet_architecture(model):
    """Parse GIBlet-specific architecture."""
    layers = {
        'Layer 1': ['Input preparation'],
        'Layer 2A': [modules in video_encoder],
        'Layer 2B': [modules in audio_encoder],
        'Layer 2C': [modules in text_encoder],
        'Layer 3': ['Concatenation (no module)'],
        ...
    }
    return layers
```

**Issue**: Only works for this model, not generalizable

---

## Code References for Each Component

### Extraction Phase
**File**: `giblet/utils/visualization.py`
**Function**: `_get_layer_info(model)` (lines 21-77)

Key line:
```python
# Line 44: Skip containers
if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
    continue
```

### Color Selection Phase
**Function**: `_get_layer_color(layer_name, layer_type)` (lines 115-151)

Pattern matching used (can be enhanced):
```python
if 'video' in layer_name.lower():
    return (0.2, 0.4, 0.8, 0.7)  # Blue
elif 'audio' in layer_name.lower():
    return (0.8, 0.4, 0.2, 0.7)  # Orange
elif 'text' in layer_name.lower():
    return (0.2, 0.8, 0.4, 0.7)  # Green
```

### Size Calculation Phase
**Function**: `_calculate_layer_size(params, sizing_mode, min_size, max_size)` (lines 80-112)

Logarithmic scaling:
```python
# Line 106-108
log_params = np.log10(params + 1)
normalized = min(log_params / 8.0, 1.0)
return min_size + (max_size - min_size) * normalized
```

### Rendering Phase
**Function**: `create_network_diagram()` (lines 154-354)

Critical sections:
- Axis setup: lines 225-229
- Layer loop: lines 239-313
- Legend: lines 315-332
- Export: lines 345-350

---

## Quick Fix Checklist

### For Parallel Path Support
- [ ] Add `detect_modality()` function
- [ ] Add `modality_x_offsets` dictionary
- [ ] Parse modality in layer loop
- [ ] Update FancyBboxPatch X coordinate
- [ ] Add merging lines between modality lanes
- [ ] Update X-axis limit (0-10 → 0-7 or similar)
- [ ] Update X-axis label to show modality zones
- [ ] Test with real model
- [ ] Generate comparison images before/after

### For Horizontal Orientation
- [ ] Add `orientation` parameter to function signature
- [ ] Add conditional axis setup (lines 225-229)
- [ ] Add conditional position tracking (line 240)
- [ ] Add conditional rectangle coordinates (8 places)
- [ ] Add conditional position increment (line 313)
- [ ] Add conditional text positioning (2 places)
- [ ] Add conditional figsize swap
- [ ] Test both vertical and horizontal
- [ ] Update docstring with new parameter

### For Logical Layer Grouping
- [ ] Design architecture specification format
- [ ] Implement parser for MultimodalAutoencoder
- [ ] Map 52 layers to 11 logical layers
- [ ] Update visualization to show groups
- [ ] Consider interactive expansion of groups
- [ ] Document limitation (model-specific)

---

## Testing Strategy

### Parallel Path Support
```python
# Test that parallel paths are visible
model = MultimodalAutoencoder()
diagram = create_network_diagram(model, 'test.png')

# Manual inspection:
# - Video layers should be in left lane
# - Audio layers should be in center lane
# - Text layers should be in right lane
# - Merger point should have connecting lines
```

### Horizontal Orientation
```python
# Test both orientations
model = MultimodalAutoencoder()
create_network_diagram(model, 'vertical.png', orientation='vertical')
create_network_diagram(model, 'horizontal.png', orientation='horizontal')

# Manual inspection:
# - Both should render correctly
# - Text should be readable in both
# - Layers should fit in canvas
```

### Logical Layer Grouping
```python
# Test that 11 logical layers are shown
# Test that layer names match Issue #2 spec
# Test that total parameters match
```

---

## Performance Considerations

Current performance is good (52 layers render instantly).

Proposed changes impact:
- **Parallel paths**: O(n) for modality detection (negligible)
- **Horizontal layout**: No performance impact (same drawing calls)
- **Logical grouping**: O(n) for grouping (still negligible for 52 layers)

No optimization needed for feature set.

---

## Integration Points

### Dependencies
- `matplotlib`: For rendering
- `torch.nn`: For module types
- `pathlib`: For file paths

### Used By
- Training pipeline (could log diagram)
- Analysis notebooks
- Documentation generation

### Affects
- Visual understanding of model
- Debugging architecture issues
- Paper figures/presentations

---

## Backward Compatibility

### Parallel Path Support
- ✓ Backward compatible (same API, enhanced rendering)
- ✓ Default orientation unchanged (vertical)
- ✓ Existing code calls will work unchanged

### Horizontal Orientation
- ✓ Backward compatible (new optional parameter)
- ✓ Default remains vertical
- ✓ No breaking changes

### Logical Layer Grouping
- × Breaking change (changes layer count from 52 to 11)
- × Requires API updates
- × May affect downstream analysis
- Consider as separate "v2" function

---

## Future Enhancements (Low Priority)

1. **Interactive Visualization**: Plotly/HTML version with hover details
2. **Multi-page PDF**: Logical layers as separate pages
3. **Custom Color Schemes**: User-provided color map
4. **Dimension Flow Visualization**: Show tensor shapes changing through network
5. **ONNX/TorchScript Support**: Visualize exported models
6. **Comparison Mode**: Side-by-side visualization of two models
7. **Statistics Overlay**: Memory usage, FLOP estimates per layer
8. **Attention/Activation Maps**: Overlay module importance

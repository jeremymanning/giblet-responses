# Comprehensive Visualization Library Research for Issue #18

**Date**: 2025-10-29
**Task**: Research and compare neural network visualization libraries
**Focus**: Find best library for showing multimodal autoencoder with parallel pathways

## Architecture Requirements

The MultimodalAutoencoder has the following structure that needs visualization:
- **Total**: 11 layers, ~2.0B parameters
- **Encoder (Layers 1-6)**:
  - Layer 1: Input (video + audio + text)
  - **Layer 2A**: Video encoder (Conv2D) - parallel branch
  - **Layer 2B**: Audio encoder (Conv1D) - parallel branch
  - **Layer 2C**: Text encoder (Linear) - parallel branch
  - Layer 3: Pooled features (concatenation)
  - Layer 4: Feature convolution
  - Layer 5: Map to voxels (85,810)
  - Layer 6: Bottleneck (8,000 dim) - MIDDLE LAYER
- **Decoder (Layers 7-11)**:
  - Layer 7: Expand from bottleneck
  - Layer 8: Feature deconvolution
  - Layer 9: Unpool features
  - **Layer 10A**: Video decoder - parallel branch
  - **Layer 10B**: Audio decoder - parallel branch
  - **Layer 10C**: Text decoder - parallel branch
  - Layer 11: Output (video + audio + text)

## Key Visualization Requirements
1. Show parallel processing (Layers 2A/B/C and 10A/B/C)
2. Horizontal or flexible orientation
3. Publication quality output
4. Works with PyTorch
5. Handle large models (2B parameters)
6. Clear representation of bottleneck layer

## Research Progress

### Status Summary
- [x] PlotNeuralNet - RESEARCHED (manual work required)
- [x] visualtorch - TESTED (failed - incompatible with multi-input)
- [x] torchview - TESTED (✅ WORKS - RECOMMENDED)
- [x] hiddenlayer - TESTED (failed - incompatible with PyTorch 2.x)
- [x] nnv - RESEARCHED (too simple)
- [x] existing custom solution (partial - works but no horizontal layout)

### FINAL RECOMMENDATION: torchview ⭐

## Libraries to Investigate

### 1. PlotNeuralNet (User Preference)
- **URL**: https://github.com/HarisIqbal88/PlotNeuralNet
- **Type**: LaTeX/TikZ based
- **Expected Strengths**: Publication quality, professional style
- **Expected Challenges**: May require manual layer specification, not Python native
- **Status**: Not yet tested

### 2. visualtorch
- **URL**: https://visualtorch.readthedocs.io/
- **Type**: Python library
- **Features**: LeNet style plots, Graph style plots
- **Status**: Not yet tested

### 3. torchview
- **URL**: https://github.com/mert-kurttutan/torchview
- **Type**: Python library
- **Features**: Computational graph visualization
- **Status**: Not yet tested

### 4. hiddenlayer
- **URL**: https://github.com/waleedka/hiddenlayer
- **Type**: Python library
- **Features**: Graph-based visualization
- **Status**: Not yet tested

### 5. nnv
- **URL**: https://github.com/martisak/dotnets
- **Type**: Python library with Graphviz
- **Features**: Graphviz-based rendering
- **Status**: Not yet tested

### 6. Existing Custom Solution
- **Location**: giblet/utils/visualization.py
- **Status**: Implemented and tested (25 tests passing)
- **Strengths**:
  - Works with PyTorch natively
  - Handles 2B parameters
  - Publication quality PDF/PNG output
  - 3D layered view
  - Logarithmic sizing
- **Limitations**:
  - Vertical layout only
  - Doesn't explicitly show parallel branches
  - Sequential layer representation

## Research Methodology

For each library, I will:
1. Install the library
2. Review documentation for parallel branch support
3. Create test visualization using real MultimodalAutoencoder
4. Evaluate against requirements:
   - Can show parallel branches (Layers 2A/B/C, 10A/B/C)?
   - Horizontal orientation support?
   - Handles large models (2B params)?
   - Output quality (PDF/SVG)?
   - Ease of customization?
   - Active maintenance (recent commits)?
   - Documentation quality?
5. Generate sample diagram if possible
6. Document findings in CSV and this file

## Testing Plan

```python
# Standard test code for each library
from giblet.models import create_autoencoder

# Create real 2.0B parameter model
model = create_autoencoder()

# Try to visualize with each library
# (specific code varies by library)
```

## Expected Timeline
- Research each library: ~30 min each
- Generate sample diagrams: ~1 hour
- Comparison and documentation: ~30 min
- Total: ~4 hours

## Notes
- Previous work (2025-10-29) implemented custom solution using matplotlib
- User specifically mentions PlotNeuralNet preference
- Focus on showing parallel pathways is critical
- Horizontal orientation would be novel/useful

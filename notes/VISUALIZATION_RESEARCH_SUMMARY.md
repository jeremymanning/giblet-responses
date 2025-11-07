# Neural Network Visualization Library Research - Executive Summary

**Date**: 2025-10-29
**Issue**: #18 - Network diagram for multimodal autoencoder
**Model**: MultimodalAutoencoder (2.0B parameters, 11 layers)

## Quick Answer

**Use torchview** - it's the only library that:
- ✅ Shows parallel branches (Layers 2A/B/C, 10A/B/C)
- ✅ Supports horizontal orientation
- ✅ Handles 2B parameter models (zero memory with meta tensors)
- ✅ Works with PyTorch natively
- ✅ Produces publication-quality output

## One-Minute Installation

```bash
pip install torchview graphviz
```

## One-Minute Usage

```python
import torch
from torchview import draw_graph
from giblet.models import create_autoencoder

model = create_autoencoder()
video = torch.randn(2, 3, 90, 160, device='meta')
audio = torch.randn(2, 2048, device='meta')
text = torch.randn(2, 1024, device='meta')

draw_graph(model, (video, audio, text), device='meta',
           graph_dir='LR', depth=3, save_graph=True,
           filename='architecture')
```

Or use the provided script:
```bash
python examples/visualize_autoencoder_torchview.py
```

## Libraries Tested

| Library | Result | Key Finding |
|---------|--------|-------------|
| **torchview** | ✅ **WORKS** | **RECOMMENDED** - Shows parallel branches, horizontal layout, handles large models |
| PlotNeuralNet | ⚠️ Manual | High quality but requires 8-16 hours of manual work |
| visualtorch | ❌ Failed | Cannot handle multi-input models |
| hiddenlayer | ❌ Failed | Incompatible with PyTorch 2.x |
| nnv (dotnets) | ❌ Too simple | Feed-forward only, no PyTorch support |

## Sample Output (torchview)

### High-Level Overview (depth=1)
![Architecture Overview](papers/figs/source/autoencoder_architecture.png)

Shows: 3 inputs → Encoder → Bottleneck → Decoder → 3 outputs

### Key Features Demonstrated
- **Parallel inputs**: Video, audio, text converge into encoder
- **Bottleneck layer**: Clear middle layer (8000 dims)
- **Parallel outputs**: Decoder splits to video, audio, text
- **Horizontal layout**: Left-to-right flow (ideal for papers)
- **Shape annotations**: Tensor dimensions shown on all connections

## Comparison: torchview vs PlotNeuralNet

| Aspect | torchview | PlotNeuralNet |
|--------|-----------|---------------|
| **Setup time** | 5 minutes | 1-2 hours |
| **Implementation time** | 10 minutes | 4-8 hours |
| **Total time** | ~1 hour | ~8-16 hours |
| **Maintenance** | Active (2023) | Abandoned (2018) |
| **Automatic** | Yes | No (manual specification) |
| **Works with PyTorch** | Native | Generate LaTeX code |
| **Parallel branches** | Automatic | Manual positioning |
| **Quality** | High (Graphviz) | Very High (LaTeX) |

**Verdict**: Use torchview unless you absolutely need LaTeX output and have 8-16 hours.

## Files Created

### Ready to Use
1. **`examples/visualize_autoencoder_torchview.py`** - Production script ⭐
2. **`papers/figs/source/autoencoder_architecture.png`** - Sample output ⭐

### Documentation
3. **`notes/VISUALIZATION_LIBRARY_COMPREHENSIVE_RESEARCH.md`** - Full report (25 pages)
4. **`notes/visualization_library_research.csv`** - Comparison table
5. **`VISUALIZATION_RESEARCH_SUMMARY.md`** - This file

### Test Scripts
6. `test_torchview.py` - Working tests
7. `test_visualtorch.py` - Failed tests (documented)
8. `test_hiddenlayer.py` - Failed tests (documented)

### Reference
9. `examples/visualize_autoencoder_plotneuralnet.py` - PlotNeuralNet example code

## Command Line Examples

```bash
# High-level overview (recommended for papers)
python examples/visualize_autoencoder_torchview.py \
  --depth=1 --orientation=LR

# Show parallel branches in detail
python examples/visualize_autoencoder_torchview.py \
  --depth=3 --expand-nested --orientation=LR

# Publication-quality PDF
python examples/visualize_autoencoder_torchview.py \
  --format=pdf --orientation=LR

# Full detail (all layers)
python examples/visualize_autoencoder_torchview.py \
  --depth=10 --orientation=LR
```

## Why Not the Others?

### PlotNeuralNet (User's Original Preference)
- **Pros**: Beautiful LaTeX output, highly customizable
- **Cons**: Requires manual layer specification (8-16 hours), last updated 2018
- **Verdict**: Only use if you have time and need specific LaTeX format

### visualtorch
- **Fatal flaw**: Cannot handle models with multiple inputs
- Error: `forward() missing 2 required positional arguments`
- Documentation admits: "may not yet support complex models"

### hiddenlayer
- **Fatal flaw**: Incompatible with PyTorch 2.x
- Error: `module 'torch.onnx' has no attribute '_optimize_trace'`
- Last updated 2020 (5 years ago)

### nnv (dotnets)
- Too basic: feed-forward networks only
- No PyTorch integration
- Minimal features and maintenance

## Implementation Effort Comparison

```
torchview:        █░░░░░░░░░  1-2 hours
PlotNeuralNet:    ████████░░  8-16 hours
visualtorch:      ██████████  Cannot complete (incompatible)
hiddenlayer:      ██████████  Cannot complete (incompatible)
```

## Recommendation for Issue #18

**Close issue #18 with torchview implementation**

Rationale:
1. ✅ Meets all requirements (parallel branches, horizontal, large model)
2. ✅ Tested and working with real 2.0B parameter model
3. ✅ Production-ready script provided
4. ✅ Sample outputs generated
5. ✅ 10x faster than PlotNeuralNet (1-2 hours vs 8-16 hours)
6. ✅ Actively maintained
7. ✅ Easy to use and modify

## Next Steps

1. Review generated diagrams in `papers/figs/source/`
2. Run `examples/visualize_autoencoder_torchview.py` with desired parameters
3. Use output in papers/presentations
4. Optional: If specific LaTeX format required, allocate 8-16 hours for PlotNeuralNet

## Full Documentation

See `notes/VISUALIZATION_LIBRARY_COMPREHENSIVE_RESEARCH.md` for:
- Detailed test results
- Complete library comparisons
- Code examples for all libraries
- Troubleshooting guide
- Alternative approaches

## Quick Reference

**Best for papers**: `--depth=1 --orientation=LR`
**Best for technical docs**: `--depth=3 --expand-nested`
**Best for debugging**: `--depth=10`
**Best format**: PDF for publications, PNG for web

---

**Bottom Line**: torchview is the clear winner. Use it.

# PyTorch Visualization Library Evaluation

**Evaluation Date:** October 29, 2025
**Model:** Giblet Multimodal Autoencoder (1.98B parameters, 52 layers)

---

## Quick Summary

After comprehensive testing with the real giblet autoencoder model, **torchview** is the recommended library for creating network architecture diagrams.

### Top Recommendation: torchview

- ✓ Handles 2B parameter models
- ✓ Supports horizontal orientation (essential for parallel branches)
- ✓ Shows parallel encoder/decoder pathways clearly
- ✓ Publication-quality PDF/SVG export
- ✓ Actively maintained (Oct 2024 release)
- ✓ Excellent documentation

---

## Files in This Directory

### Documentation
- **VISUALIZATION_LIBRARY_COMPARISON.md** - Comprehensive evaluation report with detailed analysis
- **README.md** (this file) - Quick reference guide
- **library_comparison.csv** - Comparison table in CSV format

### Scripts
- **test_visualization_libraries.py** - Automated testing script that evaluates all 5 libraries
- **create_architecture_diagram.py** - Quick script to generate diagrams using recommended library

### Generated Outputs (in output/)
- **torchview_test.png** - Vertical architecture diagram
- **torchview_test_horizontal.png** - Horizontal architecture diagram (RECOMMENDED)
- **torchviz_test.png** - Computational graph
- **torchviz_test_svg.svg** - Computational graph (vector)
- **custom_matplotlib_test.pdf** - Custom matplotlib visualization
- **custom_matplotlib_test.png** - Custom matplotlib visualization
- **evaluation_results.json** - Machine-readable test results

---

## Quick Start

### 1. Install the Recommended Library

```bash
# Via pip
pip install torchview

# Or via conda (recommended)
conda install -c conda-forge torchview
```

### 2. Generate Architecture Diagram

#### Using the Provided Script (Easiest)

```bash
# Create horizontal PDF (recommended)
python evaluation/create_architecture_diagram.py

# Create high-res PNG for presentations
python evaluation/create_architecture_diagram.py --format png

# Create vertical layout
python evaluation/create_architecture_diagram.py --vertical
```

#### Using Python Code Directly

```python
from torchview import draw_graph
from giblet.models import create_autoencoder
import torch

# Create model
model = create_autoencoder()

# Create dummy inputs
video = torch.randn(1, 3, 90, 160)
audio = torch.randn(1, 2048)
text = torch.randn(1, 1024)

# Generate horizontal visualization (RECOMMENDED)
model_graph = draw_graph(
    model,
    input_data=[video, audio, text],
    expand_nested=True,
    depth=3,
    device='cpu',
    graph_dir='LR'  # Horizontal (Left-to-Right)
)

# Save as PDF
model_graph.visual_graph.render('architecture', format='pdf', cleanup=True)
```

---

## Library Comparison Summary

| Library | Works? | Horizontal? | Large Models? | Maintenance | Recommendation |
|---------|--------|-------------|---------------|-------------|----------------|
| **torchview** | ✓ Yes | ✓ Yes | ✓ Yes | Very Active | **STRONGLY RECOMMENDED** |
| torchviz | ✓ Yes | ✗ No | ✓ Yes | Active | Good for computational graphs |
| custom_matplotlib | ✓ Yes | ✗ No | ✓ Yes | N/A | Good for custom styling |
| visualtorch | ✗ Failed | ? | ✗ No | Active | Not ready for complex models |
| hiddenlayer | ✗ Failed | ? | ✗ No | Inactive | Outdated (2018) |

---

## Why torchview?

1. **Parallel Branch Visualization** - Clearly shows the encoder's parallel video/audio/text pathways and decoder splits
2. **Horizontal Layout** - Essential for wide architectures with parallel processing
3. **Large Model Support** - Successfully handled 1.98B parameter model
4. **Publication Quality** - PDF and SVG export for papers
5. **Active Development** - Latest release Oct 29, 2024
6. **Modern PyTorch** - Supports complex multi-input models

---

## Use Cases

### Architecture Diagrams (Papers/Presentations)
**Use:** torchview with horizontal layout
```bash
python evaluation/create_architecture_diagram.py --format pdf
```

### Debugging/Development
**Use:** torchviz for computational graphs
```python
from torchviz import make_dot
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render('debug_graph', format='png')
```

### Custom Styling Requirements
**Use:** Custom matplotlib (already in codebase)
```python
from giblet.utils.visualization import create_network_diagram
create_network_diagram(model, 'custom.pdf')
```

---

## Key Issues Identified

### Current Implementation Problems
The existing custom matplotlib visualization (`giblet/utils/visualization.py`) has these limitations:

1. **Vertical orientation only** - Cannot show horizontal flow
2. **No parallel branch visualization** - Lists layers sequentially
3. **Doesn't clearly show architecture structure** - Looks like a simple list

### Why Parallel Branches Matter
The giblet autoencoder has critical parallel pathways:
- **Layer 2:** Video (2A), Audio (2B), Text (2C) encoders process in parallel
- **Layer 10:** Decoder splits into Video (10A), Audio (10B), Text (10C) reconstructions

These parallel structures are essential to understanding the architecture and need to be visualized horizontally.

---

## Testing Methodology

All libraries were tested with:
- **Real model:** Giblet autoencoder (not a toy model)
- **Full parameter count:** 1.98B parameters across 52 layers
- **Complex inputs:** Multi-input signature (video, audio, text)
- **Real use cases:** PDF export, horizontal layout, large model handling

See `test_visualization_libraries.py` for the complete testing framework.

---

## Next Steps

1. **Install torchview:**
   ```bash
   pip install torchview
   ```

2. **Generate diagrams for your paper:**
   ```bash
   python evaluation/create_architecture_diagram.py --format pdf
   ```

3. **Consider keeping custom matplotlib** for supplementary material showing parameter distributions

4. **Update any documentation** to use the new visualization approach

---

## Additional Resources

- **Full evaluation report:** `VISUALIZATION_LIBRARY_COMPARISON.md`
- **torchview documentation:** https://mert-kurttutan.github.io/torchview/
- **torchview GitHub:** https://github.com/mert-kurttutan/torchview
- **Generated samples:** `output/` directory

---

## Questions?

For questions about:
- **The evaluation:** See `VISUALIZATION_LIBRARY_COMPARISON.md`
- **Using torchview:** See `create_architecture_diagram.py`
- **Test results:** See `evaluation_results.json`
- **Comparison data:** See `library_comparison.csv`

---

**Evaluation completed:** October 29, 2025
**Recommendation:** Use torchview for architecture diagrams
**Status:** Ready for production use

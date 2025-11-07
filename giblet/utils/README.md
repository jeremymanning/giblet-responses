# giblet/utils - Utility Functions

This module provides utility functions for visualization and analysis.

## Components

### visualization.py

Utilities for visualizing model architectures, training progress, and results.

**Key Functions:**

```python
from giblet.utils.visualization import (
    plot_training_curves,
    visualize_feature_maps,
    plot_reconstruction_comparison
)

# Plot training/validation loss curves
plot_training_curves(
    train_losses=[0.5, 0.4, 0.3, ...],
    val_losses=[0.6, 0.5, 0.4, ...],
    save_path='training_curves.png'
)

# Visualize intermediate feature maps
visualize_feature_maps(
    features=encoder_output,
    save_path='features.png'
)

# Compare original vs reconstructed stimuli
plot_reconstruction_comparison(
    original=original_video,
    reconstructed=reconstructed_video,
    save_path='comparison.png'
)
```

---

### plotneuralnet.py

Integration with PlotNeuralNet for generating publication-quality architecture diagrams.

**Usage:**

See [examples/visualize_autoencoder_plotneuralnet.py](../../examples/visualize_autoencoder_plotneuralnet.py) for usage examples.

```python
from giblet.utils.plotneuralnet import generate_architecture_diagram

# Generate architecture diagram
generate_architecture_diagram(
    model=autoencoder,
    save_path='architecture.pdf'
)
```

**Dependencies:**
- PlotNeuralNet: https://github.com/HarisIqbal88/PlotNeuralNet
- LaTeX (for PDF generation)

---

## Related Resources

For visualization examples, see:
- [examples/visualize_autoencoder_plotneuralnet.py](../../examples/visualize_autoencoder_plotneuralnet.py)
- [examples/visualize_autoencoder_torchview.py](../../examples/visualize_autoencoder_torchview.py)
- [examples/create_network_diagram.py](../../examples/create_network_diagram.py)

For questions or issues, see the main project [README.md](../../README.md).

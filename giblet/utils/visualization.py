"""
Network visualization utilities for the giblet autoencoder.

Creates 3D layered diagrams showing network architecture with:
- Logarithmic sizing to handle large parameter counts
- Dimension labels for each layer
- Color-coded modality-specific paths
- Export to PDF and PNG formats
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np


def _get_layer_info(model: nn.Module) -> List[Dict]:
    """
    Extract layer information from a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to analyze

    Returns
    -------
    layers : list of dict
        List of dictionaries containing layer information:
        - name: Layer name
        - type: Layer type (Linear, Conv2d, etc.)
        - params: Number of parameters
        - input_shape: Input dimensions (if available)
        - output_shape: Output dimensions (if available)
    """
    layers = []

    for name, module in model.named_modules():
        # Skip container modules and the model itself
        if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
            continue

        # Count parameters
        params = sum(p.numel() for p in module.parameters(recurse=False))

        if params > 0:  # Only include layers with parameters
            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'params': params,
                'module': module
            }

            # Try to get input/output dimensions for common layer types
            if isinstance(module, nn.Linear):
                layer_info['input_dim'] = module.in_features
                layer_info['output_dim'] = module.out_features
            elif isinstance(module, nn.Conv2d):
                layer_info['input_channels'] = module.in_channels
                layer_info['output_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
            elif isinstance(module, nn.Conv1d):
                layer_info['input_channels'] = module.in_channels
                layer_info['output_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
            elif isinstance(module, nn.BatchNorm1d):
                layer_info['num_features'] = module.num_features
            elif isinstance(module, nn.BatchNorm2d):
                layer_info['num_features'] = module.num_features

            layers.append(layer_info)

    return layers


def _calculate_layer_size(params: int, sizing_mode: str = 'logarithmic',
                         min_size: float = 20, max_size: float = 200) -> float:
    """
    Calculate visual size for a layer based on parameter count.

    Parameters
    ----------
    params : int
        Number of parameters in the layer
    sizing_mode : str
        'logarithmic' or 'linear'
    min_size : float
        Minimum visual size
    max_size : float
        Maximum visual size

    Returns
    -------
    size : float
        Visual size for the layer
    """
    if params == 0:
        return min_size

    if sizing_mode == 'logarithmic':
        # Use log scale for better visualization of large parameter differences
        log_params = np.log10(params + 1)
        # Normalize to [0, 1] range (assuming max ~100M params = 8 on log scale)
        normalized = min(log_params / 8.0, 1.0)
        return min_size + (max_size - min_size) * normalized
    else:  # linear
        # Linear scaling (less useful for large networks)
        return min_size + (max_size - min_size) * min(params / 1e7, 1.0)


def _get_layer_color(layer_name: str, layer_type: str) -> Tuple[float, float, float, float]:
    """
    Get color for a layer based on its name and type.

    Parameters
    ----------
    layer_name : str
        Name of the layer
    layer_type : str
        Type of the layer

    Returns
    -------
    color : tuple
        RGBA color tuple
    """
    # Color scheme for different components
    if 'video' in layer_name.lower():
        return (0.2, 0.4, 0.8, 0.7)  # Blue for video
    elif 'audio' in layer_name.lower():
        return (0.8, 0.4, 0.2, 0.7)  # Orange for audio
    elif 'text' in layer_name.lower():
        return (0.2, 0.8, 0.4, 0.7)  # Green for text
    elif 'bottleneck' in layer_name.lower():
        return (0.8, 0.2, 0.8, 0.7)  # Purple for bottleneck
    elif 'encoder' in layer_name.lower():
        return (0.3, 0.6, 0.9, 0.7)  # Light blue for encoder
    elif 'decoder' in layer_name.lower() or 'layer' in layer_name.lower():
        return (0.9, 0.6, 0.3, 0.7)  # Light orange for decoder
    elif 'conv' in layer_type.lower():
        return (0.5, 0.5, 0.8, 0.7)  # Purple-ish for convolutions
    elif 'linear' in layer_type.lower():
        return (0.6, 0.6, 0.6, 0.7)  # Gray for linear layers
    elif 'batchnorm' in layer_type.lower():
        return (0.8, 0.8, 0.5, 0.7)  # Yellow for batch norm
    else:
        return (0.7, 0.7, 0.7, 0.7)  # Default gray


def create_network_diagram(
    model: nn.Module,
    output_path: str,
    legend: bool = True,
    sizing_mode: str = 'logarithmic',
    show_dimension: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 24),
    dpi: int = 300
) -> Path:
    """
    Create a 3D layered network diagram for a PyTorch model.

    This function creates a visualization similar to visualkeras but works
    with PyTorch models. Layers are shown as 3D blocks with sizes proportional
    to their parameter counts (logarithmic or linear scaling).

    Parameters
    ----------
    model : nn.Module
        PyTorch model to visualize
    output_path : str or Path
        Path to save the diagram (PDF or PNG)
    legend : bool, default=True
        Whether to show a legend
    sizing_mode : str, default='logarithmic'
        How to size layers: 'logarithmic' or 'linear'
    show_dimension : bool, default=True
        Whether to show layer dimensions as text
    title : str, optional
        Title for the diagram
    figsize : tuple, default=(16, 24)
        Figure size in inches (width, height)
    dpi : int, default=300
        Resolution for output image

    Returns
    -------
    output_path : Path
        Path to the saved diagram

    Examples
    --------
    >>> from giblet.models.autoencoder import MultimodalAutoencoder
    >>> model = MultimodalAutoencoder()
    >>> create_network_diagram(
    ...     model,
    ...     'network.pdf',
    ...     legend=True,
    ...     sizing_mode='logarithmic',
    ...     show_dimension=True
    ... )

    Notes
    -----
    The visualization uses color coding:
    - Blue: Video processing layers
    - Orange: Audio processing layers
    - Green: Text processing layers
    - Purple: Bottleneck layers
    - Gray: Generic linear/batch norm layers
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract layer information
    layers = _get_layer_info(model)

    if len(layers) == 0:
        raise ValueError("No layers with parameters found in model")

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 2)
    ax.axis('off')

    # Add title
    if title is None:
        total_params = sum(layer['params'] for layer in layers)
        title = f"Network Architecture ({len(layers)} layers, {total_params:,} parameters)"
    ax.text(5, len(layers) + 1.5, title,
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Draw each layer
    y_pos = len(layers)
    max_params = max(layer['params'] for layer in layers)

    for layer in layers:
        # Calculate size
        size = _calculate_layer_size(layer['params'], sizing_mode)
        color = _get_layer_color(layer['name'], layer['type'])

        # Draw 3D-style rectangle (main face)
        rect = mpatches.FancyBboxPatch(
            (5 - size/40, y_pos - 0.4),
            size/20, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)

        # Add 3D effect (side face)
        offset = 0.1
        side = mpatches.Polygon(
            [
                (5 + size/40, y_pos - 0.4),
                (5 + size/40 + offset, y_pos - 0.4 - offset),
                (5 + size/40 + offset, y_pos + 0.4 - offset),
                (5 + size/40, y_pos + 0.4)
            ],
            facecolor=(color[0]*0.7, color[1]*0.7, color[2]*0.7, color[3]),
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(side)

        # Add 3D effect (top face)
        top = mpatches.Polygon(
            [
                (5 - size/40, y_pos + 0.4),
                (5 - size/40 + offset, y_pos + 0.4 - offset),
                (5 + size/40 + offset, y_pos + 0.4 - offset),
                (5 + size/40, y_pos + 0.4)
            ],
            facecolor=(color[0]*0.85, color[1]*0.85, color[2]*0.85, color[3]),
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(top)

        # Add dimension text
        if show_dimension:
            # Layer name and type
            name_text = layer['name'].replace('encoder.', '').replace('decoder.', '')
            if len(name_text) > 30:
                name_text = name_text[:27] + '...'

            ax.text(5 + size/40 + offset + 0.2, y_pos,
                   f"{name_text}",
                   va='center', ha='left', fontsize=8, fontweight='bold')

            # Layer details
            details = f"{layer['type']}, {layer['params']:,} params"
            if 'input_dim' in layer and 'output_dim' in layer:
                details += f"\n{layer['input_dim']} → {layer['output_dim']}"
            elif 'input_channels' in layer and 'output_channels' in layer:
                details += f"\n{layer['input_channels']} → {layer['output_channels']} channels"
            elif 'num_features' in layer:
                details += f"\n{layer['num_features']} features"

            ax.text(5 + size/40 + offset + 0.2, y_pos - 0.2,
                   details,
                   va='top', ha='left', fontsize=7, style='italic',
                   color='#333333')

        y_pos -= 1

    # Add legend
    if legend:
        legend_elements = [
            mpatches.Patch(facecolor=(0.2, 0.4, 0.8, 0.7),
                          edgecolor='black', label='Video Encoder'),
            mpatches.Patch(facecolor=(0.8, 0.4, 0.2, 0.7),
                          edgecolor='black', label='Audio Encoder'),
            mpatches.Patch(facecolor=(0.2, 0.8, 0.4, 0.7),
                          edgecolor='black', label='Text Encoder'),
            mpatches.Patch(facecolor=(0.8, 0.2, 0.8, 0.7),
                          edgecolor='black', label='Bottleneck'),
            mpatches.Patch(facecolor=(0.9, 0.6, 0.3, 0.7),
                          edgecolor='black', label='Decoder'),
            mpatches.Patch(facecolor=(0.6, 0.6, 0.6, 0.7),
                          edgecolor='black', label='Generic Layer')
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 fontsize=10, framealpha=0.9)

    # Add sizing note
    if sizing_mode == 'logarithmic':
        ax.text(5, 0.5,
               f"Layer sizes scaled logarithmically by parameter count\n"
               f"Largest layer: {max_params:,} parameters",
               ha='center', va='bottom', fontsize=9, style='italic',
               color='#666666')

    # Save figure
    plt.tight_layout()

    if output_path.suffix.lower() == '.pdf':
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    plt.close(fig)

    return output_path


def create_model_summary(model: nn.Module, output_path: Optional[str] = None) -> str:
    """
    Create a text summary of the model architecture.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to summarize
    output_path : str or Path, optional
        Path to save the summary as a text file

    Returns
    -------
    summary : str
        Text summary of the model

    Examples
    --------
    >>> from giblet.models.autoencoder import MultimodalAutoencoder
    >>> model = MultimodalAutoencoder()
    >>> summary = create_model_summary(model, 'model_summary.txt')
    >>> print(summary)
    """
    layers = _get_layer_info(model)
    total_params = sum(layer['params'] for layer in layers)

    summary_lines = [
        "=" * 80,
        f"Model Architecture Summary",
        "=" * 80,
        f"Total layers: {len(layers)}",
        f"Total parameters: {total_params:,}",
        "=" * 80,
        "",
        f"{'Layer Name':<40} {'Type':<20} {'Parameters':>15}",
        "-" * 80
    ]

    for layer in layers:
        name = layer['name']
        if len(name) > 38:
            name = name[:35] + '...'
        summary_lines.append(
            f"{name:<40} {layer['type']:<20} {layer['params']:>15,}"
        )

    summary_lines.extend([
        "=" * 80,
        f"Total parameters: {total_params:,}",
        "=" * 80
    ])

    summary = '\n'.join(summary_lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary)

    return summary

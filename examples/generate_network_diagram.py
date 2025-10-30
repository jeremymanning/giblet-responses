#!/usr/bin/env python3
"""
Generate network architecture diagram for the paper.

This script creates a publication-quality visualization of the multimodal
autoencoder architecture using PlotNeuralNet.

Usage:
    python examples/generate_network_diagram.py

Outputs:
    - paper/figs/source/network.tex (LaTeX source)
    - paper/figs/source/network.pdf (rendered diagram)

References:
    Issue #18: PlotNeuralNet visualization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from giblet.utils.plotneuralnet import generate_network_diagram


def main():
    """Generate the network architecture diagram."""

    # Output path
    output_path = project_root / "paper" / "figs" / "source" / "network.pdf"

    print("=" * 70)
    print("Generating Multimodal Autoencoder Architecture Diagram")
    print("=" * 70)
    print()
    print(f"Project root: {project_root}")
    print(f"Output path: {output_path}")
    print()

    # Architecture parameters (matching the actual model)
    architecture_params = {
        'video_height': 90,
        'video_width': 160,
        'audio_mels': 2048,
        'text_dim': 1024,
        'n_voxels': 85810,
        'bottleneck_dim': 8000,
        'video_features': 1024,
        'audio_features': 256,
        'text_features': 256
    }

    print("Architecture parameters:")
    for key, value in architecture_params.items():
        print(f"  {key}: {value}")
    print()

    # Generate diagram
    result = generate_network_diagram(
        output_pdf_path=str(output_path),
        architecture_params=architecture_params,
        keep_tex=True
    )

    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Success: {result['success']}")
    print(f"LaTeX file: {result['tex_path']}")
    print(f"PDF file: {result['pdf_path']}")
    print()

    if result['success']:
        print("Network diagram generated successfully!")
        print()
        print("The diagram shows:")
        print("  - Encoder (Layers 1-6) on the left")
        print("  - Parallel branches at Layer 2 (video/audio/text)")
        print("  - Bottleneck (Layer 6) in the middle (smallest layer)")
        print("  - Decoder (Layers 7-11) on the right")
        print("  - Parallel branches at Layer 10 (video/audio/text)")
        print()
        print(f"View the PDF at: {result['pdf_path']}")
    else:
        print("PDF generation failed.")
        print(f"LaTeX source available at: {result['tex_path']}")
        print()
        print("You can compile manually with:")
        print(f"  cd {output_path.parent}")
        print(f"  pdflatex network.tex")

    print("=" * 70)

    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())

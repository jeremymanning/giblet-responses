#!/usr/bin/env python3
"""
Quick script to generate publication-quality architecture diagrams for giblet autoencoder.

This script uses the recommended torchview library to create horizontal architecture
diagrams that clearly show parallel branches (encoder pathways and decoder splits).

Usage:
    python create_architecture_diagram.py [--output OUTPUT] [--format FORMAT]

Options:
    --output OUTPUT    Output filename (default: giblet_architecture)
    --format FORMAT    Output format: pdf, png, svg (default: pdf)
    --horizontal      Use horizontal (left-to-right) layout (default)
    --vertical        Use vertical (top-to-bottom) layout
    --depth DEPTH     Visualization depth (default: 3)

Examples:
    # Create PDF with horizontal layout (recommended)
    python create_architecture_diagram.py

    # Create high-res PNG for presentation
    python create_architecture_diagram.py --format png --output presentation_fig

    # Create SVG for editing
    python create_architecture_diagram.py --format svg
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from giblet.models import create_autoencoder


def create_diagram_torchview(
    model,
    output_path: str,
    format: str = "pdf",
    horizontal: bool = True,
    depth: int = 3,
    expand_nested: bool = True,
):
    """
    Create architecture diagram using torchview (recommended).

    Parameters
    ----------
    model : nn.Module
        PyTorch model to visualize
    output_path : str
        Output file path (without extension)
    format : str
        Output format: 'pdf', 'png', or 'svg'
    horizontal : bool
        If True, use horizontal (LR) layout; if False, use vertical (TB)
    depth : int
        Depth of visualization (higher = more detail)
    expand_nested : bool
        Whether to expand nested modules
    """
    try:
        from torchview import draw_graph
    except ImportError:
        print("ERROR: torchview not installed. Install with:")
        print("  pip install torchview")
        print("  or")
        print("  conda install -c conda-forge torchview")
        sys.exit(1)

    print(f"\nGenerating architecture diagram with torchview...")
    print(f"  Layout: {'Horizontal (LR)' if horizontal else 'Vertical (TB)'}")
    print(f"  Format: {format.upper()}")
    print(f"  Depth: {depth}")

    # Create dummy inputs
    video = torch.randn(1, 3, 90, 160)
    audio = torch.randn(1, 2048)
    text = torch.randn(1, 1024)

    # Generate graph
    graph_dir = "LR" if horizontal else "TB"

    model_graph = draw_graph(
        model,
        input_data=[video, audio, text],
        expand_nested=expand_nested,
        depth=depth,
        device="cpu",
        graph_dir=graph_dir,
    )

    # Render
    output_file = model_graph.visual_graph.render(
        output_path, format=format, cleanup=True
    )

    print(f"✓ Diagram saved to: {output_file}")
    return output_file


def create_diagram_custom(model, output_path: str, format: str = "pdf"):
    """
    Create architecture diagram using custom matplotlib (alternative).

    Parameters
    ----------
    model : nn.Module
        PyTorch model to visualize
    output_path : str
        Output file path (with extension)
    format : str
        Output format: 'pdf' or 'png'
    """
    from giblet.utils.visualization import create_network_diagram

    print(f"\nGenerating architecture diagram with custom matplotlib...")
    print(f"  Format: {format.upper()}")
    print(f"  Note: Vertical layout only (no parallel branch visualization)")

    # Add extension if not present
    if not output_path.endswith(f".{format}"):
        output_path = f"{output_path}.{format}"

    create_network_diagram(
        model,
        output_path=output_path,
        legend=True,
        sizing_mode="logarithmic",
        show_dimension=True,
        title="Giblet Multimodal Autoencoder",
        figsize=(16, 24),
        dpi=300,
    )

    print(f"✓ Diagram saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture diagrams for giblet autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # PDF with horizontal layout
  %(prog)s --format png                       # PNG with horizontal layout
  %(prog)s --vertical --format svg            # SVG with vertical layout
  %(prog)s --output my_diagram --format pdf   # Custom output name
  %(prog)s --method custom                    # Use custom matplotlib
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="giblet_architecture",
        help="Output filename without extension (default: giblet_architecture)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format (default: pdf)",
    )

    parser.add_argument(
        "--horizontal",
        action="store_true",
        default=True,
        help="Use horizontal (left-to-right) layout (default)",
    )

    parser.add_argument(
        "--vertical", action="store_true", help="Use vertical (top-to-bottom) layout"
    )

    parser.add_argument(
        "--depth", type=int, default=3, help="Visualization depth (default: 3)"
    )

    parser.add_argument(
        "--method",
        choices=["torchview", "custom"],
        default="torchview",
        help="Visualization method (default: torchview)",
    )

    args = parser.parse_args()

    # Handle layout flags
    if args.vertical:
        horizontal = False
    else:
        horizontal = True

    # Create model
    print("\n" + "=" * 70)
    print("Giblet Autoencoder Architecture Diagram Generator")
    print("=" * 70)

    print("\nCreating autoencoder model...")
    model = create_autoencoder()
    model.eval()

    # Get parameter count
    param_count = model.get_parameter_count()
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {param_count['total']:,}")
    print(f"  Encoder: {param_count['encoder']:,}")
    print(f"  Decoder: {param_count['decoder']:,}")

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate diagram
    if args.method == "torchview":
        output_file = create_diagram_torchview(
            model,
            str(output_path),
            format=args.format,
            horizontal=horizontal,
            depth=args.depth,
        )
    else:  # custom
        if horizontal:
            print("\nWarning: Custom matplotlib only supports vertical layout")
        output_file = create_diagram_custom(model, str(output_path), format=args.format)

    print("\n" + "=" * 70)
    print("✓ Diagram generation complete!")
    print(f"  Output: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

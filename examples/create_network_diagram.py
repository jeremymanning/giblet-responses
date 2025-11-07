"""
Example script to create network architecture diagrams.

This script demonstrates how to use the visualization utilities to create
3D layered network diagrams showing the MultimodalAutoencoder architecture.

Usage:
    python examples/create_network_diagram.py [--output OUTPUT] [--sizing-mode MODE]

Examples:
    # Create diagram with default settings (logarithmic sizing)
    python examples/create_network_diagram.py

    # Create diagram with linear sizing
    python examples/create_network_diagram.py --sizing-mode linear

    # Create diagram with custom output path
    python examples/create_network_diagram.py --output my_network.pdf
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.models.autoencoder import MultimodalAutoencoder
from giblet.utils.visualization import create_model_summary, create_network_diagram


def main():
    """Main function to create network diagrams."""
    parser = argparse.ArgumentParser(
        description="Create network architecture diagram for MultimodalAutoencoder"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="papers/figs/source/network.pdf",
        help="Output path for network diagram (default: papers/figs/source/network.pdf)",
    )
    parser.add_argument(
        "--sizing-mode",
        type=str,
        choices=["logarithmic", "linear"],
        default="logarithmic",
        help="Layer sizing mode (default: logarithmic)",
    )
    parser.add_argument("--no-legend", action="store_true", help="Disable legend")
    parser.add_argument(
        "--no-dimensions", action="store_true", help="Disable dimension labels"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Multimodal fMRI Autoencoder Architecture",
        help="Diagram title",
    )
    parser.add_argument(
        "--summary", type=str, help="Optional path to save text summary"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for output image (default: 300)"
    )

    args = parser.parse_args()

    # Create model
    print("Creating MultimodalAutoencoder...")
    model = MultimodalAutoencoder()

    # Print parameter information
    param_info = model.get_parameter_count()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  Encoder parameters: {param_info['encoder']:,}")
    print(f"  Decoder parameters: {param_info['decoder']:,}")
    print()

    # Create network diagram
    print(f"Generating network diagram...")
    print(f"  Output: {args.output}")
    print(f"  Sizing mode: {args.sizing_mode}")
    print(f"  Legend: {not args.no_legend}")
    print(f"  Dimensions: {not args.no_dimensions}")
    print(f"  DPI: {args.dpi}")
    print()

    output_path = create_network_diagram(
        model=model,
        output_path=args.output,
        legend=not args.no_legend,
        sizing_mode=args.sizing_mode,
        show_dimension=not args.no_dimensions,
        title=args.title,
        dpi=args.dpi,
    )

    print(f"Network diagram saved to: {output_path}")

    # Create text summary if requested
    if args.summary:
        print(f"\nGenerating model summary...")
        summary = create_model_summary(model, args.summary)
        print(f"Model summary saved to: {args.summary}")
        print("\nSummary preview:")
        print(summary[:500] + "..." if len(summary) > 500 else summary)

    print("\nDone!")


if __name__ == "__main__":
    main()

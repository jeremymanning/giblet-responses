#!/usr/bin/env python3
"""
Create network visualization using torchview.

This is the RECOMMENDED approach for visualizing the MultimodalAutoencoder
because it:
1. Shows parallel branches (Layers 2A/B/C and 10A/B/C)
2. Supports horizontal orientation
3. Handles large models (2B params) with meta tensors
4. Produces publication-quality output
5. Works natively with PyTorch

Usage:
    python examples/visualize_autoencoder_torchview.py
"""
import torch
from torchview import draw_graph
from giblet.models import create_autoencoder
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize MultimodalAutoencoder with torchview')
    parser.add_argument('--output', type=str, default='papers/figs/source/autoencoder_architecture',
                        help='Output file path (without extension)')
    parser.add_argument('--orientation', type=str, default='LR', choices=['LR', 'TB', 'BT', 'RL'],
                        help='Graph orientation: LR (horizontal), TB (vertical), BT, RL')
    parser.add_argument('--depth', type=int, default=3,
                        help='Visualization depth (1=collapsed, 10=detailed)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                        help='Output format')
    parser.add_argument('--expand-nested', action='store_true',
                        help='Expand nested modules to show parallel branches')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("="*70)
    print("MultimodalAutoencoder Visualization with torchview")
    print("="*70)

    print("\nCreating MultimodalAutoencoder (2.0B params)...")
    model = create_autoencoder()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs using meta device (no memory consumption)
    print("\nCreating dummy inputs (meta tensors - no memory used)...")
    batch_size = 2
    video = torch.randn(batch_size, 3, 90, 160, device='meta')
    audio = torch.randn(batch_size, 2048, device='meta')
    text = torch.randn(batch_size, 1024, device='meta')

    print(f"\nGenerating visualization...")
    print(f"  Orientation: {args.orientation} ({'horizontal' if args.orientation == 'LR' else 'vertical'})")
    print(f"  Depth: {args.depth}")
    print(f"  Expand nested: {args.expand_nested}")
    print(f"  Format: {args.format}")

    # Generate visualization
    model_graph = draw_graph(
        model,
        input_data=(video, audio, text),
        device='meta',
        graph_dir=args.orientation,
        depth=args.depth,
        expand_nested=args.expand_nested,
        save_graph=True,
        filename=args.output,
        graph_name='MultimodalAutoencoder'
    )

    print(f"\n✓ Visualization saved to: {args.output}.{args.format}")

    print("\n" + "="*70)
    print("Visualization Tips:")
    print("="*70)
    print("1. Use --depth=1 for high-level overview (shows Encoder/Decoder)")
    print("2. Use --depth=3 for medium detail (shows modality branches)")
    print("3. Use --depth=10 for full detail (shows all layers)")
    print("4. Use --orientation=LR for horizontal layout (best for papers)")
    print("5. Use --expand-nested to see parallel branches clearly")
    print("="*70)

    print("\nExample commands:")
    print("  # High-level overview (horizontal)")
    print("  python examples/visualize_autoencoder_torchview.py --depth=1 --orientation=LR")
    print("\n  # Show parallel branches (horizontal)")
    print("  python examples/visualize_autoencoder_torchview.py --depth=3 --expand-nested --orientation=LR")
    print("\n  # Publication quality PDF")
    print("  python examples/visualize_autoencoder_torchview.py --format=pdf --orientation=LR")

if __name__ == '__main__':
    main()

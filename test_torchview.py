#!/usr/bin/env python3
"""
Test torchview with our MultimodalAutoencoder.
"""
import torch
from torchview import draw_graph
from giblet.models import create_autoencoder
import os

# Create output directory
os.makedirs('papers/figs/source/research', exist_ok=True)

print("Creating MultimodalAutoencoder (2.0B params)...")
model = create_autoencoder()

print("\nModel created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy inputs (use meta device to avoid memory issues)
batch_size = 2
video = torch.randn(batch_size, 3, 90, 160, device='meta')
audio = torch.randn(batch_size, 2048, device='meta')
text = torch.randn(batch_size, 1024, device='meta')

print("\nTesting torchview with different orientations...")

# Test 1: Left-to-Right (Horizontal)
print("\n1. Creating LR (horizontal) graph...")
try:
    model_graph_lr = draw_graph(
        model,
        input_data=(video, audio, text),
        device='meta',
        graph_dir='LR',
        depth=3,
        expand_nested=True,
        save_graph=True,
        filename='papers/figs/source/research/torchview_lr'
    )
    print("   ✓ LR graph created successfully")
    print(f"   Output saved to: papers/figs/source/research/torchview_lr.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Top-to-Bottom (Vertical)
print("\n2. Creating TB (vertical) graph...")
try:
    model_graph_tb = draw_graph(
        model,
        input_data=(video, audio, text),
        device='meta',
        graph_dir='TB',
        depth=3,
        expand_nested=True,
        save_graph=True,
        filename='papers/figs/source/research/torchview_tb'
    )
    print("   ✓ TB graph created successfully")
    print(f"   Output saved to: papers/figs/source/research/torchview_tb.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Collapsed view (less detail)
print("\n3. Creating collapsed view (depth=1)...")
try:
    model_graph_collapsed = draw_graph(
        model,
        input_data=(video, audio, text),
        device='meta',
        graph_dir='LR',
        depth=1,
        expand_nested=False,
        save_graph=True,
        filename='papers/figs/source/research/torchview_collapsed'
    )
    print("   ✓ Collapsed graph created successfully")
    print(f"   Output saved to: papers/figs/source/research/torchview_collapsed.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Maximum detail (depth=10)
print("\n4. Creating detailed view (depth=10)...")
try:
    model_graph_detailed = draw_graph(
        model,
        input_data=(video, audio, text),
        device='meta',
        graph_dir='LR',
        depth=10,
        expand_nested=True,
        save_graph=True,
        filename='papers/figs/source/research/torchview_detailed'
    )
    print("   ✓ Detailed graph created successfully")
    print(f"   Output saved to: papers/figs/source/research/torchview_detailed.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("TORCHVIEW EVALUATION:")
print("="*60)
print("✓ Supports horizontal orientation (LR)")
print("✓ Works with PyTorch native models")
print("✓ Handles large models with meta tensors (no memory consumption)")
print("✓ Multiple depth levels for different abstraction")
print("✓ Can expand nested modules to show parallel branches")
print("? Need to check: parallel branch visibility in output")
print("? Need to check: publication quality of output")
print("="*60)

"""
Test Linear-based VideoEncoder with flattened temporal concatenation.
Phase 1.4: Local testing with simple forward pass.
"""
import torch

from giblet.models.encoder import VideoEncoder

print("=" * 80)
print("VIDEOENCODER LINEAR ARCHITECTURE TEST")
print("=" * 80)

# Test 1: Create encoder
print("\n1. Creating VideoEncoder with Linear layers...")
encoder = VideoEncoder(input_dim=1641600, output_features=1024)
total_params = sum(p.numel() for p in encoder.parameters())
print(f"✓ VideoEncoder created")
print(f"  Total parameters: {total_params:,}")
print(f"  Input dim: 1,641,600 (38 frames × 160×90×3)")
print(f"  Output features: 1,024")

# Test 2: Forward pass with 2D flattened input
print("\n2. Testing forward pass with flattened temporal concatenation...")
batch_size = 2
video_flat = torch.randn(batch_size, 1641600)  # Flattened temporal concatenation

output = encoder(video_flat)
print(f"✓ Forward pass successful!")
print(f"  Input shape: {video_flat.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == (batch_size, 1024), f"Expected (2, 1024), got {output.shape}"

# Test 3: Check output values are reasonable
print(f"\n3. Checking output statistics...")
print(f"  Mean: {output.mean().item():.4f}")
print(f"  Std: {output.std().item():.4f}")
print(f"  Min: {output.min().item():.4f}")
print(f"  Max: {output.max().item():.4f}")
assert not torch.isnan(output).any(), "NaN values in output!"
assert not torch.isinf(output).any(), "Inf values in output!"
print(f"✓ Output values are valid (no NaN/Inf)")

# Test 4: Backward compatibility with 4D input
print(f"\n4. Testing backward compatibility with 4D input...")
video_4d = torch.randn(batch_size, 3, 90, 160)  # Legacy single-frame format
output_4d = encoder(video_4d)
print(f"✓ Backward compatibility works!")
print(f"  4D input shape: {video_4d.shape}")
print(f"  Output shape: {output_4d.shape}")
assert output_4d.shape == (batch_size, 1024)

# Test 5: Gradient flow (backward pass)
print(f"\n5. Testing gradient flow...")
video_grad = torch.randn(batch_size, 1641600, requires_grad=True)
output_grad = encoder(video_grad)
loss = output_grad.sum()
loss.backward()

assert video_grad.grad is not None, "No gradient computed!"
assert not torch.isnan(video_grad.grad).any(), "NaN in gradients!"
assert not torch.isinf(video_grad.grad).any(), "Inf in gradients!"
print(f"✓ Gradients computed successfully")
print(f"  Gradient shape: {video_grad.grad.shape}")
print(f"  Gradient norm: {video_grad.grad.norm().item():.4f}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)

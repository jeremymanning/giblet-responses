"""
Test Linear VideoEncoder with REAL Sherlock dataset.
NO MOCKS - uses actual stimuli_Sherlock.m4v data.
"""
import torch
from giblet.data.dataset import MultimodalDataset
from giblet.models.autoencoder import MultimodalAutoencoder

print("="*80)
print("REAL DATASET INTEGRATION TEST")
print("="*80)

# Test 1: Load REAL Sherlock dataset
print("\n1. Loading REAL Sherlock dataset (subject 1, 10 TRs)...")
dataset = MultimodalDataset(
    'data',
    subjects=[1],
    max_trs=10,
    use_encodec=True,
    encodec_bandwidth=3.0
)

print(f"✓ Dataset loaded: {len(dataset)} samples")
print(f"  Feature dims:")
print(f"    Video: {dataset.feature_dims['video']}")
print(f"    Audio: {dataset.feature_dims['audio']}")
print(f"    Text: {dataset.feature_dims['text']}")
print(f"    fMRI: {dataset.feature_dims['fmri']}")

# Test 2: Create model with correct dimensions
print("\n2. Creating MultimodalAutoencoder...")
model = MultimodalAutoencoder(
    video_height=90,
    video_width=160,
    audio_mels=896,  # EnCodec: 8 codebooks × 112 frames (flattened)
    text_dim=3072,  # 3 × 1024 (temporal concatenation)
    n_voxels=85810,
    bottleneck_dim=2048,
    use_encodec=True,
    audio_frames_per_tr=112
)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created")
print(f"  Total parameters: {total_params:,}")

# Test 3: Forward pass with REAL batch
print("\n3. Testing forward pass with REAL Sherlock data...")
batch = dataset.get_batch([0, 1])

print(f"  Batch shapes:")
print(f"    Video: {batch['video'].shape}")
print(f"    Audio: {batch['audio'].shape}")
print(f"    Text: {batch['text'].shape}")
print(f"    fMRI: {batch['fmri'].shape}")

model.eval()
with torch.no_grad():
    outputs = model(
        batch['video'],
        batch['audio'],
        batch['text'],
        fmri_target=batch['fmri']
    )

print(f"\n✓ Forward pass SUCCESSFUL!")
print(f"  Outputs:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"    {key}: {value.shape}")
    else:
        print(f"    {key}: {value}")

# Test 4: Verify output shapes
print(f"\n4. Verifying output shapes...")
assert outputs['bottleneck'].shape == (2, 2048), f"Bottleneck shape mismatch"
assert outputs['predicted_fmri'].shape == (2, 85810), f"fMRI pred shape mismatch"
assert outputs['video_recon'].shape == (2, 1641600), f"Video recon shape mismatch"
print(f"✓ All output shapes correct!")

# Test 5: Verify no NaN/Inf
print(f"\n5. Checking for NaN/Inf values...")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        assert not torch.isnan(value).any(), f"NaN in {key}!"
        assert not torch.isinf(value).any(), f"Inf in {key}!"
print(f"✓ All outputs valid (no NaN/Inf)")

print("\n" + "="*80)
print("✓ REAL DATASET TEST PASSED!")
print("="*80)
print("\nReady for cluster deployment and training test.")

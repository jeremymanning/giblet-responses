"""Quick evaluation diagnostic to identify where the script is getting stuck."""
import sys
from pathlib import Path

print("Step 1: Imports...")
import torch
print("  ✓ PyTorch imported")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from giblet.training.config import TrainingConfig
print("  ✓ Config imported")

# Test 1: Load config
print("\nStep 2: Loading config...")
config = TrainingConfig.from_yaml("configs/training/tensor02_test_50epoch_config.yaml")
print(f"  ✓ Config loaded: {config.model.bottleneck_dim} bottleneck dims")

# Test 2: Load checkpoint metadata only
print("\nStep 3: Loading checkpoint metadata...")
checkpoint = torch.load("checkpoints_local/tensor02_fixed_lr/best_checkpoint.pt",
                       map_location='cpu', weights_only=False)
print(f"  ✓ Checkpoint loaded: epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.2f}")

# Test 3: Import model
print("\nStep 4: Importing model...")
from giblet.models.autoencoder import MultimodalAutoEncoder
print("  ✓ Model class imported")

# Test 4: Create model
print("\nStep 5: Creating model instance...")
model = MultimodalAutoEncoder(
    video_height=config.model.video_height,
    video_width=config.model.video_width,
    audio_mels=config.model.audio_mels,
    text_dim=config.model.text_dim,
    n_voxels=config.model.n_voxels,
    bottleneck_dim=config.model.bottleneck_dim,
    video_features=config.model.video_features,
    audio_features=config.model.audio_features,
    text_features=config.model.text_features,
    decoder_hidden_dim=config.model.decoder_hidden_dim,
    decoder_dropout=config.model.decoder_dropout,
    use_encodec=config.model.use_encodec,
    audio_frames_per_tr=config.model.audio_frames_per_tr,
)
print("  ✓ Model created")

# Test 5: Load weights
print("\nStep 6: Loading model weights...")
model.load_state_dict(checkpoint['model_state_dict'])
print("  ✓ Weights loaded")
model.eval()
print("  ✓ Model set to eval mode")

# Test 6: Import dataset
print("\nStep 7: Importing dataset class...")
from giblet.data.dataset import SherlocfMRIDataset
print("  ✓ Dataset class imported")

# Test 7: Try to create dataset (THIS is likely where it hangs)
print("\nStep 8: Creating test dataset (this may take a while)...")
try:
    dataset = SherlocfMRIDataset(
        data_dir=config.data.data_dir,
        subjects=config.data.subjects,
        split='val',
        apply_hrf=config.data.apply_hrf,
        mode=config.data.mode,
        frame_skip=config.data.frame_skip,
        fps=config.data.fps,
        tr=config.data.tr,
    )
    print(f"  ✓ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"  ✗ Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ ALL TESTS PASSED! Evaluation script should work.")
print("  The issue must be elsewhere or with matplotlib/visualization.")

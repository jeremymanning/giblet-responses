# Encoder Architecture Update - Layer 7 Bottleneck

**Date**: 2025-10-31
**Issue**: Fix encoder to have Layer 7 as bottleneck in 13-layer architecture

## Changes Made

### Previous Structure (Incorrect)
```
Encoder: 1536 (pooled) → 4096 → 8000 (bottleneck_dim)
```

**Problem**: Only 6 layers in encoder, bottleneck at Layer 6 (8000 dims) wasn't the smallest layer.

### New Structure (Correct)
```
Layer 1:   Input (video + audio + text)
Layer 2A:  Video encoder → 1024 dims
Layer 2B:  Audio encoder → 256 dims
Layer 2C:  Text encoder → 256 dims
Layer 3:   Pooled features → 1536 dims (1024 + 256 + 256)
Layer 4:   Feature convolution → 1536 dims
Layer 5:   First expansion → 4096 dims (1536 → 4096)
Layer 6:   Second expansion → 8000 dims (4096 → 8000)
Layer 7:   BOTTLENECK → 2048 dims (8000 → 2048) ← SMALLEST LAYER
```

**Solution**: Now 7 layers in encoder, with Layer 7 as the bottleneck (2048 dims, smallest in expansion path).

## File Modified

**File**: `/Users/jmanning/giblet-responses/giblet/models/encoder.py`

### Key Changes

1. **Updated `__init__` method**:
   - Changed from single `to_bottleneck` Sequential to three separate layers:
     - `self.layer5`: Linear(1536, 4096) + BN + ReLU + Dropout
     - `self.layer6`: Linear(4096, 8000) + BN + ReLU + Dropout
     - `self.layer7_bottleneck`: Linear(8000, 2048) + BN (NO ReLU)

2. **Updated default `bottleneck_dim`**: 8000 → 2048

3. **Updated `forward()` method**:
   - Now passes through layer5 → layer6 → layer7_bottleneck sequentially
   - Bottleneck output is 2048 dimensions (not 8000)

4. **Updated docstrings**: Reflect 7-layer encoder structure

5. **Removed ReLU from Layer 7**: Allows negative values in latent space

6. **Updated `get_parameter_count()`**: Returns counts for layer5, layer6, layer7_bottleneck separately

## Architecture Rationale

### Why expand then compress?
- **Layer 5-6**: Expand from 1536 → 4096 → 8000 to increase representational capacity
- **Layer 7**: Compress to 2048 as bottleneck, forcing information compression
- This creates an information bottleneck that learns most salient features

### Why no ReLU at bottleneck?
- Allows negative values in latent space
- More expressive representation
- Standard practice in autoencoders/VAEs

### Parameter counts (with bottleneck_dim=2048)
```
Video encoder:           16,119,040 params
Audio encoder:            8,519,424 params
Text encoder:               657,664 params
Feature conv:             2,363,904 params
Layer 5 (1536→4096):      6,303,744 params
Layer 6 (4096→8000):     32,792,000 params
Layer 7 (8000→2048):     16,390,144 params
Bottleneck→voxels:    1,439,600,434 params
----------------------------------------
TOTAL:                1,522,746,354 params
```

## Testing

### Test 1: Architecture Structure (`test_encoder_architecture.py`)
```bash
python test_encoder_architecture.py
```

**Results**: ✓ ALL TESTS PASSED
- ✓ Default bottleneck_dim = 2048
- ✓ Layer 5: 1536 → 4096
- ✓ Layer 6: 4096 → 8000
- ✓ Layer 7: 8000 → 2048 (bottleneck, smallest in expansion path)
- ✓ Layer 7 has no ReLU activation
- ✓ Forward pass produces correct output shapes

### Test 2: 13-Layer Verification (`verify_13_layer_architecture.py`)
```bash
python verify_13_layer_architecture.py
```

**Results**: ✓ VERIFIED
- ✓ Encoder: 7 layers implemented
- ✓ Decoder: 6 layers (to be implemented)
- ✓ Total: 13 layers

## 13-Layer Autoencoder Design

### ENCODER (7 layers) - IMPLEMENTED ✓
```
Layer 1:   Input (video + audio + text)
Layer 2A:  Video encoder (Conv2D)
Layer 2B:  Audio encoder (Conv1D)
Layer 2C:  Text encoder (Linear)
Layer 3:   Pooled multimodal features (1536 dims)
Layer 4:   Feature space convolution (1536 dims)
Layer 5:   First expansion (1536 → 4096)
Layer 6:   Second expansion (4096 → 8000)
Layer 7:   BOTTLENECK (8000 → 2048) ← SMALLEST LAYER
```

### DECODER (6 layers) - TO BE IMPLEMENTED
```
Layer 8:   Expansion from bottleneck (2048 → 8000)
Layer 9:   Expansion (8000 → 4096)
Layer 10:  Compression (4096 → 1536)
Layer 11A: Video decoder (Deconv2D)
Layer 11B: Audio decoder (Deconv1D)
Layer 11C: Text decoder (Linear)
Layer 12:  Unpooled features
Layer 13:  Output (reconstructed video + audio + text)
```

## Usage Example

```python
import torch
from giblet.models.encoder import MultimodalEncoder

# Create encoder (default bottleneck_dim=2048)
encoder = MultimodalEncoder()

# Prepare inputs
batch_size = 4
video = torch.randn(batch_size, 3, 90, 160)    # RGB frames
audio = torch.randn(batch_size, 2048)          # Mel spectrograms
text = torch.randn(batch_size, 1024)           # Text embeddings

# Forward pass
bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

print(f"Bottleneck shape: {bottleneck.shape}")  # (4, 2048)
print(f"Voxels shape: {voxels.shape}")          # (4, 85810)
```

## Next Steps

1. ✓ **Encoder fixed**: Layer 7 is now the bottleneck (2048 dims)
2. **Decoder implementation**: Implement symmetric 6-layer decoder
3. **Full autoencoder**: Combine encoder + decoder into single model
4. **Training**: Train on Sherlock fMRI dataset

## Notes

- **Backward compatibility**: Models using custom `bottleneck_dim` still work
- **Voxel prediction**: Optional auxiliary output (not part of main 13-layer path)
- **Bottleneck is smallest**: 2048 < 4096 < 8000 in the expansion path
  - Note: Pooled layer (1536) comes before expansion, so it's not part of the comparison

# Audio Reconstruction Debugging Session
Date: 2025-10-29

## Problem
Audio reconstruction producing static/garbled output despite good mel correlation.

## Attempts

### Attempt 1: 22.05kHz, 128 mels
- Result: Static, correlation: ~0.91

### Attempt 2: 44.1kHz, 256 mels  
- Result: Still static, correlation: ~0.97

### Attempt 3: 22.05kHz, 2048 mels
- Result: Correlation: 0.97, testing audio quality
- Warning: Empty mel filters (too many mels for sample rate)

## Key Insight
High mel correlation doesn't guarantee good audio quality.
Griffin-Lim is fundamentally limited - it reconstructs phase randomly.

## Next Steps
1. Listen to test_audio/reconstructed_2048mels.wav
2. If still poor: Griffin-Lim may not be suitable
3. Alternative: Use neural vocoder (WaveGlow, Parallel WaveGAN)
4. Or: Accept lower audio quality for now, focus on fMRI prediction

## Decision Point
Does audio reconstruction quality matter for the core goal (brain region simulation)?
- Encoder: Needs good audio→features (WORKING ✓)
- Decoder: Needs features→audio (PROBLEMATIC)

For "lesioning" brain regions, we primarily care about VIDEO reconstruction quality.
Audio may be secondary.

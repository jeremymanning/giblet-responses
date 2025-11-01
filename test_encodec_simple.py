#!/usr/bin/env python
"""
Simple manual test for EnCodec integration.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile

print("=" * 80)
print("EnCodec Integration Test")
print("=" * 80)

# Step 1: Check imports
print("\n1. Checking imports...")
try:
    from giblet.data.audio import AudioProcessor, ENCODEC_AVAILABLE
    print(f"   EnCodec available: {ENCODEC_AVAILABLE}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if not ENCODEC_AVAILABLE:
    print("   EnCodec not available. Install transformers package.")
    exit(1)

# Step 2: Generate test audio
print("\n2. Generating test audio...")
duration = 5.0
sample_rate = 24000
t = np.linspace(0, duration, int(sample_rate * duration))
chirp = np.sin(2 * np.pi * (100 * t + 7900 * t**2 / (2 * duration)))
chirp = chirp / np.max(np.abs(chirp)) * 0.9

temp_dir = Path(tempfile.mkdtemp())
audio_path = temp_dir / "test.wav"
sf.write(str(audio_path), chirp, sample_rate)
print(f"   Saved: {audio_path}")

# Step 3: Initialize AudioProcessor
print("\n3. Initializing AudioProcessor...")
try:
    processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)
    print("   Success!")
    print(f"   Using EnCodec: {processor.use_encodec}")
    print(f"   Bandwidth: {processor.encodec_bandwidth} kbps")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Encode audio
print("\n4. Encoding audio...")
try:
    features, metadata = processor.audio_to_features(audio_path, from_video=False)
    print(f"   Features shape: {features.shape}")
    print(f"   Features dtype: {features.dtype}")
    print(f"   Metadata columns: {list(metadata.columns)}")
    print(f"   Encoding mode: {metadata['encoding_mode'].iloc[0]}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Decode audio
print("\n5. Decoding audio...")
try:
    output_path = temp_dir / "decoded.wav"
    processor.features_to_audio(features, output_path)
    print(f"   Saved: {output_path}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Verify quality
print("\n6. Verifying quality...")
try:
    import librosa
    original, _ = librosa.load(str(audio_path), sr=24000)
    reconstructed, _ = librosa.load(str(output_path), sr=24000)

    # Trim to same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Correlation
    correlation = np.corrcoef(original, reconstructed)[0, 1]
    print(f"   Correlation: {correlation:.3f}")

    # STOI (if available)
    try:
        from pystoi import stoi
        stoi_score = stoi(original, reconstructed, 24000, extended=False)
        print(f"   STOI: {stoi_score:.3f}")
    except ImportError:
        print("   STOI: Not available (pystoi not installed)")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
import shutil
shutil.rmtree(temp_dir)

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)

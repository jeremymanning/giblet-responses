"""
Test EnCodec integration in AudioProcessor (Issue #24, Task 2.1).

Tests:
1. EnCodec encoding produces correct dimensions
2. Round-trip: audio -> codes -> audio
3. TR alignment
4. Audio quality metrics (STOI)
5. Backwards compatibility with mel spectrograms
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Audio processing
from giblet.data.audio import AudioProcessor, ENCODEC_AVAILABLE

# Audio quality metrics
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# Audio I/O
import librosa
import soundfile as sf


@pytest.fixture
def test_audio_dir():
    """Create temporary directory for test audio files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_audio(test_audio_dir):
    """Generate a sample audio file for testing (5 seconds at 24kHz)."""
    # Generate chirp signal: sweep from 100 Hz to 8000 Hz
    duration = 5.0  # seconds
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Chirp signal
    f0 = 100  # Start frequency
    f1 = 8000  # End frequency
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))

    # Add some harmonic content
    chirp += 0.3 * np.sin(4 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))

    # Normalize
    chirp = chirp / np.max(np.abs(chirp)) * 0.9

    # Save
    audio_path = test_audio_dir / "test_chirp.wav"
    sf.write(str(audio_path), chirp, sample_rate)

    return audio_path, sample_rate, duration


@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available (transformers not installed)")
class TestEnCodecIntegration:
    """Test EnCodec integration in AudioProcessor."""

    def test_encodec_initialization(self):
        """Test that EnCodec model loads successfully."""
        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0)

        assert processor.use_encodec is True
        assert hasattr(processor, 'encodec_model')
        assert hasattr(processor, 'encodec_processor')
        assert processor.encodec_sample_rate == 24000
        assert processor.encodec_bandwidth == 3.0

    def test_encodec_encoding_dimensions(self, sample_audio):
        """Test that EnCodec encoding produces correct dimensions."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)

        # Encode
        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check dimensions
        n_trs = int(np.floor(duration / 1.5))
        encodec_frame_rate = 75  # EnCodec fixed frame rate
        frames_per_tr = int(encodec_frame_rate * 1.5)

        assert features.shape[0] == n_trs, f"Expected {n_trs} TRs, got {features.shape[0]}"
        assert features.shape[1] == 1, f"Expected 1 codebook (mono), got {features.shape[1]}"
        assert features.shape[2] == frames_per_tr, f"Expected {frames_per_tr} frames/TR, got {features.shape[2]}"

        # Check dtype
        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Check metadata
        assert len(metadata) == n_trs
        assert 'encoding_mode' in metadata.columns
        assert metadata['encoding_mode'].iloc[0] == 'encodec'

        print(f"\nEnCodec encoding test passed:")
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")
        print(f"  TRs: {n_trs}")
        print(f"  Frames/TR: {frames_per_tr}")

    def test_encodec_round_trip(self, sample_audio, test_audio_dir):
        """Test round-trip: audio -> EnCodec codes -> audio."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)

        # Encode
        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Decode
        output_path = test_audio_dir / "roundtrip_encodec.wav"
        processor.features_to_audio(features, output_path)

        # Check output exists
        assert output_path.exists()

        # Load original and reconstructed
        original, sr_orig = librosa.load(str(audio_path), sr=24000)
        reconstructed, sr_recon = librosa.load(str(output_path), sr=24000)

        # Check sample rates match
        assert sr_orig == sr_recon == 24000

        # Trim to same length (may differ slightly)
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

        # Check correlation (should be high for round-trip)
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        assert correlation > 0.7, f"Low correlation: {correlation:.3f}"

        print(f"\nRound-trip test passed:")
        print(f"  Original length: {len(original)} samples")
        print(f"  Reconstructed length: {len(reconstructed)} samples")
        print(f"  Correlation: {correlation:.3f}")

        return original, reconstructed

    @pytest.mark.skipif(not STOI_AVAILABLE, reason="pystoi not available")
    def test_encodec_quality_metrics(self, sample_audio, test_audio_dir):
        """Test audio quality metrics for EnCodec round-trip."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)

        # Encode and decode
        features, _ = processor.audio_to_features(audio_path, from_video=False)
        output_path = test_audio_dir / "quality_test.wav"
        processor.features_to_audio(features, output_path)

        # Load audio
        original, sr = librosa.load(str(audio_path), sr=24000)
        reconstructed, _ = librosa.load(str(output_path), sr=24000)

        # Trim to same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

        # Compute STOI (Short-Time Objective Intelligibility)
        stoi_score = stoi(original, reconstructed, sr, extended=False)

        # Based on Batch 1 results: STOI ~ 0.74 for 3.0 kbps
        # Accept anything >= 0.6 to allow for test signal differences
        assert stoi_score >= 0.6, f"STOI too low: {stoi_score:.3f}"

        print(f"\nQuality metrics:")
        print(f"  STOI: {stoi_score:.3f}")
        print(f"  Expected: ~0.74 (based on Batch 1 results)")

        return stoi_score

    def test_encodec_tr_alignment(self, sample_audio):
        """Test that EnCodec codes are properly aligned to TRs."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)

        # Encode
        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check metadata
        for idx, row in metadata.iterrows():
            expected_start = idx * 1.5
            expected_end = (idx + 1) * 1.5

            assert np.isclose(row['start_time'], expected_start), \
                f"TR {idx}: start_time mismatch"
            assert np.isclose(row['end_time'], expected_end), \
                f"TR {idx}: end_time mismatch"

        print(f"\nTR alignment test passed:")
        print(f"  TRs: {len(metadata)}")
        print(f"  TR duration: 1.5s")

    def test_encodec_max_trs(self, sample_audio):
        """Test max_trs parameter with EnCodec."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=3.0, tr=1.5)

        # Encode with max_trs
        max_trs = 2
        features, metadata = processor.audio_to_features(
            audio_path,
            max_trs=max_trs,
            from_video=False
        )

        assert features.shape[0] == max_trs
        assert len(metadata) == max_trs

        print(f"\nmax_trs test passed:")
        print(f"  Requested: {max_trs} TRs")
        print(f"  Got: {features.shape[0]} TRs")


class TestBackwardsCompatibility:
    """Test backwards compatibility with mel spectrograms."""

    def test_mel_spectrogram_fallback(self, sample_audio):
        """Test that mel spectrogram mode still works."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=False, tr=1.5)

        # Encode
        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check dimensions
        n_trs = int(np.floor(duration / 1.5))
        assert features.shape[0] == n_trs
        assert features.shape[1] == 2048  # n_mels

        # Check dtype (float for mel spectrogram)
        assert features.dtype == np.float32

        # Check metadata
        assert metadata['encoding_mode'].iloc[0] == 'mel_spectrogram'

        print(f"\nMel spectrogram fallback test passed:")
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")

    @pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
    def test_feature_format_auto_detection(self, sample_audio, test_audio_dir):
        """Test automatic detection of feature format in features_to_audio."""
        audio_path, sample_rate, duration = sample_audio

        # Test EnCodec format (integer)
        processor_encodec = AudioProcessor(use_encodec=True, tr=1.5)
        features_encodec, _ = processor_encodec.audio_to_features(audio_path, from_video=False)

        output_encodec = test_audio_dir / "auto_detect_encodec.wav"
        processor_encodec.features_to_audio(features_encodec, output_encodec)
        assert output_encodec.exists()

        # Test mel format (float)
        processor_mel = AudioProcessor(use_encodec=False, tr=1.5)
        features_mel, _ = processor_mel.audio_to_features(audio_path, from_video=False)

        output_mel = test_audio_dir / "auto_detect_mel.wav"
        processor_mel.features_to_audio(features_mel, output_mel)
        assert output_mel.exists()

        print(f"\nAuto-detection test passed:")
        print(f"  EnCodec features dtype: {features_encodec.dtype}")
        print(f"  Mel features dtype: {features_mel.dtype}")


class TestEnCodecBandwidths:
    """Test different EnCodec bandwidth settings."""

    @pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
    @pytest.mark.parametrize("bandwidth", [1.5, 3.0, 6.0, 12.0, 24.0])
    def test_bandwidth_settings(self, sample_audio, test_audio_dir, bandwidth):
        """Test EnCodec with different bandwidth settings."""
        audio_path, sample_rate, duration = sample_audio

        processor = AudioProcessor(use_encodec=True, encodec_bandwidth=bandwidth, tr=1.5)

        # Encode
        features, _ = processor.audio_to_features(audio_path, from_video=False)

        # Decode
        output_path = test_audio_dir / f"bandwidth_{bandwidth}.wav"
        processor.features_to_audio(features, output_path)

        assert output_path.exists()

        print(f"\nBandwidth {bandwidth} kbps test passed")


if __name__ == "__main__":
    # Run tests manually
    import sys

    print("=" * 80)
    print("EnCodec Integration Tests")
    print("=" * 80)

    if not ENCODEC_AVAILABLE:
        print("ERROR: EnCodec not available (transformers not installed)")
        sys.exit(1)

    # Create test directory
    test_dir = Path(tempfile.mkdtemp())

    try:
        # Generate test audio
        print("\nGenerating test audio...")
        duration = 5.0
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        chirp = np.sin(2 * np.pi * (100 * t + 7900 * t**2 / (2 * duration)))
        chirp = chirp / np.max(np.abs(chirp)) * 0.9
        audio_path = test_dir / "test.wav"
        sf.write(str(audio_path), chirp, sample_rate)
        print(f"  Saved: {audio_path}")

        # Run tests
        test_class = TestEnCodecIntegration()

        print("\n1. Testing EnCodec initialization...")
        test_class.test_encodec_initialization()

        print("\n2. Testing encoding dimensions...")
        test_class.test_encodec_encoding_dimensions((audio_path, sample_rate, duration))

        print("\n3. Testing round-trip...")
        test_class.test_encodec_round_trip((audio_path, sample_rate, duration), test_dir)

        if STOI_AVAILABLE:
            print("\n4. Testing quality metrics...")
            test_class.test_encodec_quality_metrics((audio_path, sample_rate, duration), test_dir)
        else:
            print("\n4. Skipping quality metrics (pystoi not available)")

        print("\n5. Testing TR alignment...")
        test_class.test_encodec_tr_alignment((audio_path, sample_rate, duration))

        print("\n6. Testing max_trs...")
        test_class.test_encodec_max_trs((audio_path, sample_rate, duration))

        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)

    finally:
        # Cleanup
        shutil.rmtree(test_dir)

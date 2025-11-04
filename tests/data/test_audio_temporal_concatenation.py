"""
Test audio temporal concatenation fix for Issue #26, Task 1.2.

Tests the critical dimension consistency fix that resolves:
    RuntimeError: stack expects each tensor to be equal size,
    but got [1, 4, 106697] at entry 0 and [1, 0, 106705] at entry 1

Key requirements:
1. All TRs must have identical shape
2. Consistent codebook count across all TRs
3. Flattened output for training compatibility
4. Configurable TR length
"""

import pytest
import numpy as np
import torch

from giblet.data.audio import AudioProcessor, ENCODEC_AVAILABLE

# Audio I/O
import librosa
import soundfile as sf


@pytest.fixture
def sample_audio_5s(tmp_path):
    """Generate 5-second audio for testing."""
    duration = 5.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Complex signal with varying content
    signal = (
        np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.5 * np.sin(2 * np.pi * 880 * t) +  # Harmonic
        0.3 * np.random.randn(len(t))  # Noise
    )
    signal = signal / np.max(np.abs(signal)) * 0.9

    audio_path = tmp_path / "test_5s.wav"
    sf.write(str(audio_path), signal, sample_rate)

    return audio_path, sample_rate, duration


@pytest.fixture
def sample_audio_long(tmp_path):
    """Generate longer audio (10 seconds) for edge case testing."""
    duration = 10.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Varying frequency sweep
    f0, f1 = 200, 2000
    signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    signal = signal / np.max(np.abs(signal)) * 0.9

    audio_path = tmp_path / "test_10s.wav"
    sf.write(str(audio_path), signal, sample_rate)

    return audio_path, sample_rate, duration


@pytest.mark.unit
@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestDimensionConsistency:
    """Test that all TRs have consistent dimensions (fixes RuntimeError)."""

    def test_all_trs_same_shape(self, audio_processor, sample_audio_5s):
        """Critical test: All TRs must have identical shape."""
        audio_path, sr, duration = sample_audio_5s

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check all TRs have same shape
        n_trs = features.shape[0]
        expected_shape = features[0].shape

        for tr_idx in range(n_trs):
            tr_shape = features[tr_idx].shape
            assert tr_shape == expected_shape, (
                f"TR {tr_idx} has shape {tr_shape}, expected {expected_shape}. "
                f"This would cause: RuntimeError: stack expects each tensor to be equal size"
            )

        print(f"\n✓ All {n_trs} TRs have consistent shape: {expected_shape}")

    def test_consistent_codebook_count(self, audio_processor, sample_audio_5s):
        """Test that codebook count is consistent across all TRs."""
        audio_path, sr, duration = sample_audio_5s

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Features are flattened: (n_trs, n_codebooks * frames_per_tr)
        # Verify all metadata has same n_codebooks
        codebook_counts = metadata['n_codebooks'].unique()

        assert len(codebook_counts) == 1, (
            f"Variable codebook counts detected: {codebook_counts}. "
            f"This causes the dimension mismatch error."
        )

        n_codebooks = codebook_counts[0]
        assert n_codebooks == 8, f"Expected 8 codebooks for 3.0 kbps, got {n_codebooks}"

        print(f"\n✓ Consistent codebook count: {n_codebooks}")

    def test_flattened_output_shape(self, audio_processor, sample_audio_5s):
        """Test that output is flattened to (n_trs, n_codebooks * frames_per_tr)."""
        audio_path, sr, duration = sample_audio_5s

        # Note: Using default audio_processor with TR=1.5s and bandwidth=3.0 kbps
        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check shape
        assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"

        n_trs, flat_dim = features.shape
        expected_n_trs = int(duration / 1.5)

        # Expected dimensions
        encodec_fps = 75  # Fixed EnCodec frame rate
        frames_per_tr = int(encodec_fps * 1.5)  # 112
        n_codebooks = 8  # For 3.0 kbps
        expected_flat_dim = n_codebooks * frames_per_tr  # 896

        assert n_trs == expected_n_trs, f"Expected {expected_n_trs} TRs, got {n_trs}"
        assert flat_dim == expected_flat_dim, (
            f"Expected flat_dim={expected_flat_dim} "
            f"({n_codebooks} codebooks × {frames_per_tr} frames), got {flat_dim}"
        )

        print(f"\n✓ Shape: {features.shape}")
        print(f"  {n_trs} TRs × {flat_dim} codes ({n_codebooks} codebooks × {frames_per_tr} frames)")

    def test_torch_stack_compatibility(self, audio_processor, sample_audio_5s):
        """Test that features can be stacked with torch.stack() (training requirement)."""
        audio_path, sr, duration = sample_audio_5s

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Convert to list of tensors (simulates batch processing)
        tr_list = [torch.tensor(features[i]) for i in range(features.shape[0])]

        # This should NOT raise RuntimeError
        try:
            stacked = torch.stack(tr_list)
            print(f"\n✓ torch.stack() succeeded: {stacked.shape}")
        except RuntimeError as e:
            pytest.fail(f"torch.stack() failed: {e}")

    def test_dtype_consistency(self, audio_processor, sample_audio_5s):
        """Test that all features have consistent dtype (int64)."""
        audio_path, sr, duration = sample_audio_5s

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Check all values are valid codes [0, 1023]
        assert np.all(features >= 0), "Found negative codes"
        assert np.all(features < 1024), "Found codes >= 1024"

        print(f"\n✓ Dtype: {features.dtype}")
        print(f"  Code range: [{features.min()}, {features.max()}]")


@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestConfigurableTR:
    """Test configurable TR length (Task 1.2 requirement)."""

    @pytest.mark.parametrize("tr_length", [1.0, 1.5, 2.0, 2.6])
    def test_different_tr_lengths(self, sample_audio_long, tr_length):
        """Test that TR length is configurable."""
        audio_path, sr, duration = sample_audio_long

        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=tr_length
        )

        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False,
            tr_length=tr_length
        )

        # Check TR count
        expected_n_trs = int(duration / tr_length)
        assert features.shape[0] == expected_n_trs, (
            f"Expected {expected_n_trs} TRs for TR={tr_length}s, got {features.shape[0]}"
        )

        # Check flat dimension
        encodec_fps = 75
        frames_per_tr = int(encodec_fps * tr_length)
        n_codebooks = 8
        expected_flat_dim = n_codebooks * frames_per_tr

        assert features.shape[1] == expected_flat_dim, (
            f"Expected flat_dim={expected_flat_dim} for TR={tr_length}s, got {features.shape[1]}"
        )

        print(f"\n✓ TR={tr_length}s: shape={features.shape}")

    def test_tr_override(self, sample_audio_5s):
        """Test that tr_length parameter overrides self.tr."""
        audio_path, sr, duration = sample_audio_5s

        # Initialize with TR=1.5s
        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=1.5
        )

        # Extract with TR=2.0s (override)
        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False,
            tr_length=2.0
        )

        # Should use 2.0s, not 1.5s
        expected_n_trs = int(duration / 2.0)
        assert features.shape[0] == expected_n_trs, (
            f"tr_length parameter did not override self.tr"
        )

        print(f"\n✓ TR override works: {features.shape[0]} TRs @ 2.0s")


@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestRoundTrip:
    """Test encoding → decoding round-trip with flattened format."""

    def test_flattened_reconstruction(self, audio_processor, sample_audio_5s, tmp_path):
        """Test that flattened codes can be decoded back to audio."""
        audio_path, sr, duration = sample_audio_5s

        # Encode (flattened)
        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        assert features.ndim == 2, "Features should be flattened"

        # Decode
        output_path = tmp_path / "reconstructed_flattened.wav"
        audio_processor.features_to_audio(features, output_path)

        # Check output exists
        assert output_path.exists()

        # Load and compare
        original, sr_orig = librosa.load(str(audio_path), sr=24000)
        reconstructed, sr_recon = librosa.load(str(output_path), sr=24000)

        # Trim to same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

        # Check correlation
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        assert correlation > 0.7, f"Low correlation: {correlation:.3f}"

        print(f"\n✓ Round-trip reconstruction successful")
        print(f"  Correlation: {correlation:.3f}")

    def test_legacy_3d_format_still_works(self, audio_processor, sample_audio_5s, tmp_path):
        """Test backwards compatibility: legacy 3D format still works."""
        audio_path, sr, duration = sample_audio_5s

        # Get flattened features
        features_flat, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Manually reshape to legacy 3D format
        n_trs, flat_dim = features_flat.shape
        n_codebooks = 8
        frames_per_tr = 112
        features_3d = features_flat.reshape(n_trs, n_codebooks, frames_per_tr)

        # Decode 3D format
        output_path = tmp_path / "reconstructed_3d.wav"
        audio_processor.features_to_audio(features_3d, output_path)

        assert output_path.exists()

        print(f"\n✓ Legacy 3D format still works")


@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_first_tr_vs_last_tr_same_shape(self, audio_processor, sample_audio_5s):
        """Test that first and last TR have identical shapes (common failure point)."""
        audio_path, sr, duration = sample_audio_5s

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        first_shape = features[0].shape
        last_shape = features[-1].shape

        assert first_shape == last_shape, (
            f"First TR shape {first_shape} != last TR shape {last_shape}. "
            f"This is the exact error from training: [1, 4, 106697] vs [1, 0, 106705]"
        )

        print(f"\n✓ First TR == Last TR: {first_shape}")

    def test_short_audio_padding(self, audio_processor, tmp_path):
        """Test handling of audio shorter than 1 TR."""
        # Generate 0.5s audio (less than 1.5s TR)
        duration = 0.5
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)

        audio_path = tmp_path / "short.wav"
        sf.write(str(audio_path), signal, sample_rate)

        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Should get 0 TRs (audio too short)
        assert features.shape[0] == 0, "Audio < 1 TR should produce 0 TRs"

        print(f"\n✓ Short audio handled correctly: {features.shape[0]} TRs")

    def test_max_trs_parameter(self, audio_processor, sample_audio_5s):
        """Test max_trs parameter limits output."""
        audio_path, sr, duration = sample_audio_5s

        max_trs = 2
        features, metadata = audio_processor.audio_to_features(
            audio_path,
            from_video=False,
            max_trs=max_trs
        )

        assert features.shape[0] == max_trs, (
            f"Expected {max_trs} TRs, got {features.shape[0]}"
        )

        # All TRs should still have consistent shape
        for i in range(max_trs):
            assert features[i].shape == features[0].shape

        print(f"\n✓ max_trs={max_trs} works: {features.shape}")


@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestBandwidthSettings:
    """Test different bandwidth settings produce consistent codebook counts."""

    @pytest.mark.parametrize("bandwidth,expected_codebooks", [
        (1.5, 2),
        (3.0, 8),
        (6.0, 16),
    ])
    def test_bandwidth_codebook_mapping(
        self, sample_audio_5s, bandwidth, expected_codebooks
    ):
        """Test that bandwidth setting produces expected codebook count.

        Note: This test creates its own processor with specific bandwidth settings.
        """
        audio_path, sr, duration = sample_audio_5s

        from giblet.data.audio import AudioProcessor
        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=bandwidth,
            tr=1.5
        )

        features, metadata = processor.audio_to_features(
            audio_path,
            from_video=False
        )

        # Check metadata
        n_codebooks = metadata['n_codebooks'].iloc[0]
        assert n_codebooks == expected_codebooks, (
            f"Bandwidth {bandwidth} kbps should produce {expected_codebooks} codebooks, "
            f"got {n_codebooks}"
        )

        # Check flattened dimension
        encodec_fps = 75
        frames_per_tr = int(encodec_fps * 1.5)
        expected_flat_dim = expected_codebooks * frames_per_tr

        assert features.shape[1] == expected_flat_dim, (
            f"Expected flat_dim={expected_flat_dim}, got {features.shape[1]}"
        )

        print(f"\n✓ Bandwidth {bandwidth} kbps → {expected_codebooks} codebooks")


if __name__ == "__main__":
    # Run tests manually
    import sys

    print("=" * 80)
    print("Audio Temporal Concatenation Tests (Issue #26, Task 1.2)")
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
        signal = np.sin(2 * np.pi * 440 * t)
        signal = signal / np.max(np.abs(signal)) * 0.9
        audio_path = test_dir / "test.wav"
        sf.write(str(audio_path), signal, sample_rate)
        print(f"  Saved: {audio_path}")

        # Run critical tests
        test_class = TestDimensionConsistency()

        print("\n1. Testing all TRs have same shape...")
        test_class.test_all_trs_same_shape((audio_path, sample_rate, duration))

        print("\n2. Testing consistent codebook count...")
        test_class.test_consistent_codebook_count((audio_path, sample_rate, duration))

        print("\n3. Testing flattened output shape...")
        test_class.test_flattened_output_shape((audio_path, sample_rate, duration))

        print("\n4. Testing torch.stack() compatibility...")
        test_class.test_torch_stack_compatibility((audio_path, sample_rate, duration))

        print("\n5. Testing dtype consistency...")
        test_class.test_dtype_consistency((audio_path, sample_rate, duration))

        print("\n" + "=" * 80)
        print("✓ All critical tests passed!")
        print("=" * 80)

    finally:
        # Cleanup
        shutil.rmtree(test_dir)

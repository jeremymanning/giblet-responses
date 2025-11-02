"""
Integration tests for EnCodec audio processing with real Sherlock data.

These tests verify that the dimension fix works correctly with actual
video data, ensuring all TRs have consistent dimensions.

Run with: pytest tests/data/test_encodec_sherlock_integration.py -v
"""

import pytest
import numpy as np
from pathlib import Path
from giblet.data.audio import AudioProcessor


# Check if Sherlock video exists
SHERLOCK_PATH = Path('data/stimuli_Sherlock.m4v')
SHERLOCK_AVAILABLE = SHERLOCK_PATH.exists()

# Skip all tests if Sherlock video is not available
pytestmark = pytest.mark.skipif(
    not SHERLOCK_AVAILABLE,
    reason="Sherlock video not found at data/stimuli_Sherlock.m4v"
)


class TestEnCodecSherlockIntegration:
    """Integration tests with real Sherlock video data"""

    @pytest.fixture
    def processor(self):
        """Create AudioProcessor with EnCodec enabled"""
        return AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=1.5
        )

    def test_small_subset(self, processor):
        """Test with 5 TRs from Sherlock - quick sanity check"""
        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=5,
            from_video=True
        )

        # Verify shape
        assert features.shape == (5, 896), f"Expected (5, 896), got {features.shape}"

        # Verify dtype
        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Verify metadata
        assert len(metadata) == 5
        assert 'n_codebooks' in metadata.columns
        assert 'n_frames' in metadata.columns

    def test_medium_subset(self, processor):
        """Test with 50 TRs from Sherlock - moderate duration"""
        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=50,
            from_video=True
        )

        # Verify shape
        assert features.shape == (50, 896), f"Expected (50, 896), got {features.shape}"

        # Verify dtype
        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Verify metadata
        assert len(metadata) == 50

    def test_large_subset(self, processor):
        """Test with 100 TRs from Sherlock - extended duration"""
        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=100,
            from_video=True
        )

        # Verify shape
        assert features.shape == (100, 896), f"Expected (100, 896), got {features.shape}"

        # Verify dtype
        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Verify metadata
        assert len(metadata) == 100

    def test_all_trs_identical_shape(self, processor):
        """Verify every TR has exactly (896,) shape - critical dimension test"""
        features, _ = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=20,
            from_video=True
        )

        # Check each TR individually
        for i in range(20):
            tr_shape = features[i].shape
            assert tr_shape == (896,), f"TR {i} has wrong shape: {tr_shape}, expected (896,)"

    def test_no_dimension_mismatch_error(self, processor):
        """Regression test - ensure the original dimension error doesn't occur"""
        # This test specifically checks that we don't get the error:
        # RuntimeError: The expanded size of the tensor (112) must match
        # the existing size (106697) at non-singleton dimension 1.

        try:
            features, _ = processor.audio_to_features(
                str(SHERLOCK_PATH),
                max_trs=10,
                from_video=True
            )

            # If we get here, no error occurred
            assert features.shape == (10, 896)

        except RuntimeError as e:
            if "expanded size" in str(e) and "must match" in str(e):
                pytest.fail(f"Dimension mismatch error still occurs: {e}")
            else:
                # Different error - re-raise
                raise

    def test_metadata_consistency(self, processor):
        """Verify metadata has consistent values across all TRs"""
        _, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=20,
            from_video=True
        )

        # All TRs should have same n_frames
        unique_n_frames = metadata['n_frames'].unique()
        assert len(unique_n_frames) == 1, f"Inconsistent n_frames: {unique_n_frames}"
        assert unique_n_frames[0] == 112, f"Expected 112 frames per TR, got {unique_n_frames[0]}"

        # All TRs should have same n_codebooks
        unique_n_codebooks = metadata['n_codebooks'].unique()
        assert len(unique_n_codebooks) == 1, f"Inconsistent n_codebooks: {unique_n_codebooks}"
        assert unique_n_codebooks[0] == 8, f"Expected 8 codebooks, got {unique_n_codebooks[0]}"

        # All TRs should have encoding_mode = 'encodec'
        assert (metadata['encoding_mode'] == 'encodec').all()

    def test_valid_codebook_indices(self, processor):
        """Verify all values are valid codebook indices (0-1023)"""
        features, _ = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=10,
            from_video=True
        )

        # EnCodec uses 1024 codebook entries (indices 0-1023)
        min_val = features.min()
        max_val = features.max()

        assert min_val >= 0, f"Negative codebook index found: {min_val}"
        assert max_val <= 1023, f"Codebook index exceeds range: {max_val}"

    def test_different_bandwidths(self):
        """Test with different EnCodec bandwidths"""
        bandwidths_and_shapes = [
            (1.5, 2, 224),   # 2 codebooks × 112 frames = 224
            (3.0, 8, 896),   # 8 codebooks × 112 frames = 896
            (6.0, 16, 1792), # 16 codebooks × 112 frames = 1792
        ]

        for bandwidth, expected_codebooks, expected_flat_dim in bandwidths_and_shapes:
            processor = AudioProcessor(
                use_encodec=True,
                encodec_bandwidth=bandwidth,
                tr=1.5
            )

            features, metadata = processor.audio_to_features(
                str(SHERLOCK_PATH),
                max_trs=5,
                from_video=True
            )

            assert features.shape == (5, expected_flat_dim), \
                f"Bandwidth {bandwidth}: Expected (5, {expected_flat_dim}), got {features.shape}"

            # Verify metadata matches
            assert (metadata['n_codebooks'] == expected_codebooks).all(), \
                f"Bandwidth {bandwidth}: Expected {expected_codebooks} codebooks, got {metadata['n_codebooks'].unique()}"

    def test_different_tr_lengths(self):
        """Test with different TR lengths"""
        tr_lengths_and_frames = [
            (1.0, 75),   # 75 Hz × 1.0s = 75 frames
            (1.5, 112),  # 75 Hz × 1.5s = 112.5 → 112 frames
            (2.0, 150),  # 75 Hz × 2.0s = 150 frames
        ]

        for tr_length, expected_frames in tr_lengths_and_frames:
            processor = AudioProcessor(
                use_encodec=True,
                encodec_bandwidth=3.0,
                tr=tr_length
            )

            expected_flat_dim = 8 * expected_frames  # 8 codebooks

            features, metadata = processor.audio_to_features(
                str(SHERLOCK_PATH),
                max_trs=5,
                from_video=True,
                tr_length=tr_length
            )

            assert features.shape == (5, expected_flat_dim), \
                f"TR={tr_length}s: Expected (5, {expected_flat_dim}), got {features.shape}"

            # Verify metadata
            assert (metadata['n_frames'] == expected_frames).all(), \
                f"TR={tr_length}s: Expected {expected_frames} frames, got {metadata['n_frames'].unique()}"

    def test_reconstruction_compatible(self, processor):
        """Verify features can be used for reconstruction"""
        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=5,
            from_video=True
        )

        # Extract metadata values
        n_codebooks = metadata['n_codebooks'].iloc[0]
        n_frames = metadata['n_frames'].iloc[0]

        # Verify we can reconstruct (don't actually save the file in test)
        # Just verify the shapes are compatible
        n_trs, flat_dim = features.shape

        assert flat_dim == n_codebooks * n_frames, \
            f"Incompatible dimensions for reconstruction: {flat_dim} != {n_codebooks} × {n_frames}"

        # Verify we can reshape to 3D
        features_3d = features.reshape(n_trs, n_codebooks, n_frames)
        assert features_3d.shape == (n_trs, n_codebooks, n_frames)

        # Verify reconstruction would work (without actually running it to save time)
        # The actual reconstruction test is in test_audio_encodec.py

    def test_sequential_extraction(self, processor):
        """Test that sequential extractions of the same data give consistent results"""
        # Extract first 10 TRs twice
        features1, metadata1 = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=10,
            from_video=True
        )

        features2, metadata2 = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=10,
            from_video=True
        )

        # Should be identical
        np.testing.assert_array_equal(features1, features2,
            err_msg="Sequential extractions produced different results")

        # Metadata should also match
        assert metadata1['n_codebooks'].equals(metadata2['n_codebooks'])
        assert metadata1['n_frames'].equals(metadata2['n_frames'])


class TestEnCodecEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_tr(self):
        """Test extraction of just 1 TR"""
        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=1.5
        )

        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=1,
            from_video=True
        )

        assert features.shape == (1, 896)
        assert len(metadata) == 1

    def test_very_short_audio(self):
        """Test with very short audio (just enough for a few TRs)"""
        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=1.5
        )

        # Extract just 2 TRs (3 seconds of audio)
        features, metadata = processor.audio_to_features(
            str(SHERLOCK_PATH),
            max_trs=2,
            from_video=True
        )

        assert features.shape == (2, 896)
        assert len(metadata) == 2

    def test_get_audio_info(self):
        """Test get_audio_info method with real Sherlock data"""
        processor = AudioProcessor(
            use_encodec=True,
            encodec_bandwidth=3.0,
            tr=1.5
        )

        info = processor.get_audio_info(str(SHERLOCK_PATH), from_video=True)

        # Verify info has expected fields
        assert 'sample_rate' in info
        assert 'duration' in info
        assert 'samples' in info
        assert 'n_trs' in info

        # Verify values make sense
        assert info['sample_rate'] > 0
        assert info['duration'] > 0
        assert info['samples'] > 0
        assert info['n_trs'] > 0

        # Sherlock video is ~48 minutes, should have many TRs
        # At TR=1.5s, that's ~1900 TRs
        assert info['n_trs'] > 1000, f"Expected >1000 TRs, got {info['n_trs']}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

"""
Unit tests for EnCodec dimension mismatch fix.

Tests the fix for bug:
RuntimeError: The expanded size of the tensor (112) must match
the existing size (106697) at non-singleton dimension 1

The bug was caused by using tr_codes.shape[1] when creating normalized_codes,
instead of using the known correct value frames_per_tr.
"""

import pytest
import numpy as np

from giblet.data.audio import AudioProcessor


# Check if EnCodec is available
try:
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(
    not ENCODEC_AVAILABLE, reason="EnCodec not available (transformers required)"
)
class TestEnCodecDimensionFix:
    """Test suite for EnCodec dimension fix."""

    @pytest.fixture
    def sherlock_video(self, data_dir):
        """Path to Sherlock video."""
        video_path = data_dir / "stimuli_Sherlock.m4v"
        if not video_path.exists():
            pytest.skip(f"Sherlock video not found at {video_path}")
        return str(video_path)

    def test_consistent_dimensions_small(self, audio_processor, sherlock_video):
        """Test that all TRs have consistent dimensions (5 TRs)."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video, max_trs=5
        )

        # Check overall shape
        assert features.shape == (5, 896), f"Wrong shape: {features.shape}"

        # Check dtype
        assert features.dtype == np.int64, f"Wrong dtype: {features.dtype}"

        # Check each TR individually
        for i in range(5):
            assert features[i].shape == (
                896,
            ), f"TR {i} wrong shape: {features[i].shape}"

        # Check metadata
        assert len(metadata) == 5
        assert all(metadata["encoding_mode"] == "encodec")
        assert all(metadata["n_codebooks"] == 8)
        assert all(metadata["n_frames"] == 112)

    def test_consistent_dimensions_medium(self, audio_processor, sherlock_video):
        """Test that all TRs have consistent dimensions (20 TRs)."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video, max_trs=20
        )

        assert features.shape == (20, 896), f"Wrong shape: {features.shape}"
        assert features.dtype == np.int64

        # Verify all TRs have same shape
        unique_shapes = set(features[i].shape for i in range(20))
        assert len(unique_shapes) == 1, f"Inconsistent shapes: {unique_shapes}"
        assert unique_shapes.pop() == (896,)

    def test_consistent_dimensions_large(self, audio_processor, sherlock_video):
        """Test that all TRs have consistent dimensions (100 TRs)."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video, max_trs=100
        )

        assert features.shape == (100, 896), f"Wrong shape: {features.shape}"
        assert features.dtype == np.int64

        # Sample check (checking all 100 would be slow)
        for i in [0, 25, 50, 75, 99]:
            assert features[i].shape == (896,), f"TR {i} wrong shape"

    def test_different_bandwidths(self, sherlock_video):
        """Test that different bandwidths produce correct dimensions."""
        bandwidth_configs = [
            (1.5, 2, 224),  # 2 codebooks × 112 frames
            (3.0, 8, 896),  # 8 codebooks × 112 frames
            (6.0, 16, 1792),  # 16 codebooks × 112 frames
        ]

        for bandwidth, n_codebooks, expected_features in bandwidth_configs:
            processor = AudioProcessor(
                use_encodec=True, encodec_bandwidth=bandwidth, tr=1.5, device="cpu"
            )

            features, metadata = processor.audio_to_features(sherlock_video, max_trs=5)

            assert features.shape == (
                5,
                expected_features,
            ), f"Bandwidth {bandwidth}: expected shape (5, {expected_features}), got {features.shape}"  # Two spaces before comment

            assert all(
                metadata["n_codebooks"] == n_codebooks
            ), (  # Two spaces before comment
                f"Bandwidth {bandwidth}: expected {n_codebooks} codebooks"
            )

    def test_different_tr_lengths(self, sherlock_video):
        """Test that different TR lengths produce correct dimensions."""
        tr_configs = [
            (1.0, 75, 600),  # 1.0s × 75Hz = 75 frames, 8 codebooks × 75 = 600
            (1.5, 112, 896),  # 1.5s × 75Hz = 112 frames, 8 codebooks × 112 = 896
            (2.0, 150, 1200),  # 2.0s × 75Hz = 150 frames, 8 codebooks × 150 = 1200
        ]

        for tr, frames_per_tr, expected_features in tr_configs:
            processor = AudioProcessor(
                use_encodec=True, encodec_bandwidth=3.0, tr=tr, device="cpu"
            )

            features, metadata = processor.audio_to_features(sherlock_video, max_trs=5)

            assert features.shape == (
                5,
                expected_features,
            ), f"TR {tr}s: expected shape (5, {expected_features}), got {features.shape}"

            assert all(
                metadata["n_frames"] == frames_per_tr
            ), f"TR {tr}s: expected {frames_per_tr} frames per TR"

    def test_no_dimension_mismatch_error(self, audio_processor, sherlock_video):
        """
        Regression test: Ensure the dimension mismatch error doesn't occur.

        This was the original bug:
        RuntimeError: The expanded size of the tensor (112) must match
        the existing size (106697) at non-singleton dimension 1
        """
        # This should NOT raise an error
        try:
            features, metadata = audio_processor.audio_to_features(
                sherlock_video, max_trs=50
            )
            # If we got here, no error occurred
            assert True
        except RuntimeError as e:
            if "expanded size" in str(e) and "must match" in str(e):
                pytest.fail(f"Dimension mismatch error still occurs: {e}")
            else:
                # Some other RuntimeError
                raise

    def test_features_are_integers(self, audio_processor, sherlock_video):
        """Test that EnCodec features are integers (codebook indices)."""
        features, _ = audio_processor.audio_to_features(sherlock_video, max_trs=10)

        # Should be integer dtype
        assert features.dtype == np.int64

        # Values should be integer codebook indices (0-1023 for EnCodec)
        assert np.all(features >= 0)
        assert np.all(features < 1024)

    def test_reconstruction_compatible(self, audio_processor, sherlock_video, tmp_path):
        """Test that features can be reconstructed to audio."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video, max_trs=10
        )

        # Try to reconstruct audio
        output_path = tmp_path / "reconstructed.wav"
        audio_processor.features_to_audio(
            features, output_path, n_codebooks=8, frames_per_tr=112
        )

        # Check that audio file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_metadata_completeness(self, audio_processor, sherlock_video):
        """Test that metadata contains all required fields."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video, max_trs=10
        )

        required_fields = [
            "tr_index",
            "start_time",
            "end_time",
            "n_frames",
            "n_codebooks",
            "encoding_mode",
        ]
        for field in required_fields:
            assert field in metadata.columns, f"Missing metadata field: {field}"

        # Check values
        assert list(metadata["tr_index"]) == list(range(10))
        assert all(metadata["encoding_mode"] == "encodec")
        assert all(metadata["n_codebooks"] == 8)
        assert all(metadata["n_frames"] == 112)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

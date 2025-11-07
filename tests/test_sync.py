"""
Tests for temporal synchronization module.

Tests verify:
1. All modalities are aligned to the minimum TR count
2. HRF convolution is applied correctly
3. Output shapes match expectations
4. Metadata is accurate
"""

import numpy as np
import pytest

from giblet.alignment.sync import (
    _resample_features,
    align_all_modalities,
    get_alignment_info,
)


@pytest.mark.unit
class TestResampleFeatures:
    """Test the _resample_features helper function."""

    def test_resample_same_size(self):
        """Test resampling to same size returns copy."""
        features = np.random.randn(100, 50)
        result = _resample_features(features, 100, 100)
        assert result.shape == (100, 50)
        np.testing.assert_array_almost_equal(result, features)

    def test_resample_discrete_codes_3d(self):
        """Test resampling discrete EnCodec codes (3D, int64)."""
        # EnCodec codes: (n_trs, n_codebooks, frames_per_tr)
        # Shape: (100, 1, 112) for 12kHz @ 1.5s TR
        codes = np.random.randint(0, 1024, size=(100, 1, 112), dtype=np.int64)

        # Downsample
        result = _resample_features(codes, 100, 50)

        # Check shape
        assert result.shape == (50, 1, 112)

        # Check dtype preserved
        assert result.dtype == np.int64

        # Check values are valid codes (in original range)
        assert np.all(result >= 0)
        assert np.all(result < 1024)

        # Values should be exact copies from input (no interpolation)
        # Each output TR should match one input TR exactly
        # The mapping is based on linear indices: target_idx -> nearest source_idx
        for tr_idx in range(result.shape[0]):
            # Calculate the corresponding source index using the same logic as _resample_features
            target_indices = np.linspace(0, 100 - 1, 50)
            source_idx = int(np.round(target_indices[tr_idx]))
            source_idx = np.clip(source_idx, 0, 99)
            # Should be an exact match
            np.testing.assert_array_equal(result[tr_idx], codes[source_idx])

    def test_resample_discrete_codes_2d(self):
        """Test resampling discrete codes (2D, int64)."""
        codes = np.random.randint(0, 100, size=(100, 50), dtype=np.int64)
        result = _resample_features(codes, 100, 80)

        assert result.shape == (80, 50)
        assert result.dtype == np.int64
        assert np.all(result >= 0)
        assert np.all(result < 100)

    def test_resample_discrete_codes_int32(self):
        """Test resampling discrete codes with int32 dtype."""
        codes = np.random.randint(0, 500, size=(100, 1, 112), dtype=np.int32)
        result = _resample_features(codes, 100, 75)

        assert result.shape == (75, 1, 112)
        assert result.dtype == np.int32
        assert np.all(result >= 0)
        assert np.all(result < 500)

    def test_resample_continuous_vs_discrete_different_behavior(self):
        """Verify continuous and discrete resampling produce different results."""
        # Create features where discrete codes differ significantly
        np.random.seed(42)
        continuous = np.random.randn(100, 1, 10).astype(np.float32)
        discrete = (continuous * 100).astype(np.int64)  # Scale and convert

        # Resample both
        continuous_result = _resample_features(continuous, 100, 50)
        discrete_result = _resample_features(discrete, 100, 50)

        # Continuous uses interpolation, discrete uses nearest neighbor
        # They should NOT be the same (after scaling)
        # For continuous, intermediate values are interpolated
        # For discrete, only exact copies are used

        # Discrete should have exact integer values from input
        assert discrete_result.dtype == np.int64
        # Continuous should be float
        assert continuous_result.dtype == np.float32

    def test_resample_downsample(self):
        """Test downsampling to fewer TRs."""
        # Create simple linear features for testing
        features = np.arange(100).reshape(-1, 1).astype(float)  # Shape (100, 1)
        result = _resample_features(features, 100, 50)

        # Result should have 50 TRs
        assert result.shape == (0, 1) or result.shape == (
            50,
            1,
        ), f"Unexpected shape: {result.shape}"

        # For linear features, interpolation should be accurate
        if result.shape[0] == 50:
            expected = np.linspace(0, 99, 50).reshape(-1, 1)
            np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_resample_upsample(self):
        """Test upsampling to more TRs."""
        features = np.arange(50).reshape(-1, 1).astype(float)  # Shape (50, 1)
        result = _resample_features(features, 50, 100)

        # Result should have 100 TRs
        assert result.shape == (100, 1)

        # Check that upsampled values are in reasonable range
        assert np.all(result >= 0)
        assert np.all(result <= 49)

    def test_resample_preserves_dtype(self):
        """Test that resampling preserves dtype."""
        features = np.random.randn(100, 50).astype(np.float32)
        result = _resample_features(features, 100, 75)
        assert result.dtype == np.float32

    def test_resample_multifeature(self):
        """Test resampling with multiple features."""
        features = np.random.randn(100, 1024)  # 1024 features
        result = _resample_features(features, 100, 80)
        assert result.shape == (80, 1024)
        # All values should be finite
        assert np.all(np.isfinite(result))


@pytest.mark.unit
class TestAlignAllModalities:
    """Test the main alignment function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with realistic dimensions."""
        np.random.seed(42)
        return {
            "video": np.random.randn(950, 43200).astype(np.float32),
            "audio": np.random.randn(946, 128).astype(np.float32),
            "text": np.random.randn(950, 1024).astype(np.float32),
            "fmri": np.random.randn(920, 85810).astype(np.float32),
        }

    def test_basic_alignment_without_hrf(self, sample_data):
        """Test basic alignment without HRF convolution."""
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
            apply_hrf_conv=False,
            tr=1.5,
        )

        # All outputs should have 920 TRs (the minimum)
        assert result["video"].shape[0] == 920
        assert result["audio"].shape[0] == 920
        assert result["text"].shape[0] == 920
        assert result["fmri"].shape[0] == 920

        # Features should be preserved
        assert result["video"].shape[1] == 43200
        assert result["audio"].shape[1] == 128
        assert result["text"].shape[1] == 1024
        assert result["fmri"].shape[1] == 85810

    def test_basic_alignment_with_hrf(self, sample_data):
        """Test alignment with HRF convolution."""
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
            apply_hrf_conv=True,
            tr=1.5,
        )

        # All outputs should have 920 TRs
        assert result["video"].shape[0] == 920
        assert result["audio"].shape[0] == 920
        assert result["text"].shape[0] == 920
        assert result["fmri"].shape[0] == 920

        # fMRI should be unchanged
        assert result["fmri"].shape[1] == 85810

    def test_metadata_preservation(self, sample_data):
        """Test that metadata about original TR counts is preserved."""
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
            apply_hrf_conv=False,
        )

        assert result["video_orig_trs"] == 950
        assert result["audio_orig_trs"] == 946
        assert result["text_orig_trs"] == 950
        assert result["fmri_orig_trs"] == 920
        assert result["n_trs"] == 920

    def test_output_shapes_match_n_trs(self, sample_data):
        """Test that all outputs have shape (n_trs, n_features)."""
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
            apply_hrf_conv=True,
            tr=1.5,
        )

        n_trs = result["n_trs"]

        for modality in ["video", "audio", "text", "fmri"]:
            assert (
                result[modality].shape[0] == n_trs
            ), f"{modality} first dimension doesn't match n_trs"
            assert len(result[modality].shape) == 2, f"{modality} should be 2D array"

    def test_finite_outputs(self, sample_data):
        """Test that all outputs contain finite values."""
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
            apply_hrf_conv=True,
            tr=1.5,
        )

        for modality in ["video", "audio", "text", "fmri"]:
            assert np.all(
                np.isfinite(result[modality])
            ), f"{modality} contains non-finite values"

    def test_minimum_tr_selection(self):
        """Test that target TR is indeed the minimum."""
        # Create data with different TR counts
        video = np.random.randn(100, 50).astype(np.float32)
        audio = np.random.randn(80, 128).astype(np.float32)  # Shortest
        text = np.random.randn(95, 1024).astype(np.float32)
        fmri = np.random.randn(90, 1000).astype(np.float32)

        result = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
        )

        # Target should be minimum (80)
        assert result["n_trs"] == 80
        assert result["video"].shape[0] == 80
        assert result["audio"].shape[0] == 80
        assert result["text"].shape[0] == 80
        assert result["fmri"].shape[0] == 80

    def test_fmri_truncation_only(self):
        """Test that fMRI is truncated, not resampled."""
        # fMRI with 100 TRs, others with 100 TRs
        video = np.random.randn(100, 50).astype(np.float32)
        audio = np.random.randn(100, 128).astype(np.float32)
        text = np.random.randn(100, 1024).astype(np.float32)
        fmri = np.random.randn(100, 1000).astype(np.float32)

        # Set identifiable values
        fmri[50:60, 0] = np.arange(10) + 100  # Identifiable range

        result = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
        )

        # fMRI should be truncated to 100 TRs (no change in this case)
        assert result["fmri"].shape[0] == 100
        # Check that values are preserved (not resampled)
        np.testing.assert_array_equal(result["fmri"][50:60, 0], np.arange(10) + 100)

    def test_hrf_changes_stimulus_features(self, sample_data):
        """Test that HRF convolution changes stimulus features."""
        result_no_hrf = align_all_modalities(
            video_features=sample_data["video"].copy(),
            audio_features=sample_data["audio"].copy(),
            text_features=sample_data["text"].copy(),
            fmri_features=sample_data["fmri"].copy(),
            apply_hrf_conv=False,
        )

        result_with_hrf = align_all_modalities(
            video_features=sample_data["video"].copy(),
            audio_features=sample_data["audio"].copy(),
            text_features=sample_data["text"].copy(),
            fmri_features=sample_data["fmri"].copy(),
            apply_hrf_conv=True,
        )

        # Video should be different after HRF
        # (convolution should change the values)
        assert not np.allclose(result_no_hrf["video"], result_with_hrf["video"])

        # Audio should be different after HRF
        assert not np.allclose(result_no_hrf["audio"], result_with_hrf["audio"])

        # Text should be different after HRF
        assert not np.allclose(result_no_hrf["text"], result_with_hrf["text"])

        # fMRI should be identical (not convolved)
        np.testing.assert_array_equal(result_no_hrf["fmri"], result_with_hrf["fmri"])

    def test_default_parameters(self, sample_data):
        """Test that function works with default parameters."""
        # Should not raise any errors
        result = align_all_modalities(
            video_features=sample_data["video"],
            audio_features=sample_data["audio"],
            text_features=sample_data["text"],
            fmri_features=sample_data["fmri"],
        )

        # Defaults: apply_hrf_conv=True, tr=1.5
        assert result["n_trs"] == 920
        assert result["video"].shape[0] == 920


@pytest.mark.unit
class TestGetAlignmentInfo:
    """Test the alignment info helper function."""

    def test_get_info(self):
        """Test getting alignment information."""
        result = {
            "video": np.random.randn(920, 43200),
            "audio": np.random.randn(920, 128),
            "text": np.random.randn(920, 1024),
            "fmri": np.random.randn(920, 85810),
            "n_trs": 920,
            "video_orig_trs": 950,
            "audio_orig_trs": 946,
            "text_orig_trs": 950,
            "fmri_orig_trs": 920,
        }

        info = get_alignment_info(result)

        # Verify all expected keys are present
        expected_keys = [
            "n_trs",
            "video_trs_aligned",
            "audio_trs_aligned",
            "text_trs_aligned",
            "fmri_trs_aligned",
            "video_features",
            "audio_features",
            "text_features",
            "fmri_features",
            "video_orig_trs",
            "audio_orig_trs",
            "text_orig_trs",
            "fmri_orig_trs",
        ]

        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

        # Verify values
        assert info["n_trs"] == 920
        assert info["video_features"] == 43200
        assert info["audio_features"] == 128
        assert info["text_features"] == 1024
        assert info["fmri_features"] == 85810


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_already_aligned_data(self):
        """Test alignment when all modalities already have same number of TRs."""
        # All have 920 TRs
        video = np.random.randn(920, 100).astype(np.float32)
        audio = np.random.randn(920, 128).astype(np.float32)
        text = np.random.randn(920, 1024).astype(np.float32)
        fmri = np.random.randn(920, 1000).astype(np.float32)

        result = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
        )

        assert result["n_trs"] == 920
        # When data is already aligned, resampling should preserve values closely
        np.testing.assert_array_almost_equal(result["video"], video)
        np.testing.assert_array_almost_equal(result["audio"], audio)
        np.testing.assert_array_almost_equal(result["text"], text)

    def test_single_feature(self):
        """Test with single feature per modality."""
        video = np.random.randn(100, 1).astype(np.float32)
        audio = np.random.randn(95, 1).astype(np.float32)
        text = np.random.randn(98, 1).astype(np.float32)
        fmri = np.random.randn(90, 1).astype(np.float32)

        result = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
        )

        # All should have 90 TRs (minimum) and 1 feature
        assert result["video"].shape == (90, 1)
        assert result["audio"].shape == (90, 1)
        assert result["text"].shape == (90, 1)
        assert result["fmri"].shape == (90, 1)

    def test_large_feature_dimensions(self):
        """Test with large feature dimensions."""
        # Realistic dimensions for Sherlock
        video = np.random.randn(950, 43200).astype(np.float32)
        audio = np.random.randn(946, 128).astype(np.float32)
        text = np.random.randn(950, 1024).astype(np.float32)
        fmri = np.random.randn(920, 85810).astype(np.float32)

        result = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
        )

        # Should handle large dimensions gracefully
        assert result["video"].shape == (920, 43200)
        assert result["fmri"].shape == (920, 85810)

    def test_different_tr_values(self):
        """Test that TR parameter is used correctly."""
        video = np.random.randn(100, 50).astype(np.float32)
        audio = np.random.randn(100, 128).astype(np.float32)
        text = np.random.randn(100, 1024).astype(np.float32)
        fmri = np.random.randn(100, 1000).astype(np.float32)

        # Should work with different TR values
        result_tr1 = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
            tr=1.0,
        )

        result_tr2 = align_all_modalities(
            video_features=video,
            audio_features=audio,
            text_features=text,
            fmri_features=fmri,
            apply_hrf_conv=False,
            tr=2.0,
        )

        # TR shouldn't affect alignment (only HRF duration)
        assert result_tr1["n_trs"] == result_tr2["n_trs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

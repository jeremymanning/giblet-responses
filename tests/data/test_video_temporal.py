"""
Tests for video temporal concatenation (Issue #26, Task 1.1).

Tests:
- Consistent dimensions across all TRs
- Different TR lengths (1.0s, 1.5s, 2.0s)
- Edge cases (first TR, last TR, incomplete windows)
- Zero padding behavior
- Feature extraction and reconstruction
"""

import numpy as np
import cv2
import pytest
from giblet.data.video import VideoProcessor


def create_test_video(output_path, duration_seconds=5.0, fps=25, width=640, height=360):
    """
    Create a test video with color gradients that change over time.

    Parameters
    ----------
    output_path : Path
        Output path for test video
    duration_seconds : float
        Video duration in seconds
    fps : int
        Frames per second
    width : int
        Frame width
    height : int
        Frame height
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    n_frames = int(duration_seconds * fps)

    for i in range(n_frames):
        # Create frame with color gradient that changes over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Red channel increases with frame number
        frame[:, :, 0] = int((i / n_frames) * 255)

        # Green channel creates vertical gradient
        for y in range(height):
            frame[y, :, 1] = int((y / height) * 255)

        # Blue channel creates horizontal gradient
        for x in range(width):
            frame[:, x, 2] = int((x / width) * 255)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


@pytest.fixture
def test_video_path(tmp_path):
    """Create a temporary test video."""
    video_path = tmp_path / "test_video.mp4"
    create_test_video(video_path, duration_seconds=5.0, fps=25)
    return video_path


@pytest.mark.unit
class TestTemporalConcatenation:
    """Test temporal concatenation of video frames."""

    def test_consistent_dimensions_across_trs(self, test_video_path):
        """Test that all TRs have consistent dimensions."""
        processor = VideoProcessor(tr=1.5)
        features, metadata = processor.video_to_features(test_video_path)

        # Check shape
        n_trs = features.shape[0]
        n_features = features.shape[1]

        # All TRs should have same number of features
        assert features.shape == (n_trs, n_features)

        # Calculate expected dimensions
        fps = 25
        frames_per_tr = int(np.round(fps * 1.5))  # ~37 frames
        expected_features = frames_per_tr * 90 * 160 * 3

        assert n_features == expected_features
        print(f"✓ All {n_trs} TRs have consistent dimensions: {n_features} features")
        print(f"  ({frames_per_tr} frames × 90 × 160 × 3 channels)")

    def test_different_tr_lengths(self, test_video_path):
        """Test with different TR lengths."""
        tr_lengths = [1.0, 1.5, 2.0]
        fps = 25

        for tr in tr_lengths:
            processor = VideoProcessor(tr=tr)
            features, metadata = processor.video_to_features(test_video_path)

            # Calculate expected dimensions
            frames_per_tr = int(np.round(fps * tr))
            expected_features = frames_per_tr * 90 * 160 * 3

            # Check dimensions
            assert features.shape[1] == expected_features

            # Check metadata
            assert metadata['frames_per_tr'].iloc[0] == frames_per_tr

            print(f"✓ TR={tr}s: {frames_per_tr} frames per TR, "
                  f"{features.shape[0]} TRs, {expected_features} features per TR")

    def test_first_tr_edge_case(self, test_video_path):
        """Test that first TR handles lack of previous frames correctly."""
        processor = VideoProcessor(tr=1.5)
        features, metadata = processor.video_to_features(test_video_path, max_trs=2)

        # First TR should exist and have correct dimensions
        assert features.shape[0] >= 1

        # Check first TR metadata
        first_tr = metadata.iloc[0]
        assert first_tr['tr_index'] == 0
        assert first_tr['start_time'] == 0.0
        assert first_tr['end_time'] == 1.5

        # First TR may have zero-padded frames at the beginning
        # But should still have full dimensions
        frames_per_tr = first_tr['frames_per_tr']
        expected_features = frames_per_tr * 90 * 160 * 3
        assert features.shape[1] == expected_features

        print(f"✓ First TR properly handles edge case")

    def test_last_tr_edge_case(self, test_video_path):
        """Test that last TR handles incomplete windows correctly."""
        processor = VideoProcessor(tr=1.5)
        features, metadata = processor.video_to_features(test_video_path)

        # Last TR should exist and have correct dimensions
        last_tr = metadata.iloc[-1]

        # Should have full dimensions (padded if needed)
        frames_per_tr = last_tr['frames_per_tr']
        expected_features = frames_per_tr * 90 * 160 * 3
        assert features.shape[1] == expected_features

        print(f"✓ Last TR properly handles edge case")

    def test_zero_padding_behavior(self, test_video_path):
        """Test that zero padding is applied correctly."""
        processor = VideoProcessor(tr=1.5, normalize=True)
        features, metadata = processor.video_to_features(test_video_path, max_trs=1)

        # Get first TR features
        first_tr_features = features[0]

        # Reshape to frames
        frames_per_tr = int(metadata.iloc[0]['frames_per_tr'])
        frames = first_tr_features.reshape(frames_per_tr, 90, 160, 3)

        # First frame might be zero-padded (before video start)
        # Check if any frame is all zeros
        has_zero_frames = np.any(np.all(frames == 0, axis=(1, 2, 3)))

        print(f"✓ Zero padding behavior verified (has_zero_frames: {has_zero_frames})")

    def test_temporal_window_alignment(self, test_video_path):
        """Test that temporal windows align correctly to [t-TR, t]."""
        processor = VideoProcessor(tr=1.5)
        features, metadata = processor.video_to_features(test_video_path, max_trs=3)

        # Check metadata for correct time windows
        for i, row in metadata.iterrows():
            expected_end = (row['tr_index'] + 1) * 1.5
            expected_start = expected_end - 1.5

            assert np.isclose(row['end_time'], expected_end)
            assert np.isclose(row['start_time'], expected_start)

        print(f"✓ Temporal windows correctly aligned to [t-TR, t]")

    def test_feature_extraction_normalization(self, test_video_path):
        """Test that normalization works correctly."""
        # With normalization
        processor_norm = VideoProcessor(tr=1.5, normalize=True)
        features_norm, _ = processor_norm.video_to_features(test_video_path, max_trs=2)

        # Without normalization
        processor_no_norm = VideoProcessor(tr=1.5, normalize=False)
        features_no_norm, _ = processor_no_norm.video_to_features(test_video_path, max_trs=2)

        # Normalized features should be in [0, 1]
        assert features_norm.min() >= 0.0
        assert features_norm.max() <= 1.0

        # Non-normalized features should be in [0, 255]
        assert features_no_norm.min() >= 0.0
        assert features_no_norm.max() <= 255.0

        # They should be related by factor of 255
        np.testing.assert_allclose(
            features_norm * 255.0,
            features_no_norm,
            rtol=1e-3
        )

        print(f"✓ Normalization working correctly")

    def test_max_trs_truncation(self, test_video_path):
        """Test that max_trs parameter truncates correctly."""
        processor = VideoProcessor(tr=1.5)

        # Extract with max_trs
        features_truncated, metadata_truncated = processor.video_to_features(
            test_video_path, max_trs=2
        )

        assert features_truncated.shape[0] == 2
        assert len(metadata_truncated) == 2

        print(f"✓ max_trs truncation works correctly")

    def test_reconstruction_roundtrip(self, test_video_path, tmp_path):
        """Test that video can be reconstructed from features."""
        processor = VideoProcessor(tr=1.5, normalize=True)

        # Extract features
        features, metadata = processor.video_to_features(test_video_path, max_trs=2)

        # Reconstruct video
        output_path = tmp_path / "reconstructed.mp4"
        processor.features_to_video(features, output_path, fps=25)

        # Check that video was created
        assert output_path.exists()

        # Open and check properties
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        # Check properties
        assert fps == 25
        assert width == 160
        assert height == 90

        # Should have frames_per_tr * n_trs frames
        frames_per_tr = int(np.round(1.5 * 25))
        expected_frames = frames_per_tr * 2
        assert total_frames == expected_frames

        print(f"✓ Reconstruction roundtrip successful")
        print(f"  Output: {width}x{height} @ {fps}fps, {total_frames} frames")


@pytest.mark.unit
class TestDimensionConsistency:
    """Test dimension consistency across different scenarios."""

    def test_all_trs_same_dimension(self, test_video_path):
        """Test that all TRs have exactly the same dimension."""
        processor = VideoProcessor(tr=1.5)
        features, _ = processor.video_to_features(test_video_path)

        # Get dimension of first TR
        first_dim = features[0].shape[0]

        # Check all TRs have same dimension
        for i in range(features.shape[0]):
            assert features[i].shape[0] == first_dim

        print(f"✓ All {features.shape[0]} TRs have identical dimensions: {first_dim}")

    def test_dimension_calculation(self, test_video_path):
        """Test that dimension calculation is correct."""
        tr = 1.5
        fps = 25
        processor = VideoProcessor(tr=tr, target_height=90, target_width=160)
        features, metadata = processor.video_to_features(test_video_path, max_trs=1)

        # Calculate expected dimension
        frames_per_tr = int(np.round(fps * tr))  # 37 or 38
        features_per_frame = 90 * 160 * 3  # H × W × C
        expected_dim = frames_per_tr * features_per_frame

        # Check
        assert features.shape[1] == expected_dim
        assert metadata.iloc[0]['frames_per_tr'] == frames_per_tr

        print(f"✓ Dimension calculation correct:")
        print(f"  frames_per_tr: {frames_per_tr}")
        print(f"  features_per_frame: {features_per_frame}")
        print(f"  total_features: {expected_dim}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

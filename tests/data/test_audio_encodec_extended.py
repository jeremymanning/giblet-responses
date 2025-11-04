"""
Extended EnCodec tests with real Sherlock data (Issue #28).

All tests use REAL data from stimuli_Sherlock.m4v.
NO MOCKS OR SIMULATIONS.

Test coverage:
1. EnCodec with real Sherlock audio at multiple scales (5, 100 TRs)
2. Dimension consistency verification across all TRs
3. Audio quality measurement with STOI
4. Round-trip encoding/decoding
"""

import pytest
import numpy as np
from pathlib import Path

from giblet.data.audio import AudioProcessor, ENCODEC_AVAILABLE

# Audio quality metrics
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# Audio I/O
import librosa


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not ENCODEC_AVAILABLE, reason="EnCodec not available")
class TestEnCodecRealData:
    """EnCodec tests with real Sherlock data."""

    @pytest.fixture
    def sherlock_video_path(self, data_dir):
        """Path to real Sherlock video file."""
        path = data_dir / 'stimuli_Sherlock.m4v'
        if not path.exists():
            pytest.skip(f"Sherlock video not found: {path}")
        return path

    def test_encodec_sherlock_5trs(self, audio_processor, sherlock_video_path):
        """Test EnCodec with 5 TRs of real Sherlock audio."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video_path,
            max_trs=5,
            from_video=True
        )

        # Verify shape
        assert features.shape == (5, 896), f"Expected (5, 896), got {features.shape}"
        assert features.dtype == np.int64, f"Expected int64, got {features.dtype}"

        # Verify metadata
        assert len(metadata) == 5
        assert all(metadata['encoding_mode'] == 'encodec')

        print(f"\n✓ 5 TRs test passed: {features.shape}")

    def test_encodec_sherlock_100trs(self, audio_processor, sherlock_video_path):
        """Test EnCodec with 100 TRs (full preprocessing scale)."""
        features, metadata = audio_processor.audio_to_features(
            sherlock_video_path,
            max_trs=100,
            from_video=True
        )

        # Verify shape
        assert features.shape == (100, 896), f"Expected (100, 896), got {features.shape}"
        assert features.dtype == np.int64

        # Verify all TRs processed
        assert len(metadata) == 100

        print(f"\n✓ 100 TRs test passed: {features.shape}")

    def test_encodec_dimension_consistency(self, audio_processor, sherlock_video_path):
        """Test that ALL TRs have exactly 896 codes (8 codebooks × 112 frames)."""
        features, _ = audio_processor.audio_to_features(
            sherlock_video_path,
            max_trs=50,
            from_video=True
        )

        # Check every TR individually
        for tr_idx in range(features.shape[0]):
            tr_codes = features[tr_idx]
            assert tr_codes.shape == (896,), \
                f"TR {tr_idx} has wrong shape: {tr_codes.shape} (expected (896,))"

            # Verify codes are in valid range (EnCodec uses 1024-entry codebook)
            assert tr_codes.min() >= 0, f"TR {tr_idx} has negative codes: {tr_codes.min()}"
            assert tr_codes.max() < 1024, f"TR {tr_idx} has codes >= 1024: {tr_codes.max()}"

        print(f"\n✓ Dimension consistency verified for 50 TRs")
        print(f"  All TRs have shape (896,)")
        print(f"  All codes in range [0, 1023]")

    @pytest.mark.skipif(not STOI_AVAILABLE, reason="pystoi not available")
    def test_encodec_quality_sherlock(self, audio_processor, sherlock_video_path, tmp_path):
        """Test EnCodec quality on real Sherlock audio segment."""
        # Encode 10 seconds of Sherlock
        duration = 10.0
        max_trs = int(duration / 1.5)

        features, _ = audio_processor.audio_to_features(
            sherlock_video_path,
            max_trs=max_trs,
            from_video=True
        )

        # Decode
        output_path = tmp_path / 'sherlock_encodec_quality.wav'
        audio_processor.features_to_audio(features, output_path)

        # Load original and reconstructed at 24kHz (EnCodec output rate)
        y_orig, sr = librosa.load(str(sherlock_video_path), sr=24000, duration=duration)
        y_recon, _ = librosa.load(str(output_path), sr=24000)

        # Compute STOI
        min_len = min(len(y_orig), len(y_recon))
        stoi_score = stoi(y_orig[:min_len], y_recon[:min_len], sr, extended=False)

        # User-approved quality: STOI ~0.74 for 3.0 kbps
        # Accept >= 0.6 to account for test variability
        assert stoi_score >= 0.6, f"STOI too low: {stoi_score:.3f}"

        print(f"\n✓ Quality test passed on real Sherlock audio")
        print(f"  STOI: {stoi_score:.3f} (expected ~0.74)")

    def test_encodec_round_trip_sherlock(self, audio_processor, sherlock_video_path, tmp_path):
        """Test round-trip encoding/decoding with real Sherlock audio."""
        # Encode 5 TRs
        features, _ = audio_processor.audio_to_features(
            sherlock_video_path,
            max_trs=5,
            from_video=True
        )

        # Decode
        output_path = tmp_path / 'sherlock_roundtrip.wav'
        audio_processor.features_to_audio(features, output_path)

        # Verify output exists and has correct format
        assert output_path.exists()

        # Load and check duration (5 TRs × 1.5s = 7.5s)
        y_recon, sr = librosa.load(str(output_path), sr=24000)
        duration = len(y_recon) / sr

        # Should be close to 7.5s (allow ±0.5s for boundary effects)
        assert 7.0 <= duration <= 8.0, f"Duration {duration:.2f}s not in expected range [7.0, 8.0]"

        print(f"\n✓ Round-trip test passed")
        print(f"  Reconstructed duration: {duration:.2f}s (expected ~7.5s)")


if __name__ == "__main__":
    # Run tests standalone
    pytest.main([__file__, "-v"])

"""
Tests for EnCodec-aware audio decoding in MultimodalDecoder.

Tests the new EnCodec discrete code prediction path and backwards
compatibility with mel spectrograms.
"""

import pytest
import torch
import numpy as np
from giblet.models.decoder import MultimodalDecoder


@pytest.mark.unit


class TestEnCodecAudioDecoder:
    """Test suite for EnCodec audio decoding."""

    @pytest.fixture
    def encodec_decoder(self):
        """Create decoder with EnCodec enabled."""
        return MultimodalDecoder(
            bottleneck_dim=2048,
            video_dim=43200,
            audio_dim=2048,
            audio_frames_per_tr=112,  # EnCodec @ 75Hz
            text_dim=1024,
            dropout=0.3,
            use_encodec=True,
            n_codebooks=8
        )

    @pytest.fixture
    def mel_decoder(self):
        """Create decoder with mel spectrograms (legacy)."""
        return MultimodalDecoder(
            bottleneck_dim=2048,
            video_dim=43200,
            audio_dim=2048,
            audio_frames_per_tr=65,  # Mel @ 44Hz
            text_dim=1024,
            dropout=0.3,
            use_encodec=False
        )

    def test_encodec_initialization(self, encodec_decoder):
        """Test decoder initializes correctly with EnCodec enabled."""
        assert encodec_decoder.use_encodec is True
        assert encodec_decoder.n_codebooks == 8
        assert encodec_decoder.audio_frames_per_tr == 112

        # EnCodec decoder should not have temporal upsampling layers
        assert encodec_decoder.audio_temporal_init_frames is None
        assert encodec_decoder.audio_temporal_upsample is None
        assert encodec_decoder.audio_temporal_adjust is None

    def test_mel_initialization(self, mel_decoder):
        """Test decoder initializes correctly with mel spectrograms."""
        assert mel_decoder.use_encodec is False
        assert mel_decoder.audio_frames_per_tr == 65

        # Mel decoder should have temporal upsampling layers
        assert mel_decoder.audio_temporal_init_frames == 8
        assert mel_decoder.audio_temporal_upsample is not None
        assert mel_decoder.audio_temporal_adjust is not None

    def test_encodec_output_shape(self, encodec_decoder):
        """Test EnCodec decoder outputs correct shape."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        video, audio, text = encodec_decoder(bottleneck)

        # Check shapes
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 8, 112), f"Expected (32, 8, 112), got {audio.shape}"
        assert text.shape == (batch_size, 1024)

    def test_encodec_code_range_training(self, encodec_decoder):
        """Test EnCodec codes are in valid range during training."""
        batch_size = 16
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.train()
        _, audio, _ = encodec_decoder(bottleneck)

        # During training, codes should be continuous in [0, 1023]
        assert audio.dtype == torch.float32
        assert torch.all(audio >= 0), f"Min value: {audio.min()}"
        assert torch.all(audio <= 1023), f"Max value: {audio.max()}"

    def test_encodec_code_range_inference(self, encodec_decoder):
        """Test EnCodec codes are discrete integers during inference."""
        batch_size = 16
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        _, audio, _ = encodec_decoder(bottleneck)

        # During inference, codes should be rounded integers in [0, 1023]
        assert audio.dtype == torch.float32  # Still float, but rounded
        assert torch.all(audio >= 0)
        assert torch.all(audio <= 1023)

        # Check that values are integers (no fractional part)
        assert torch.all(audio == torch.round(audio)), "Codes should be integers during inference"

    def test_encodec_code_distribution(self, encodec_decoder):
        """Test EnCodec codes use full vocabulary range."""
        batch_size = 64  # Larger batch for better statistics
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        _, audio, _ = encodec_decoder(bottleneck)

        # Codes should span a reasonable range of the vocabulary
        # (not all clustered at 0 or 1023)
        unique_codes = torch.unique(audio)
        assert len(unique_codes) > 100, "Codes should use diverse vocabulary"
        assert audio.min() < 200, "Should have some low codes"
        assert audio.max() > 800, "Should have some high codes"

    def test_backward_compatibility_mel(self, mel_decoder):
        """Test mel spectrogram path still works."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        mel_decoder.eval()
        video, audio, text = mel_decoder(bottleneck)

        # Check shapes for mel path
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65), f"Expected (32, 2048, 65), got {audio.shape}"
        assert text.shape == (batch_size, 1024)

    def test_gradient_flow_encodec(self, encodec_decoder):
        """Test gradients flow through EnCodec path."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 2048, requires_grad=True)

        encodec_decoder.train()
        video, audio, text = encodec_decoder(bottleneck)

        # Create dummy loss
        loss = video.sum() + audio.sum() + text.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert bottleneck.grad is not None
        assert not torch.isnan(bottleneck.grad).any()

        # Check all parameters have gradients
        for name, param in encodec_decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_decode_audio_only_encodec(self, encodec_decoder):
        """Test audio-only decoding with EnCodec."""
        batch_size = 16
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        audio = encodec_decoder.decode_audio_only(bottleneck)

        # Check shape and properties
        assert audio.shape == (batch_size, 8, 112)
        assert torch.all(audio >= 0)
        assert torch.all(audio <= 1023)
        assert torch.all(audio == torch.round(audio))

    def test_decode_audio_only_mel(self, mel_decoder):
        """Test audio-only decoding with mel spectrograms."""
        batch_size = 16
        bottleneck = torch.randn(batch_size, 2048)

        mel_decoder.eval()
        audio = mel_decoder.decode_audio_only(bottleneck)

        # Check shape
        assert audio.shape == (batch_size, 2048, 65)

    def test_training_vs_inference_consistency(self, encodec_decoder):
        """Test that training and inference modes produce outputs in expected range."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        # Training mode - outputs continuous values
        encodec_decoder.train()
        _, audio_train, _ = encodec_decoder(bottleneck)
        assert torch.all(audio_train >= 0)
        assert torch.all(audio_train <= 1023)

        # Inference mode - outputs discrete integers
        encodec_decoder.eval()
        _, audio_eval, _ = encodec_decoder(bottleneck)
        assert torch.all(audio_eval >= 0)
        assert torch.all(audio_eval <= 1023)
        assert torch.all(audio_eval == torch.round(audio_eval))

        # Both modes should produce valid codes
        # (exact values may differ due to dropout/batch norm, but both should be valid)

    def test_no_nan_or_inf_encodec(self, encodec_decoder):
        """Test EnCodec path produces no NaN or Inf values."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        video, audio, text = encodec_decoder(bottleneck)

        # Check no NaN or Inf
        assert not torch.isnan(video).any()
        assert not torch.isnan(audio).any()
        assert not torch.isnan(text).any()
        assert not torch.isinf(video).any()
        assert not torch.isinf(audio).any()
        assert not torch.isinf(text).any()

    def test_batch_size_variations_encodec(self, encodec_decoder):
        """Test EnCodec decoder works with various batch sizes."""
        batch_sizes = [1, 4, 16, 32, 64, 128]

        encodec_decoder.eval()

        for batch_size in batch_sizes:
            bottleneck = torch.randn(batch_size, 2048)
            video, audio, text = encodec_decoder(bottleneck)

            assert video.shape == (batch_size, 43200)
            assert audio.shape == (batch_size, 8, 112)
            assert text.shape == (batch_size, 1024)
            assert torch.all(audio >= 0)
            assert torch.all(audio <= 1023)

    def test_deterministic_inference_encodec(self, encodec_decoder):
        """Test EnCodec decoder is deterministic in eval mode."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()

        # Multiple forward passes should be identical
        outputs1 = [encodec_decoder(bottleneck) for _ in range(3)]
        outputs2 = [encodec_decoder(bottleneck) for _ in range(3)]

        for i in range(3):
            video1, audio1, text1 = outputs1[i]
            video2, audio2, text2 = outputs2[i]

            assert torch.allclose(video1, video2)
            assert torch.allclose(audio1, audio2)
            assert torch.allclose(text1, text2)

    def test_zero_bottleneck_encodec(self, encodec_decoder):
        """Test EnCodec decoder handles zero input gracefully."""
        batch_size = 4
        bottleneck = torch.zeros(batch_size, 2048)

        encodec_decoder.eval()
        video, audio, text = encodec_decoder(bottleneck)

        # Should still produce valid outputs (not all zeros due to biases)
        assert not torch.allclose(audio, torch.zeros_like(audio))
        assert torch.all(audio >= 0)
        assert torch.all(audio <= 1023)
        assert not torch.isnan(audio).any()

    def test_code_statistics(self, encodec_decoder):
        """Test EnCodec code statistics are reasonable."""
        batch_size = 64
        bottleneck = torch.randn(batch_size, 2048)

        encodec_decoder.eval()
        _, audio, _ = encodec_decoder(bottleneck)

        # Codes should be somewhat spread across the range
        mean_code = audio.float().mean()
        std_code = audio.float().std()

        # Mean should be roughly in middle of range
        assert 200 < mean_code < 800, f"Mean code: {mean_code}"

        # Should have reasonable variance
        assert std_code > 50, f"Std: {std_code}"

    def test_different_n_codebooks(self):
        """Test decoder works with different numbers of codebooks."""
        batch_size = 8

        for n_codebooks in [4, 8, 16]:
            decoder = MultimodalDecoder(
                bottleneck_dim=2048,
                use_encodec=True,
                n_codebooks=n_codebooks,
                audio_frames_per_tr=112
            )

            decoder.eval()
            bottleneck = torch.randn(batch_size, 2048)
            _, audio, _ = decoder(bottleneck)

            assert audio.shape == (batch_size, n_codebooks, 112)
            assert torch.all(audio >= 0)
            assert torch.all(audio <= 1023)

    def test_different_frames_per_tr(self):
        """Test decoder works with different frame counts."""
        batch_size = 8

        for frames_per_tr in [56, 112, 224]:
            decoder = MultimodalDecoder(
                bottleneck_dim=2048,
                use_encodec=True,
                n_codebooks=8,
                audio_frames_per_tr=frames_per_tr
            )

            decoder.eval()
            bottleneck = torch.randn(batch_size, 2048)
            _, audio, _ = decoder(bottleneck)

            assert audio.shape == (batch_size, 8, frames_per_tr)

    def test_parameter_count_encodec_vs_mel(self, encodec_decoder, mel_decoder):
        """Test parameter counts for EnCodec vs mel decoders."""
        encodec_params = encodec_decoder.count_parameters()
        mel_params = mel_decoder.count_parameters()

        # EnCodec decoder should have similar or fewer parameters
        # (no temporal upsampling layers)
        print(f"\nEnCodec decoder parameters: {encodec_params['total']:,}")
        print(f"Mel decoder parameters: {mel_params['total']:,}")

        # EnCodec should be smaller (no Conv1D upsampling)
        assert encodec_params['total'] <= mel_params['total']


@pytest.mark.integration
class TestEnCodecIntegration:
    """Integration tests for EnCodec audio decoder."""

    @pytest.mark.slow
    def test_realistic_fmri_bottleneck(self):
        """Test with realistic fMRI bottleneck dimensions."""
        batch_size = 920  # Full Sherlock dataset
        bottleneck_dim = 2048

        decoder = MultimodalDecoder(
            bottleneck_dim=bottleneck_dim,
            use_encodec=True,
            n_codebooks=8,
            audio_frames_per_tr=112
        )

        decoder.eval()

        # Process in smaller chunks (memory efficiency)
        chunk_size = 64
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_bottleneck = torch.randn(end - start, bottleneck_dim)

            video, audio, text = decoder(chunk_bottleneck)

            assert audio.shape == (end - start, 8, 112)
            assert torch.all(audio >= 0)
            assert torch.all(audio <= 1023)

    def test_mixed_training_inference(self):
        """Test switching between training and inference modes."""
        decoder = MultimodalDecoder(
            bottleneck_dim=2048,
            use_encodec=True
        )

        batch_size = 16
        bottleneck = torch.randn(batch_size, 2048)

        # Train mode
        decoder.train()
        _, audio_train, _ = decoder(bottleneck)
        assert audio_train.dtype == torch.float32
        # May have fractional values

        # Eval mode
        decoder.eval()
        _, audio_eval, _ = decoder(bottleneck)
        assert audio_eval.dtype == torch.float32
        # Should be integers
        assert torch.all(audio_eval == torch.round(audio_eval))

        # Back to train
        decoder.train()
        _, audio_train2, _ = decoder(bottleneck)
        # May have fractional values again

    def test_memory_efficiency(self):
        """Test decoder doesn't use excessive memory."""
        decoder = MultimodalDecoder(
            bottleneck_dim=2048,
            use_encodec=True
        )

        decoder.eval()

        # Process large batch
        batch_size = 256
        bottleneck = torch.randn(batch_size, 2048)

        # Should complete without memory error
        video, audio, text = decoder(bottleneck)

        assert audio.shape == (batch_size, 8, 112)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

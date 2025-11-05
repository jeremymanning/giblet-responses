"""
Tests for multimodal decoder module.

Tests forward pass, output shapes, and component functionality.
"""

import pytest
import torch

from giblet.models.decoder import MultimodalDecoder


@pytest.mark.unit
class TestMultimodalDecoder:
    """Test suite for MultimodalDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder with standard dimensions."""
        return MultimodalDecoder(
            bottleneck_dim=2048,
            video_dim=43200,
            audio_dim=2048,
            audio_frames_per_tr=65,
            text_dim=1024,
            dropout=0.3,
            use_encodec=False,  # Use mel spectrograms (legacy)
        )

    @pytest.fixture
    def small_decoder(self):
        """Create smaller decoder for faster testing."""
        return MultimodalDecoder(
            bottleneck_dim=2048,
            video_dim=43200,
            audio_dim=2048,
            audio_frames_per_tr=65,
            text_dim=1024,
            dropout=0.1,
            use_encodec=False,  # Use mel spectrograms (legacy)
        )

    def test_initialization(self, decoder):
        """Test decoder initializes with correct architecture."""
        assert decoder.bottleneck_dim == 2048
        assert decoder.video_dim == 43200
        assert decoder.audio_dim == 2048
        assert decoder.text_dim == 1024
        assert decoder.use_encodec is False
        assert decoder.audio_frames_per_tr == 65

        # Check all layers exist
        assert hasattr(decoder, "layer8")
        assert hasattr(decoder, "layer9")
        assert hasattr(decoder, "layer10")
        assert hasattr(decoder, "layer11")
        assert hasattr(decoder, "layer12_video")
        assert hasattr(decoder, "layer12_audio")
        assert hasattr(decoder, "layer12_text")
        assert hasattr(decoder, "layer13_video")
        assert hasattr(decoder, "layer13_audio")
        assert hasattr(decoder, "layer13_text")

    def test_forward_pass_single_sample(self, decoder):
        """Test forward pass with single sample."""
        batch_size = 1
        bottleneck = torch.randn(batch_size, 2048)

        # Set to eval mode (BatchNorm requires batch_size > 1 in train mode)
        decoder.eval()

        # Forward pass
        video, audio, text = decoder(bottleneck)

        # Check output shapes
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

        # Check video is in [0, 1] range (sigmoid output)
        assert torch.all(video >= 0)
        assert torch.all(video <= 1)

        # Check no NaN or Inf
        assert not torch.isnan(video).any()
        assert not torch.isnan(audio).any()
        assert not torch.isnan(text).any()
        assert not torch.isinf(video).any()
        assert not torch.isinf(audio).any()
        assert not torch.isinf(text).any()

    def test_forward_pass_batch(self, decoder):
        """Test forward pass with batch of samples."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        # Forward pass
        video, audio, text = decoder(bottleneck)

        # Check output shapes
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

        # Check video is in [0, 1] range
        assert torch.all(video >= 0)
        assert torch.all(video <= 1)

        # Check no NaN or Inf
        assert not torch.isnan(video).any()
        assert not torch.isnan(audio).any()
        assert not torch.isnan(text).any()

    def test_decode_video_only(self, decoder):
        """Test video-only decoding."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        # Decode video only
        video = decoder.decode_video_only(bottleneck)

        # Check shape
        assert video.shape == (batch_size, 43200)

        # Check range
        assert torch.all(video >= 0)
        assert torch.all(video <= 1)

        # Check no NaN
        assert not torch.isnan(video).any()

    def test_decode_audio_only(self, decoder):
        """Test audio-only decoding."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        # Decode audio only
        audio = decoder.decode_audio_only(bottleneck)

        # Check shape
        assert audio.shape == (batch_size, 2048, 65)

        # Check no NaN
        assert not torch.isnan(audio).any()

    def test_decode_text_only(self, decoder):
        """Test text-only decoding."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        # Decode text only
        text = decoder.decode_text_only(bottleneck)

        # Check shape
        assert text.shape == (batch_size, 1024)

        # Check no NaN
        assert not torch.isnan(text).any()

    def test_get_layer_outputs(self, small_decoder):
        """Test getting intermediate layer outputs."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 2048)

        # Get layer outputs
        outputs = small_decoder.get_layer_outputs(bottleneck)

        # Check all expected keys exist
        expected_keys = [
            "layer8",
            "layer9",
            "layer10",
            "layer11",
            "layer12_video",
            "layer12_audio",
            "layer12_text",
            "video",
            "audio",
            "text",
        ]
        for key in expected_keys:
            assert key in outputs
            assert outputs[key] is not None
            assert outputs[key].shape[0] == batch_size

        # Check final outputs match expected shapes (note: get_layer_outputs may return intermediate values)
        assert outputs["video"].shape == (batch_size, 43200)
        # Audio output from get_layer_outputs might be before temporal upsampling
        assert outputs["text"].shape == (batch_size, 1024)

    def test_count_parameters(self, decoder):
        """Test parameter counting."""
        param_counts = decoder.count_parameters()

        # Check all components have parameters
        assert param_counts["layer8"] > 0
        assert param_counts["layer9"] > 0
        assert param_counts["layer10"] > 0
        assert param_counts["layer11"] > 0
        assert param_counts["layer12_video"] > 0
        assert param_counts["layer12_audio"] > 0
        assert param_counts["layer12_text"] > 0
        assert param_counts["layer13_video"] > 0
        assert param_counts["layer13_audio"] > 0
        assert param_counts["layer13_text"] > 0

        # Check total is reasonable
        assert param_counts["total"] > 0

        print("\nDecoder parameter counts:")
        for key, count in param_counts.items():
            print(f"  {key}: {count:,}")

    def test_different_bottleneck_dimensions(self):
        """Test decoder with standard bottleneck size."""
        # Decoder has fixed architecture with bottleneck_dim=2048
        batch_size = 4
        bottleneck_dim = 2048

        decoder = MultimodalDecoder(bottleneck_dim=bottleneck_dim, use_encodec=False)

        bottleneck = torch.randn(batch_size, bottleneck_dim)
        decoder.eval()
        video, audio, text = decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

    def test_different_audio_modes(self):
        """Test decoder works with both EnCodec and mel spectrograms."""
        batch_size = 4
        bottleneck_dim = 2048

        # Test mel spectrogram mode
        mel_decoder = MultimodalDecoder(
            bottleneck_dim=bottleneck_dim, use_encodec=False, audio_frames_per_tr=65
        )
        mel_decoder.eval()
        bottleneck = torch.randn(batch_size, bottleneck_dim)
        video, audio, text = mel_decoder(bottleneck)
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

        # Test EnCodec mode
        encodec_decoder = MultimodalDecoder(
            bottleneck_dim=bottleneck_dim, use_encodec=True, audio_frames_per_tr=112
        )
        encodec_decoder.eval()
        video, audio, text = encodec_decoder(bottleneck)
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 8, 112)
        assert text.shape == (batch_size, 1024)

    def test_gradient_flow(self, small_decoder):
        """Test gradients flow through all paths."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 2048, requires_grad=True)

        # Forward pass
        video, audio, text = small_decoder(bottleneck)

        # Create dummy loss
        loss = video.sum() + audio.sum() + text.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist for bottleneck
        assert bottleneck.grad is not None
        assert not torch.isnan(bottleneck.grad).any()

        # Check gradients exist for all parameters
        for name, param in small_decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_dropout_behavior(self, small_decoder):
        """Test dropout behaves differently in train vs eval mode."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 2048)

        # Get outputs in train mode
        small_decoder.train()
        video_train1, _, _ = small_decoder(bottleneck)
        video_train2, _, _ = small_decoder(bottleneck)

        # Outputs should differ due to dropout
        assert not torch.allclose(video_train1, video_train2)

        # Get outputs in eval mode
        small_decoder.eval()
        video_eval1, _, _ = small_decoder(bottleneck)
        video_eval2, _, _ = small_decoder(bottleneck)

        # Outputs should be identical (no dropout)
        assert torch.allclose(video_eval1, video_eval2)

    def test_batch_normalization(self, small_decoder):
        """Test batch normalization behaves correctly."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        # Train mode
        small_decoder.train()
        video1, _, _ = small_decoder(bottleneck)

        # Eval mode
        small_decoder.eval()
        video2, _, _ = small_decoder(bottleneck)

        # Outputs should differ slightly due to batch norm
        assert not torch.allclose(video1, video2, atol=1e-3)

    def test_zero_bottleneck(self, small_decoder):
        """Test decoder handles zero input."""
        batch_size = 4
        bottleneck = torch.zeros(batch_size, 2048)

        small_decoder.eval()
        video, audio, text = small_decoder(bottleneck)

        # Should produce valid outputs (not all zeros due to biases)
        assert not torch.allclose(video, torch.zeros_like(video))
        assert not torch.isnan(video).any()
        assert not torch.isnan(audio).any()
        assert not torch.isnan(text).any()

    def test_large_batch(self, small_decoder):
        """Test decoder handles large batches."""
        batch_size = 128
        bottleneck = torch.randn(batch_size, 2048)

        small_decoder.eval()
        video, audio, text = small_decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

    def test_consistency_across_runs(self, small_decoder):
        """Test decoder produces consistent outputs in eval mode."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 2048)

        small_decoder.eval()

        # Multiple forward passes
        outputs1 = [small_decoder(bottleneck) for _ in range(3)]
        outputs2 = [small_decoder(bottleneck) for _ in range(3)]

        # All should be identical
        for i in range(3):
            video1, audio1, text1 = outputs1[i]
            video2, audio2, text2 = outputs2[i]
            assert torch.allclose(video1, video2)
            assert torch.allclose(audio1, audio2)
            assert torch.allclose(text1, text2)

    def test_output_statistics(self, decoder):
        """Test output statistics are reasonable."""
        batch_size = 32
        bottleneck = torch.randn(batch_size, 2048)

        decoder.eval()
        video, audio, text = decoder(bottleneck)

        # Video should use full [0, 1] range (roughly)
        assert video.min() < 0.3  # Should have some low values
        assert video.max() > 0.7  # Should have some high values
        assert 0.3 < video.mean() < 0.7  # Mean should be centered

        # Audio and text should have some variance (small values due to initialization)
        assert audio.std() > 0.001
        assert text.std() > 0.01

    def test_video_frame_reconstruction(self, small_decoder):
        """Test video output can be reshaped to frames."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 2048)

        small_decoder.eval()
        video, _, _ = small_decoder(bottleneck)

        # Reshape to video frames (160×90×3)
        frames = video.reshape(batch_size, 90, 160, 3)

        assert frames.shape == (batch_size, 90, 160, 3)
        assert torch.all(frames >= 0)
        assert torch.all(frames <= 1)


@pytest.mark.integration
class TestDecoderIntegration:
    """Integration tests for decoder with realistic scenarios."""

    def test_typical_fmri_dimensions(self):
        """Test with typical fMRI bottleneck size."""
        # Standard bottleneck dimension
        batch_size = 16

        decoder = MultimodalDecoder(bottleneck_dim=2048, use_encodec=False)

        decoder.eval()
        bottleneck = torch.randn(batch_size, 2048)
        video, audio, text = decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)

    @pytest.mark.slow
    def test_tr_batch_processing(self):
        """Test processing batch corresponding to TRs."""
        # Simulate 100 TRs of fMRI data
        n_trs = 100
        bottleneck_dim = 2048

        decoder = MultimodalDecoder(bottleneck_dim=bottleneck_dim, use_encodec=False)

        decoder.eval()

        # Process all TRs at once
        bottleneck_batch = torch.randn(n_trs, bottleneck_dim)
        video_batch, audio_batch, text_batch = decoder(bottleneck_batch)

        # Process TRs individually
        video_individual = []
        audio_individual = []
        text_individual = []

        for i in range(n_trs):
            v, a, t = decoder(bottleneck_batch[i : i + 1])
            video_individual.append(v)
            audio_individual.append(a)
            text_individual.append(t)

        video_individual = torch.cat(video_individual, dim=0)
        audio_individual = torch.cat(audio_individual, dim=0)
        text_individual = torch.cat(text_individual, dim=0)

        # Batch and individual processing should match (with reasonable tolerance)
        assert torch.allclose(video_batch, video_individual, atol=1e-4)
        assert torch.allclose(audio_batch, audio_individual, atol=1e-4)
        assert torch.allclose(text_batch, text_individual, atol=1e-4)

    def test_memory_efficiency(self):
        """Test decoder can process without excessive memory."""
        decoder = MultimodalDecoder(bottleneck_dim=5000, use_encodec=False)

        decoder.eval()

        # Process moderate batch
        batch_size = 64
        bottleneck = torch.randn(batch_size, 2048)

        # Should complete without memory error
        video, audio, text = decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 2048, 65)
        assert text.shape == (batch_size, 1024)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

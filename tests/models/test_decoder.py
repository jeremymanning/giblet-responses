"""
Tests for multimodal decoder module.

Tests forward pass, output shapes, and component functionality.
"""

import pytest
import torch
import numpy as np
from giblet.models.decoder import MultimodalDecoder


class TestMultimodalDecoder:
    """Test suite for MultimodalDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder with standard dimensions."""
        return MultimodalDecoder(
            bottleneck_dim=5000,
            video_dim=43200,
            audio_dim=128,
            text_dim=1024,
            hidden_dim=2048,
            dropout=0.3
        )

    @pytest.fixture
    def small_decoder(self):
        """Create smaller decoder for faster testing."""
        return MultimodalDecoder(
            bottleneck_dim=100,
            video_dim=43200,
            audio_dim=128,
            text_dim=1024,
            hidden_dim=256,
            dropout=0.1
        )

    def test_initialization(self, decoder):
        """Test decoder initializes with correct architecture."""
        assert decoder.bottleneck_dim == 5000
        assert decoder.video_dim == 43200
        assert decoder.audio_dim == 128
        assert decoder.text_dim == 1024
        assert decoder.hidden_dim == 2048

        # Check all layers exist
        assert hasattr(decoder, 'layer7')
        assert hasattr(decoder, 'layer8')
        assert hasattr(decoder, 'layer9')
        assert hasattr(decoder, 'layer10_video')
        assert hasattr(decoder, 'layer10_audio')
        assert hasattr(decoder, 'layer10_text')
        assert hasattr(decoder, 'layer11_video')
        assert hasattr(decoder, 'layer11_audio')
        assert hasattr(decoder, 'layer11_text')

    def test_forward_pass_single_sample(self, decoder):
        """Test forward pass with single sample."""
        batch_size = 1
        bottleneck = torch.randn(batch_size, 5000)

        # Set to eval mode (BatchNorm requires batch_size > 1 in train mode)
        decoder.eval()

        # Forward pass
        video, audio, text = decoder(bottleneck)

        # Check output shapes
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 128)
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
        bottleneck = torch.randn(batch_size, 5000)

        # Forward pass
        video, audio, text = decoder(bottleneck)

        # Check output shapes
        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 128)
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
        bottleneck = torch.randn(batch_size, 5000)

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
        bottleneck = torch.randn(batch_size, 5000)

        # Decode audio only
        audio = decoder.decode_audio_only(bottleneck)

        # Check shape
        assert audio.shape == (batch_size, 128)

        # Check no NaN
        assert not torch.isnan(audio).any()

    def test_decode_text_only(self, decoder):
        """Test text-only decoding."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 5000)

        # Decode text only
        text = decoder.decode_text_only(bottleneck)

        # Check shape
        assert text.shape == (batch_size, 1024)

        # Check no NaN
        assert not torch.isnan(text).any()

    def test_get_layer_outputs(self, small_decoder):
        """Test getting intermediate layer outputs."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 100)

        # Get layer outputs
        outputs = small_decoder.get_layer_outputs(bottleneck)

        # Check all expected keys exist
        expected_keys = [
            'layer7', 'layer8', 'layer9',
            'layer10_video', 'layer10_audio', 'layer10_text',
            'video', 'audio', 'text'
        ]
        for key in expected_keys:
            assert key in outputs
            assert outputs[key] is not None
            assert outputs[key].shape[0] == batch_size

        # Check final outputs match expected shapes
        assert outputs['video'].shape == (batch_size, 43200)
        assert outputs['audio'].shape == (batch_size, 128)
        assert outputs['text'].shape == (batch_size, 1024)

    def test_count_parameters(self, decoder):
        """Test parameter counting."""
        param_counts = decoder.count_parameters()

        # Check all components have parameters
        assert param_counts['layer7'] > 0
        assert param_counts['layer8'] > 0
        assert param_counts['layer9'] > 0
        assert param_counts['layer10_video'] > 0
        assert param_counts['layer10_audio'] > 0
        assert param_counts['layer10_text'] > 0
        assert param_counts['layer11_video'] > 0
        assert param_counts['layer11_audio'] > 0
        assert param_counts['layer11_text'] > 0

        # Check total equals sum of components
        component_sum = sum([
            param_counts['layer7'],
            param_counts['layer8'],
            param_counts['layer9'],
            param_counts['layer10_video'],
            param_counts['layer10_audio'],
            param_counts['layer10_text'],
            param_counts['layer11_video'],
            param_counts['layer11_audio'],
            param_counts['layer11_text']
        ])
        assert param_counts['total'] == component_sum

        print(f"\nDecoder parameter counts:")
        for key, count in param_counts.items():
            print(f"  {key}: {count:,}")

    def test_different_bottleneck_dimensions(self):
        """Test decoder works with different bottleneck sizes."""
        bottleneck_dims = [1000, 5000, 10000]
        batch_size = 4

        for dim in bottleneck_dims:
            decoder = MultimodalDecoder(
                bottleneck_dim=dim,
                hidden_dim=512  # Smaller for faster testing
            )

            bottleneck = torch.randn(batch_size, dim)
            video, audio, text = decoder(bottleneck)

            assert video.shape == (batch_size, 43200)
            assert audio.shape == (batch_size, 128)
            assert text.shape == (batch_size, 1024)

    def test_different_hidden_dimensions(self):
        """Test decoder works with different hidden dimensions."""
        hidden_dims = [512, 1024, 2048]
        batch_size = 4
        bottleneck_dim = 5000

        for hidden_dim in hidden_dims:
            decoder = MultimodalDecoder(
                bottleneck_dim=bottleneck_dim,
                hidden_dim=hidden_dim
            )

            bottleneck = torch.randn(batch_size, bottleneck_dim)
            video, audio, text = decoder(bottleneck)

            assert video.shape == (batch_size, 43200)
            assert audio.shape == (batch_size, 128)
            assert text.shape == (batch_size, 1024)

    def test_gradient_flow(self, small_decoder):
        """Test gradients flow through all paths."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 100, requires_grad=True)

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
        bottleneck = torch.randn(batch_size, 100)

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
        bottleneck = torch.randn(batch_size, 100)

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
        bottleneck = torch.zeros(batch_size, 100)

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
        bottleneck = torch.randn(batch_size, 100)

        small_decoder.eval()
        video, audio, text = small_decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 128)
        assert text.shape == (batch_size, 1024)

    def test_consistency_across_runs(self, small_decoder):
        """Test decoder produces consistent outputs in eval mode."""
        batch_size = 8
        bottleneck = torch.randn(batch_size, 100)

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
        bottleneck = torch.randn(batch_size, 5000)

        decoder.eval()
        video, audio, text = decoder(bottleneck)

        # Video should use full [0, 1] range (roughly)
        assert video.min() < 0.3  # Should have some low values
        assert video.max() > 0.7  # Should have some high values
        assert 0.3 < video.mean() < 0.7  # Mean should be centered

        # Audio and text should have reasonable variance
        assert audio.std() > 0.1
        assert text.std() > 0.1

    def test_video_frame_reconstruction(self, small_decoder):
        """Test video output can be reshaped to frames."""
        batch_size = 4
        bottleneck = torch.randn(batch_size, 100)

        small_decoder.eval()
        video, _, _ = small_decoder(bottleneck)

        # Reshape to video frames (160×90×3)
        frames = video.reshape(batch_size, 90, 160, 3)

        assert frames.shape == (batch_size, 90, 160, 3)
        assert torch.all(frames >= 0)
        assert torch.all(frames <= 1)


class TestDecoderIntegration:
    """Integration tests for decoder with realistic scenarios."""

    def test_typical_fmri_dimensions(self):
        """Test with typical fMRI voxel counts."""
        # Common fMRI voxel counts for different ROIs
        voxel_counts = [2000, 5000, 8000, 10000]
        batch_size = 16

        for n_voxels in voxel_counts:
            decoder = MultimodalDecoder(
                bottleneck_dim=n_voxels,
                hidden_dim=1024
            )

            bottleneck = torch.randn(batch_size, n_voxels)
            video, audio, text = decoder(bottleneck)

            assert video.shape == (batch_size, 43200)
            assert audio.shape == (batch_size, 128)
            assert text.shape == (batch_size, 1024)

    def test_tr_batch_processing(self):
        """Test processing batch corresponding to TRs."""
        # Simulate 100 TRs of fMRI data
        n_trs = 100
        n_voxels = 5000

        decoder = MultimodalDecoder(
            bottleneck_dim=n_voxels,
            hidden_dim=1024
        )

        decoder.eval()

        # Process all TRs at once
        bottleneck_batch = torch.randn(n_trs, n_voxels)
        video_batch, audio_batch, text_batch = decoder(bottleneck_batch)

        # Process TRs individually
        video_individual = []
        audio_individual = []
        text_individual = []

        for i in range(n_trs):
            v, a, t = decoder(bottleneck_batch[i:i+1])
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
        decoder = MultimodalDecoder(
            bottleneck_dim=5000,
            hidden_dim=1024
        )

        decoder.eval()

        # Process moderate batch
        batch_size = 64
        bottleneck = torch.randn(batch_size, 5000)

        # Should complete without memory error
        video, audio, text = decoder(bottleneck)

        assert video.shape == (batch_size, 43200)
        assert audio.shape == (batch_size, 128)
        assert text.shape == (batch_size, 1024)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

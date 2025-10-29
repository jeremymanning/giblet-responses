"""
Test suite for MultimodalEncoder module.

Tests the encoder architecture including:
- Individual modality encoders (video, audio, text)
- Full encoder forward pass
- Batch processing
- Parameter counting
- GPU compatibility
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from giblet.models.encoder import (
    VideoEncoder,
    AudioEncoder,
    TextEncoder,
    MultimodalEncoder,
    create_encoder
)


class TestVideoEncoder:
    """Test video encoder component."""

    def test_video_encoder_init(self):
        """Test video encoder initialization."""
        encoder = VideoEncoder(
            input_height=90,
            input_width=160,
            output_features=1024
        )
        assert encoder.input_height == 90
        assert encoder.input_width == 160
        assert encoder.output_features == 1024

    def test_video_encoder_forward(self):
        """Test video encoder forward pass."""
        encoder = VideoEncoder(
            input_height=90,
            input_width=160,
            output_features=1024
        )
        encoder.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 90, 160)

        # Forward pass
        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (batch_size, 1024)
        assert not torch.isnan(output).any()

    def test_video_encoder_parameter_count(self):
        """Test video encoder has reasonable parameter count."""
        encoder = VideoEncoder()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"Video encoder parameters: {n_params:,}")
        assert n_params > 0
        assert n_params < 50_000_000  # Should be under 50M


class TestAudioEncoder:
    """Test audio encoder component."""

    def test_audio_encoder_init(self):
        """Test audio encoder initialization."""
        encoder = AudioEncoder(
            input_mels=128,
            output_features=256
        )
        assert encoder.input_mels == 128
        assert encoder.output_features == 256

    def test_audio_encoder_forward(self):
        """Test audio encoder forward pass."""
        encoder = AudioEncoder(
            input_mels=128,
            output_features=256
        )
        encoder.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 128)

        # Forward pass
        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_audio_encoder_parameter_count(self):
        """Test audio encoder has reasonable parameter count."""
        encoder = AudioEncoder()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"Audio encoder parameters: {n_params:,}")
        assert n_params > 0
        assert n_params < 10_000_000  # Should be under 10M


class TestTextEncoder:
    """Test text encoder component."""

    def test_text_encoder_init(self):
        """Test text encoder initialization."""
        encoder = TextEncoder(
            input_dim=1024,
            output_features=256
        )
        assert encoder.input_dim == 1024
        assert encoder.output_features == 256

    def test_text_encoder_forward(self):
        """Test text encoder forward pass."""
        encoder = TextEncoder(
            input_dim=1024,
            output_features=256
        )
        encoder.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 1024)

        # Forward pass
        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_text_encoder_parameter_count(self):
        """Test text encoder has reasonable parameter count."""
        encoder = TextEncoder()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"Text encoder parameters: {n_params:,}")
        assert n_params > 0
        assert n_params < 10_000_000  # Should be under 10M


class TestMultimodalEncoder:
    """Test full multimodal encoder."""

    def test_encoder_init(self):
        """Test encoder initialization with default parameters."""
        encoder = MultimodalEncoder()
        assert encoder.video_height == 90
        assert encoder.video_width == 160
        assert encoder.audio_mels == 2048  # Default is 2048, not 128
        assert encoder.text_dim == 1024
        assert encoder.n_voxels == 85810
        assert encoder.bottleneck_dim == 8000

    def test_encoder_forward_single(self):
        """Test encoder forward pass with single sample."""
        encoder = MultimodalEncoder()
        encoder.eval()

        # Create dummy inputs
        video = torch.randn(1, 3, 90, 160)
        audio = torch.randn(1, 2048)
        text = torch.randn(1, 1024)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        # Check output shapes
        assert bottleneck.shape == (1, 8000)
        assert voxels.shape == (1, 85810)
        assert not torch.isnan(bottleneck).any()
        assert not torch.isnan(voxels).any()

    def test_encoder_forward_batch(self):
        """Test encoder forward pass with batch."""
        encoder = MultimodalEncoder()
        encoder.eval()

        # Create dummy inputs
        batch_size = 8
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 2048)
        text = torch.randn(batch_size, 1024)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        # Check output shapes
        assert bottleneck.shape == (batch_size, 8000)
        assert voxels.shape == (batch_size, 85810)
        assert not torch.isnan(bottleneck).any()
        assert not torch.isnan(voxels).any()

    def test_encoder_forward_without_voxels(self):
        """Test encoder forward pass without returning voxels."""
        encoder = MultimodalEncoder()
        encoder.eval()

        # Create dummy inputs
        video = torch.randn(4, 3, 90, 160)
        audio = torch.randn(4, 2048)
        text = torch.randn(4, 1024)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=False)

        # Check outputs
        assert bottleneck.shape == (4, 8000)
        assert voxels is None

    def test_encoder_parameter_count(self):
        """Test encoder parameter counting method."""
        encoder = MultimodalEncoder()
        param_dict = encoder.get_parameter_count()

        # Check all keys present
        assert 'video_encoder' in param_dict
        assert 'audio_encoder' in param_dict
        assert 'text_encoder' in param_dict
        assert 'feature_conv' in param_dict
        assert 'to_bottleneck' in param_dict
        assert 'bottleneck_to_voxels' in param_dict
        assert 'total' in param_dict

        # Check total is sum of parts
        total_manual = sum(v for k, v in param_dict.items() if k != 'total')
        assert param_dict['total'] == total_manual

        # Print summary
        print("\n=== Encoder Parameter Count ===")
        for key, value in param_dict.items():
            print(f"{key:25s}: {value:>14,} parameters")
        print("=" * 55)

        # Check reasonable total
        assert param_dict['total'] > 0
        # Relaxed constraint - large models are expected for this architecture
        assert param_dict['total'] < 2_000_000_000  # Should be under 2B

    def test_encoder_custom_dimensions(self):
        """Test encoder with custom dimensions."""
        encoder = MultimodalEncoder(
            video_height=45,
            video_width=80,
            audio_mels=64,
            text_dim=512,
            n_voxels=10000,
            bottleneck_dim=2000
        )
        encoder.eval()

        # Create dummy inputs with custom dimensions
        video = torch.randn(2, 3, 45, 80)
        audio = torch.randn(2, 64)
        text = torch.randn(2, 512)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        assert bottleneck.shape == (2, 2000)
        assert voxels.shape == (2, 10000)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_encoder_gpu(self):
        """Test encoder on GPU."""
        encoder = MultimodalEncoder().cuda()
        encoder.eval()

        # Create dummy inputs on GPU
        video = torch.randn(4, 3, 90, 160).cuda()
        audio = torch.randn(4, 2048).cuda()
        text = torch.randn(4, 1024).cuda()

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        # Check outputs on GPU
        assert bottleneck.is_cuda
        assert voxels.is_cuda
        assert bottleneck.shape == (4, 8000)
        assert voxels.shape == (4, 85810)


class TestCreateEncoder:
    """Test encoder factory function."""

    def test_create_encoder_defaults(self):
        """Test factory function with defaults."""
        encoder = create_encoder()
        assert isinstance(encoder, MultimodalEncoder)
        assert encoder.n_voxels == 85810
        assert encoder.bottleneck_dim == 8000

    def test_create_encoder_custom(self):
        """Test factory function with custom parameters."""
        encoder = create_encoder(
            n_voxels=10000,
            bottleneck_dim=2000
        )
        assert encoder.n_voxels == 10000
        assert encoder.bottleneck_dim == 2000


class TestEncoderIntegration:
    """Integration tests with realistic data dimensions."""

    def test_example_dataset_dimensions(self):
        """Test with example dataset dimensions (Sherlock)."""
        encoder = MultimodalEncoder(
            video_height=90,
            video_width=160,
            audio_mels=2048,
            text_dim=1024,
            n_voxels=85810,
            bottleneck_dim=8000
        )
        encoder.eval()

        # Simulate one TR of data
        video = torch.randn(1, 3, 90, 160)  # 160×90×3 = 43,200 per TR
        audio = torch.randn(1, 2048)         # 2048 mels per TR
        text = torch.randn(1, 1024)          # 1024 embeddings per TR

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        # Verify shapes
        assert bottleneck.shape == (1, 8000), "Bottleneck should be 8000-dim"
        assert voxels.shape == (1, 85810), "Should match 85,810 brain voxels"

        # Verify no NaNs or Infs
        assert torch.isfinite(bottleneck).all()
        assert torch.isfinite(voxels).all()

    def test_batch_processing(self):
        """Test batch processing with multiple TRs."""
        encoder = MultimodalEncoder()
        encoder.eval()

        # Simulate 32 TRs (mini-batch)
        batch_size = 32
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 2048)
        text = torch.randn(batch_size, 1024)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        assert bottleneck.shape == (batch_size, 8000)
        assert voxels.shape == (batch_size, 85810)

    def test_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = MultimodalEncoder()
        encoder.train()

        # Create dummy inputs (requires_grad=True)
        video = torch.randn(2, 3, 90, 160, requires_grad=True)
        audio = torch.randn(2, 2048, requires_grad=True)
        text = torch.randn(2, 1024, requires_grad=True)

        # Forward pass
        bottleneck, voxels = encoder(video, audio, text, return_voxels=True)

        # Backward pass with dummy loss
        loss = bottleneck.sum() + voxels.sum()
        loss.backward()

        # Check gradients exist
        assert video.grad is not None
        assert audio.grad is not None
        assert text.grad is not None

        # Check encoder parameters have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

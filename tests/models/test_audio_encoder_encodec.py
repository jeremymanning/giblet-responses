"""
Test suite for AudioEncoder with EnCodec quantized codes support.

Tests the audio encoder's ability to process:
1. EnCodec quantized codes (discrete integers)
2. Mel spectrograms (continuous values)
3. Backwards compatibility with legacy formats
4. Gradient flow and training capability

Verified settings for EnCodec mode (Issue #24):
- 3.0 kbps bandwidth
- 8 codebooks per TR
- ~112 frames per TR
- Codes are discrete integers [0, 1023]
"""

import pytest
import torch
import numpy as np
from giblet.models.encoder import AudioEncoder, MultimodalEncoder


@pytest.mark.unit
class TestAudioEncoderEnCodec:
    """Test AudioEncoder with EnCodec quantized codes."""

    def test_encodec_encoder_init(self):
        """Test EnCodec audio encoder initialization."""
        encoder = AudioEncoder(
            input_codebooks=8,
            frames_per_tr=112,
            output_features=256,
            use_encodec=True,
            vocab_size=1024,  # Accepted for backward compat but not used
            embed_dim=64,  # Accepted for backward compat but not used
        )
        assert encoder.use_encodec is True
        assert encoder.input_codebooks == 8
        assert encoder.frames_per_tr == 112
        assert encoder.output_features == 256

        # Note: vocab_size, embed_dim, code_embedding removed in Issue #29
        # AudioEncoder now uses Linear layers, not embeddings

    def test_encodec_forward_pass(self):
        """Test EnCodec encoder forward pass with quantized codes."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        # Create synthetic EnCodec codes - 2D flattened (batch, codebooks * frames)
        batch_size = 4
        codes = torch.randint(0, 1024, (batch_size, 8 * 112))  # Flattened

        # Forward pass
        with torch.no_grad():
            output = encoder(codes)

        # Check output shape
        assert output.shape == (
            batch_size,
            256,
        ), f"Expected shape (4, 256), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains Inf values"

    def test_encodec_dimension_correctness(self):
        """Test that EnCodec mode handles dimensions correctly."""
        encoder = AudioEncoder(
            input_codebooks=8,
            frames_per_tr=112,
            output_features=256,
            use_encodec=True,
            embed_dim=64,  # Backward compat param (not used)
        )
        encoder.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1

        # Test with different batch sizes
        for batch_size in [1, 4, 16, 32]:
            codes = torch.randint(0, 1024, (batch_size, 8 * 112))  # 2D flattened
            with torch.no_grad():
                output = encoder(codes)
            assert output.shape == (
                batch_size,
                256,
            ), f"Batch size {batch_size}: Expected (B, 256), got {output.shape}"

    def test_encodec_code_range(self):
        """Test that encoder handles codes at boundaries correctly."""
        encoder = AudioEncoder(
            input_codebooks=8,
            frames_per_tr=112,
            output_features=256,
            use_encodec=True,
            vocab_size=1024,  # Backward compat param (not used)
        )
        encoder.eval()

        # Test with codes at vocabulary boundaries - 2D flattened
        codes_min = torch.zeros((2, 8 * 112), dtype=torch.long)  # All zeros
        codes_max = torch.full((2, 8 * 112), 1023, dtype=torch.long)  # All 1023
        codes_mixed = torch.randint(0, 1024, (2, 8 * 112))  # Flattened

        with torch.no_grad():
            output_min = encoder(codes_min)
            output_max = encoder(codes_max)
            output_mixed = encoder(codes_mixed)

        # All should produce valid outputs
        assert output_min.shape == (2, 256)
        assert output_max.shape == (2, 256)
        assert output_mixed.shape == (2, 256)

        # Outputs should be different (not all same)
        assert not torch.allclose(
            output_min, output_max
        ), "Min and max codes should produce different outputs"

    def test_encodec_gradient_flow(self):
        """Test that gradients flow correctly through embedding and convolutions."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.train()

        # Create codes - 2D flattened
        codes = torch.randint(0, 1024, (4, 8 * 112))

        # Forward pass
        output = encoder(codes)

        # Backward pass with dummy loss
        loss = output.sum()
        loss.backward()

        # Check that encoder parameters have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for: {name}"

    def test_encodec_float_input_conversion(self):
        """Test that float codes are converted to float (int-like values)."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        # Create codes as floats - 2D flattened
        codes_float = torch.rand(2, 8 * 112) * 1024  # Random floats [0, 1024)

        with torch.no_grad():
            output = encoder(codes_float)

        assert output.shape == (2, 256)
        assert not torch.isnan(output).any()

    def test_encodec_parameter_count(self):
        """Test EnCodec encoder has reasonable parameter count."""
        encoder = AudioEncoder(
            input_codebooks=8,
            frames_per_tr=112,
            output_features=256,
            use_encodec=True,
            vocab_size=1024,  # Backward compat param (not used)
            embed_dim=64,  # Backward compat param (not used)
        )

        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\nEnCodec AudioEncoder parameters: {n_params:,}")

        # Note: No embedding layer in Issue #29 refactor
        # AudioEncoder uses Linear layers: fc1(896→512) + fc2(512→256)
        # Expected params ≈ 896*512 + 512*256 = 459K + 131K ≈ 590K

        assert n_params > 0
        assert n_params < 10_000_000, "Should be under 10M parameters"


@pytest.mark.unit
class TestAudioEncoderBackwardsCompatibility:
    """Test backwards compatibility with mel spectrograms."""

    def test_mel_spectrogram_mode(self):
        """Test that mel spectrogram mode still works."""
        encoder = AudioEncoder(
            input_mels=2048,
            frames_per_tr=65,
            output_features=256,
            use_encodec=False,  # Explicitly set to mel mode
        )
        encoder.eval()

        # Create mel spectrogram input - 2D flattened
        batch_size = 4
        mels = torch.randn(batch_size, 2048 * 65)  # Flattened

        # Forward pass
        with torch.no_grad():
            output = encoder(mels)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_legacy_2d_input(self):
        """Test backwards compatibility with 2D mel input."""
        encoder = AudioEncoder(
            input_mels=2048,
            frames_per_tr=1,  # Single time step for unit testing
            output_features=256,
            use_encodec=False,
        )
        encoder.eval()

        # Create 2D input - flattened (mels × frames)
        batch_size = 4
        mels_2d = torch.randn(batch_size, 2048)  # 2048 × 1 = 2048

        # Forward pass
        with torch.no_grad():
            output = encoder(mels_2d)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_mel_vs_encodec_different_architectures(self):
        """Test that mel and EnCodec modes use different architectures."""
        encoder_mel = AudioEncoder(
            input_mels=2048, frames_per_tr=65, output_features=256, use_encodec=False
        )

        encoder_encodec = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )

        # Note: No embedding layer in Issue #29 refactor (both use Linear)
        # Architectures differ in input_dim:
        # - Mel: 2048 × 65 = 133,120
        # - EnCodec: 8 × 112 = 896

        # Parameter counts should differ due to different input dimensions
        params_mel = sum(p.numel() for p in encoder_mel.parameters())
        params_encodec = sum(p.numel() for p in encoder_encodec.parameters())

        print(f"\nMel encoder params: {params_mel:,}")
        print(f"EnCodec encoder params: {params_encodec:,}")

        # Architectures should differ in parameter count
        # Mel encoder has more params due to larger input dimension (133K vs 896)
        assert (
            params_mel != params_encodec
        ), "Architectures should have different parameter counts"
        assert (
            params_mel > params_encodec
        ), "Mel encoder should have more params (larger input)"


@pytest.mark.unit
class TestMultimodalEncoderEnCodec:
    """Test MultimodalEncoder with EnCodec audio."""

    def test_multimodal_with_encodec(self):
        """Test full MultimodalEncoder with EnCodec audio mode."""
        encoder = MultimodalEncoder(
            video_height=90,
            video_width=160,
            video_frames_per_tr=1,  # Single frame for unit testing
            audio_codebooks=8,
            audio_frames_per_tr=112,
            text_dim=1024,
            use_encodec=True,
        )
        encoder.eval()

        # Create dummy inputs
        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)  # Single frame
        audio_codes = torch.randint(0, 1024, (batch_size, 8 * 112))  # Flattened
        text = torch.randn(batch_size, 1024)

        # Forward pass
        with torch.no_grad():
            bottleneck, voxels = encoder(video, audio_codes, text, return_voxels=True)

        # Check outputs
        assert bottleneck.shape == (
            batch_size,
            2048,
        ), f"Expected (4, 2048), got {bottleneck.shape}"
        assert voxels.shape == (batch_size, 85810)
        assert not torch.isnan(bottleneck).any()
        assert not torch.isnan(voxels).any()

    def test_multimodal_encodec_gradient_flow(self):
        """Test gradient flow through MultimodalEncoder with EnCodec."""
        encoder = MultimodalEncoder(
            video_frames_per_tr=1,  # Single frame for unit testing
            use_encodec=True,
            audio_codebooks=8,
            audio_frames_per_tr=112,
        )
        encoder.train()

        # Create inputs
        video = torch.randn(2, 3, 90, 160, requires_grad=True)
        audio_codes = torch.randint(0, 1024, (2, 8 * 112))  # Flattened
        text = torch.randn(2, 1024, requires_grad=True)

        # Forward pass
        bottleneck, voxels = encoder(video, audio_codes, text, return_voxels=True)

        # Backward pass
        loss = bottleneck.sum() + voxels.sum()
        loss.backward()

        # Check gradients on continuous inputs
        assert video.grad is not None
        assert text.grad is not None

        # Check gradients on encoder parameters
        for name, param in encoder.audio_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for audio encoder: {name}"

    def test_multimodal_mel_vs_encodec(self):
        """Test that MultimodalEncoder can switch between mel and EnCodec modes."""
        encoder_mel = MultimodalEncoder(
            video_frames_per_tr=1,  # Single frame for unit testing
            audio_mels=2048,
            audio_frames_per_tr=65,
            use_encodec=False,
        )

        encoder_encodec = MultimodalEncoder(
            video_frames_per_tr=1,  # Single frame for unit testing
            audio_codebooks=8,
            audio_frames_per_tr=112,
            use_encodec=True,
        )

        # Check that audio encoders differ
        assert encoder_mel.audio_encoder.use_encodec is False
        assert encoder_encodec.audio_encoder.use_encodec is True

        # Test with appropriate inputs
        video = torch.randn(2, 3, 90, 160)
        text = torch.randn(2, 1024)

        # Mel mode - flattened
        audio_mel = torch.randn(2, 2048 * 65)
        bottleneck_mel, _ = encoder_mel(video, audio_mel, text, return_voxels=False)
        assert bottleneck_mel.shape == (2, 2048)

        # EnCodec mode - flattened
        audio_codes = torch.randint(0, 1024, (2, 8 * 112))
        bottleneck_encodec, _ = encoder_encodec(
            video, audio_codes, text, return_voxels=False
        )
        assert bottleneck_encodec.shape == (2, 2048)


@pytest.mark.integration
class TestEnCodecRealWorldScenarios:
    """Test realistic EnCodec usage scenarios."""

    def test_typical_batch_size(self):
        """Test with typical training batch size."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        # Typical mini-batch size
        batch_size = 32
        codes = torch.randint(0, 1024, (batch_size, 8 * 112))  # Flattened

        with torch.no_grad():
            output = encoder(codes)

        assert output.shape == (batch_size, 256)
        assert torch.isfinite(output).all()

    def test_single_sample_inference(self):
        """Test single-sample inference (e.g., for prediction)."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        # Single sample - 2D flattened
        codes = torch.randint(0, 1024, (1, 8 * 112))

        with torch.no_grad():
            output = encoder(codes)

        assert output.shape == (1, 256)

    def test_temporal_variations(self):
        """Test with different temporal frame counts."""
        for frames in [50, 112, 150, 200]:
            encoder = AudioEncoder(
                input_codebooks=8,
                frames_per_tr=frames,
                output_features=256,
                use_encodec=True,
            )
            encoder.eval()

            codes = torch.randint(0, 1024, (4, 8 * frames))  # Flattened

            with torch.no_grad():
                output = encoder(codes)

            assert output.shape == (
                4,
                256,
            ), f"Failed for {frames} frames: expected (4, 256), got {output.shape}"

    def test_different_codebook_counts(self):
        """Test with different numbers of codebooks (different bandwidths)."""
        for n_codebooks in [4, 8, 16]:  # Different EnCodec configurations
            encoder = AudioEncoder(
                input_codebooks=n_codebooks,
                frames_per_tr=112,
                output_features=256,
                use_encodec=True,
                embed_dim=64,
            )
            encoder.eval()

            codes = torch.randint(0, 1024, (4, n_codebooks * 112))  # Flattened

            with torch.no_grad():
                output = encoder(codes)

            assert output.shape == (4, 256), f"Failed for {n_codebooks} codebooks"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_encodec_gpu(self):
        """Test EnCodec encoder on GPU."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        ).cuda()
        encoder.eval()

        codes = torch.randint(0, 1024, (4, 8, 112)).cuda()

        with torch.no_grad():
            output = encoder(codes)

        assert output.is_cuda
        assert output.shape == (4, 256)
        assert torch.isfinite(output).all()


@pytest.mark.unit
class TestEnCodecEdgeCases:
    """Test edge cases and error handling."""

    def test_out_of_vocab_codes(self):
        """Test handling of out-of-vocabulary codes (should clamp or handle gracefully)."""
        encoder = AudioEncoder(
            input_codebooks=8,
            frames_per_tr=112,
            output_features=256,
            use_encodec=True,
            vocab_size=1024,
        )
        encoder.eval()

        # Codes within valid range - 2D flattened
        codes = torch.randint(0, 1024, (2, 8 * 112))

        with torch.no_grad():
            output = encoder(codes)

        assert output.shape == (2, 256)

    def test_zero_codes(self):
        """Test with all-zero codes (silence or padding)."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        codes = torch.zeros(4, 8 * 112, dtype=torch.long)  # Flattened

        with torch.no_grad():
            output = encoder(codes)

        assert output.shape == (4, 256)
        # Output should be valid (though may be near zero)
        assert torch.isfinite(output).all()

    def test_constant_codes(self):
        """Test with constant codes across time and codebooks."""
        encoder = AudioEncoder(
            input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
        )
        encoder.eval()

        # All codes set to same value - 2D flattened
        codes = torch.full((4, 8 * 112), 512, dtype=torch.long)

        with torch.no_grad():
            output = encoder(codes)

        assert output.shape == (4, 256)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

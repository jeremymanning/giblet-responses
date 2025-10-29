"""
Test suite for SherlockAutoencoder module.

Tests the full autoencoder including:
- Forward pass with and without fMRI targets
- Loss computation (reconstruction + fMRI)
- Backward pass and gradient flow
- Checkpointing (save/load)
- Multi-GPU preparation
- Integration with encoder and decoder
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from giblet.models.autoencoder import (
    SherlockAutoencoder,
    create_autoencoder,
    prepare_for_distributed
)


class TestSherlockAutoencoder:
    """Test full autoencoder."""

    def test_autoencoder_init(self):
        """Test autoencoder initialization with default parameters."""
        model = SherlockAutoencoder()
        assert model.video_height == 90
        assert model.video_width == 160
        assert model.audio_mels == 128
        assert model.text_dim == 1024
        assert model.n_voxels == 85810
        assert model.bottleneck_dim == 8000
        assert model.reconstruction_weight == 1.0
        assert model.fmri_weight == 1.0

        # Check encoder and decoder exist
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')

    def test_forward_pass_eval(self):
        """Test forward pass in eval mode (no losses)."""
        model = SherlockAutoencoder()
        model.eval()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)

        with torch.no_grad():
            outputs = model(video, audio, text)

        # Check all outputs present
        assert 'bottleneck' in outputs
        assert 'predicted_fmri' in outputs
        assert 'video_recon' in outputs
        assert 'audio_recon' in outputs
        assert 'text_recon' in outputs

        # Check shapes
        assert outputs['bottleneck'].shape == (batch_size, 8000)
        assert outputs['predicted_fmri'].shape == (batch_size, 85810)
        assert outputs['video_recon'].shape == (batch_size, 43200)  # 160*90*3
        assert outputs['audio_recon'].shape == (batch_size, 128)
        assert outputs['text_recon'].shape == (batch_size, 1024)

        # No losses in eval mode
        assert 'total_loss' not in outputs
        assert 'reconstruction_loss' not in outputs
        assert 'fmri_loss' not in outputs

    def test_forward_pass_train_no_fmri(self):
        """Test forward pass in train mode without fMRI target."""
        model = SherlockAutoencoder()
        model.train()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)

        outputs = model(video, audio, text)

        # Check outputs
        assert 'bottleneck' in outputs
        assert 'predicted_fmri' in outputs
        assert 'video_recon' in outputs
        assert 'audio_recon' in outputs
        assert 'text_recon' in outputs

        # Check losses (only reconstruction, no fMRI)
        assert 'reconstruction_loss' in outputs
        assert 'video_loss' in outputs
        assert 'audio_loss' in outputs
        assert 'text_loss' in outputs
        assert 'total_loss' in outputs
        assert 'fmri_loss' not in outputs

        # Loss should be scalar
        assert outputs['reconstruction_loss'].dim() == 0
        assert outputs['total_loss'].dim() == 0

        # Total loss should equal reconstruction loss (no fMRI)
        assert torch.allclose(
            outputs['total_loss'],
            outputs['reconstruction_loss']
        )

    def test_forward_pass_train_with_fmri(self):
        """Test forward pass in train mode with fMRI target."""
        model = SherlockAutoencoder()
        model.train()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)
        fmri_target = torch.randn(batch_size, 85810)

        outputs = model(video, audio, text, fmri_target=fmri_target)

        # Check all losses present
        assert 'reconstruction_loss' in outputs
        assert 'fmri_loss' in outputs
        assert 'total_loss' in outputs

        # Loss should be scalar
        assert outputs['reconstruction_loss'].dim() == 0
        assert outputs['fmri_loss'].dim() == 0
        assert outputs['total_loss'].dim() == 0

        # Total loss should be combination
        expected_total = (
            model.reconstruction_weight * outputs['reconstruction_loss'] +
            model.fmri_weight * outputs['fmri_loss']
        )
        assert torch.allclose(outputs['total_loss'], expected_total)

    def test_forward_pass_different_weights(self):
        """Test forward pass with different loss weights."""
        model = SherlockAutoencoder(
            reconstruction_weight=2.0,
            fmri_weight=0.5
        )
        model.train()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)
        fmri_target = torch.randn(batch_size, 85810)

        outputs = model(video, audio, text, fmri_target=fmri_target)

        # Check total loss uses correct weights
        expected_total = (
            2.0 * outputs['reconstruction_loss'] +
            0.5 * outputs['fmri_loss']
        )
        assert torch.allclose(outputs['total_loss'], expected_total)

    def test_backward_pass(self):
        """Test backward pass and gradient flow."""
        model = SherlockAutoencoder()
        model.train()

        batch_size = 2
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)
        fmri_target = torch.randn(batch_size, 85810)

        # Forward pass
        outputs = model(video, audio, text, fmri_target=fmri_target)
        loss = outputs['total_loss']

        # Backward pass
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_encode_only(self):
        """Test encoding without decoding."""
        model = SherlockAutoencoder()
        model.eval()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)

        with torch.no_grad():
            # Without voxels
            bottleneck, voxels = model.encode_only(
                video, audio, text, return_voxels=False
            )
            assert bottleneck.shape == (batch_size, 8000)
            assert voxels is None

            # With voxels
            bottleneck, voxels = model.encode_only(
                video, audio, text, return_voxels=True
            )
            assert bottleneck.shape == (batch_size, 8000)
            assert voxels.shape == (batch_size, 85810)

    def test_decode_only(self):
        """Test decoding without encoding."""
        model = SherlockAutoencoder()
        model.eval()

        batch_size = 4
        bottleneck = torch.randn(batch_size, 8000)

        with torch.no_grad():
            video_recon, audio_recon, text_recon = model.decode_only(bottleneck)

        assert video_recon.shape == (batch_size, 43200)
        assert audio_recon.shape == (batch_size, 128)
        assert text_recon.shape == (batch_size, 1024)

    def test_parameter_count(self):
        """Test parameter counting."""
        model = SherlockAutoencoder()
        param_dict = model.get_parameter_count()

        # Check keys present
        assert 'encoder' in param_dict
        assert 'decoder' in param_dict
        assert 'total' in param_dict
        assert 'encoder_breakdown' in param_dict
        assert 'decoder_breakdown' in param_dict

        # Check all positive
        assert param_dict['encoder'] > 0
        assert param_dict['decoder'] > 0
        assert param_dict['total'] > 0

        # Check total equals sum
        assert param_dict['total'] == param_dict['encoder'] + param_dict['decoder']

        # Print summary
        print("\n=== Autoencoder Parameter Count ===")
        print(f"Encoder:  {param_dict['encoder']:>14,} parameters")
        print(f"Decoder:  {param_dict['decoder']:>14,} parameters")
        print(f"Total:    {param_dict['total']:>14,} parameters")
        print("=" * 55)

    def test_custom_dimensions(self):
        """Test autoencoder with custom dimensions."""
        model = SherlockAutoencoder(
            video_height=45,
            video_width=80,
            audio_mels=64,
            text_dim=512,
            n_voxels=10000,
            bottleneck_dim=2000
        )
        model.eval()

        batch_size = 2
        video = torch.randn(batch_size, 3, 45, 80)
        audio = torch.randn(batch_size, 64)
        text = torch.randn(batch_size, 512)

        with torch.no_grad():
            outputs = model(video, audio, text)

        # Check shapes with custom dimensions
        assert outputs['bottleneck'].shape == (batch_size, 2000)
        assert outputs['predicted_fmri'].shape == (batch_size, 10000)
        assert outputs['video_recon'].shape == (batch_size, 45*80*3)
        assert outputs['audio_recon'].shape == (batch_size, 64)
        assert outputs['text_recon'].shape == (batch_size, 512)

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        model = SherlockAutoencoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')

            # Save checkpoint
            model.save_checkpoint(
                path=checkpoint_path,
                epoch=10,
                loss=0.123,
                metadata={'note': 'test checkpoint'}
            )

            assert os.path.exists(checkpoint_path)

            # Load checkpoint
            loaded_model, checkpoint = SherlockAutoencoder.load_checkpoint(
                checkpoint_path
            )

            # Check checkpoint contents
            assert checkpoint['epoch'] == 10
            assert checkpoint['loss'] == 0.123
            assert checkpoint['metadata']['note'] == 'test checkpoint'

            # Check architecture matches
            arch = checkpoint['architecture']
            assert arch['video_height'] == 90
            assert arch['video_width'] == 160
            assert arch['bottleneck_dim'] == 8000

            # Check loaded model works
            loaded_model.eval()
            batch_size = 2
            video = torch.randn(batch_size, 3, 90, 160)
            audio = torch.randn(batch_size, 128)
            text = torch.randn(batch_size, 1024)

            with torch.no_grad():
                outputs = loaded_model(video, audio, text)

            assert outputs['bottleneck'].shape == (batch_size, 8000)

    def test_checkpoint_with_optimizer(self):
        """Test checkpoint with optimizer state."""
        model = SherlockAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')

            # Save with optimizer
            model.save_checkpoint(
                path=checkpoint_path,
                epoch=5,
                optimizer_state=optimizer.state_dict()
            )

            # Load checkpoint
            loaded_model, checkpoint = SherlockAutoencoder.load_checkpoint(
                checkpoint_path
            )

            # Check optimizer state present
            assert 'optimizer_state_dict' in checkpoint

            # Create new optimizer and load state
            new_optimizer = torch.optim.Adam(loaded_model.parameters())
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        model = SherlockAutoencoder().cuda()
        model.eval()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160).cuda()
        audio = torch.randn(batch_size, 128).cuda()
        text = torch.randn(batch_size, 1024).cuda()

        with torch.no_grad():
            outputs = model(video, audio, text)

        # Check outputs on GPU
        assert outputs['bottleneck'].is_cuda
        assert outputs['predicted_fmri'].is_cuda
        assert outputs['video_recon'].is_cuda

    def test_reconstruction_quality_sanity(self):
        """Test that reconstruction is not completely random."""
        model = SherlockAutoencoder()
        model.eval()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)

        with torch.no_grad():
            outputs1 = model(video, audio, text)
            outputs2 = model(video, audio, text)

        # Same input should give same output in eval mode
        assert torch.allclose(
            outputs1['video_recon'],
            outputs2['video_recon']
        )
        assert torch.allclose(
            outputs1['audio_recon'],
            outputs2['audio_recon']
        )


class TestCreateAutoencoder:
    """Test factory function."""

    def test_create_autoencoder_defaults(self):
        """Test factory with defaults."""
        model = create_autoencoder()
        assert isinstance(model, SherlockAutoencoder)
        assert model.n_voxels == 85810
        assert model.bottleneck_dim == 8000

    def test_create_autoencoder_custom(self):
        """Test factory with custom parameters."""
        model = create_autoencoder(
            n_voxels=10000,
            bottleneck_dim=2000,
            reconstruction_weight=2.0,
            fmri_weight=0.5
        )
        assert model.n_voxels == 10000
        assert model.bottleneck_dim == 2000
        assert model.reconstruction_weight == 2.0
        assert model.fmri_weight == 0.5


class TestPrepareForDistributed:
    """Test distributed training preparation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_requires_init(self):
        """Test that prepare_for_distributed requires initialized process group."""
        model = SherlockAutoencoder()

        # Should raise error if not initialized
        with pytest.raises(RuntimeError, match="not initialized"):
            prepare_for_distributed(model)


class TestAutoencoderIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_training_step(self):
        """Test complete training step."""
        model = SherlockAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()

        # Create dummy batch
        batch_size = 8
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)
        fmri_target = torch.randn(batch_size, 85810)

        # Forward pass
        outputs = model(video, audio, text, fmri_target=fmri_target)
        loss = outputs['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(loss)

    def test_batch_consistency(self):
        """Test that batch processing is consistent."""
        model = SherlockAutoencoder()
        model.eval()

        # Create batch
        batch_size = 16
        video_batch = torch.randn(batch_size, 3, 90, 160)
        audio_batch = torch.randn(batch_size, 128)
        text_batch = torch.randn(batch_size, 1024)

        # Process as batch
        with torch.no_grad():
            outputs_batch = model(video_batch, audio_batch, text_batch)

        # Process individually
        outputs_individual = []
        for i in range(batch_size):
            with torch.no_grad():
                out = model(
                    video_batch[i:i+1],
                    audio_batch[i:i+1],
                    text_batch[i:i+1]
                )
            outputs_individual.append(out)

        # Compare bottlenecks
        bottleneck_batch = outputs_batch['bottleneck']
        bottleneck_individual = torch.cat(
            [o['bottleneck'] for o in outputs_individual],
            dim=0
        )

        assert torch.allclose(
            bottleneck_batch,
            bottleneck_individual,
            atol=1e-5
        )

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        model = SherlockAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        accumulation_steps = 4

        # Accumulate gradients
        optimizer.zero_grad()
        total_loss = 0

        for step in range(accumulation_steps):
            video = torch.randn(2, 3, 90, 160)
            audio = torch.randn(2, 128)
            text = torch.randn(2, 1024)
            fmri_target = torch.randn(2, 85810)

            outputs = model(video, audio, text, fmri_target=fmri_target)
            loss = outputs['total_loss'] / accumulation_steps
            loss.backward()

            total_loss += loss.item()

        # Update
        optimizer.step()

        # Check loss is finite
        assert np.isfinite(total_loss)

    def test_multiple_epochs(self):
        """Test training over multiple epochs."""
        model = SherlockAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()

        # Simulate 3 epochs with 5 batches each
        for epoch in range(3):
            epoch_loss = 0
            for batch in range(5):
                video = torch.randn(4, 3, 90, 160)
                audio = torch.randn(4, 128)
                text = torch.randn(4, 1024)
                fmri_target = torch.randn(4, 85810)

                outputs = model(video, audio, text, fmri_target=fmri_target)
                loss = outputs['total_loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

            # Check loss is finite
            assert np.isfinite(epoch_loss)

    def test_eval_train_mode_switch(self):
        """Test switching between eval and train modes."""
        model = SherlockAutoencoder()

        batch_size = 4
        video = torch.randn(batch_size, 3, 90, 160)
        audio = torch.randn(batch_size, 128)
        text = torch.randn(batch_size, 1024)

        # Eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(video, audio, text)
        assert 'total_loss' not in outputs_eval

        # Train mode
        model.train()
        outputs_train = model(video, audio, text)
        assert 'total_loss' in outputs_train

        # Back to eval
        model.eval()
        with torch.no_grad():
            outputs_eval2 = model(video, audio, text)
        assert 'total_loss' not in outputs_eval2


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

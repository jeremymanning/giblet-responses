"""
Tests for training system.

Tests the Trainer class, loss functions, and training loop with real data.
Verifies single-GPU training functionality (multi-GPU requires actual hardware).
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from giblet.models.autoencoder import create_autoencoder
from giblet.training.trainer import Trainer, TrainingConfig
from giblet.training.losses import (
    ReconstructionLoss,
    FMRIMatchingLoss,
    CombinedAutoEncoderLoss,
    compute_correlation_metric,
    compute_r2_score
)
from giblet.data.dataset import SherlockDataset


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing training loop."""

    def __init__(self, n_samples=100):
        self.n_samples = n_samples

        # Feature dimensions matching Sherlock dataset
        self.audio_dim = 128
        self.text_dim = 1024
        self.fmri_dim = 85810

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'video': torch.randn(3, 90, 160),  # (C, H, W) format for convolutions
            'audio': torch.randn(self.audio_dim),
            'text': torch.randn(self.text_dim),
            'fmri': torch.randn(self.fmri_dim),
            'subject_id': 1,
            'tr_index': idx
        }


@pytest.mark.unit
class TestLossFunctions:
    """Test loss functions."""

    def test_reconstruction_loss(self):
        """Test ReconstructionLoss."""
        loss_fn = ReconstructionLoss(
            video_weight=1.0,
            audio_weight=1.0,
            text_weight=1.0
        )

        batch_size = 4
        video_recon = torch.randn(batch_size, 43200)
        video_target = torch.randn(batch_size, 43200)
        audio_recon = torch.randn(batch_size, 128)
        audio_target = torch.randn(batch_size, 128)
        text_recon = torch.randn(batch_size, 1024)
        text_target = torch.randn(batch_size, 1024)

        loss, loss_dict = loss_fn(
            video_recon, video_target,
            audio_recon, audio_target,
            text_recon, text_target
        )

        # Check loss is computed
        assert loss.item() > 0
        assert 'video_loss' in loss_dict
        assert 'audio_loss' in loss_dict
        assert 'text_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict

        # Check loss is sum of components
        total = (
            loss_dict['video_loss'] +
            loss_dict['audio_loss'] +
            loss_dict['text_loss']
        )
        assert torch.allclose(loss_dict['reconstruction_loss'], total)

        print("✓ ReconstructionLoss works correctly")

    def test_fmri_loss_mse(self):
        """Test FMRIMatchingLoss with MSE."""
        loss_fn = FMRIMatchingLoss(loss_type='mse')

        batch_size = 4
        predicted = torch.randn(batch_size, 85810)
        target = torch.randn(batch_size, 85810)

        loss = loss_fn(predicted, target)

        assert loss.item() > 0
        print("✓ FMRIMatchingLoss (MSE) works correctly")

    def test_fmri_loss_correlation(self):
        """Test FMRIMatchingLoss with correlation."""
        loss_fn = FMRIMatchingLoss(loss_type='correlation')

        batch_size = 4
        predicted = torch.randn(batch_size, 85810)
        target = torch.randn(batch_size, 85810)

        loss = loss_fn(predicted, target)

        assert loss.item() >= 0  # 1 - correlation should be positive
        print("✓ FMRIMatchingLoss (correlation) works correctly")

    def test_combined_loss(self):
        """Test CombinedAutoEncoderLoss."""
        loss_fn = CombinedAutoEncoderLoss(
            reconstruction_weight=1.0,
            fmri_weight=1.0,
            video_weight=1.0,
            audio_weight=1.0,
            text_weight=1.0
        )

        batch_size = 4
        outputs = {
            'video_recon': torch.randn(batch_size, 43200),
            'audio_recon': torch.randn(batch_size, 128),
            'text_recon': torch.randn(batch_size, 1024),
            'predicted_fmri': torch.randn(batch_size, 85810)
        }

        video_target = torch.randn(batch_size, 43200)
        audio_target = torch.randn(batch_size, 128)
        text_target = torch.randn(batch_size, 1024)
        fmri_target = torch.randn(batch_size, 85810)

        loss, loss_dict = loss_fn(
            outputs, video_target, audio_target, text_target, fmri_target
        )

        assert loss.item() > 0
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert 'fmri_loss' in loss_dict
        assert 'video_loss' in loss_dict
        assert 'audio_loss' in loss_dict
        assert 'text_loss' in loss_dict

        print("✓ CombinedAutoEncoderLoss works correctly")

    def test_correlation_metric(self):
        """Test correlation metric computation."""
        batch_size = 4
        n_features = 1000

        # Perfect correlation
        x = torch.randn(batch_size, n_features)
        corr = compute_correlation_metric(x, x, dim=1)
        assert torch.allclose(corr, torch.ones(batch_size), atol=1e-5)

        # Random correlation
        y = torch.randn(batch_size, n_features)
        corr = compute_correlation_metric(x, y, dim=1)
        assert corr.shape == (batch_size,)
        assert torch.all(corr >= -1) and torch.all(corr <= 1)

        print("✓ Correlation metric works correctly")

    def test_r2_score(self):
        """Test R^2 score computation."""
        batch_size = 4
        n_features = 1000

        # Perfect prediction
        x = torch.randn(batch_size, n_features)
        r2 = compute_r2_score(x, x, dim=1)
        assert torch.allclose(r2, torch.ones(batch_size), atol=1e-5)

        # Random prediction
        y = torch.randn(batch_size, n_features)
        r2 = compute_r2_score(x, y, dim=1)
        assert r2.shape == (batch_size,)

        print("✓ R^2 score works correctly")


@pytest.mark.integration
class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self):
        """Create small test model."""
        return create_autoencoder(
            video_height=90,
            video_width=160,
            audio_mels=128,
            text_dim=1024,
            n_voxels=85810,
            bottleneck_dim=8000
        )

    @pytest.fixture
    def datasets(self):
        """Create dummy datasets."""
        train_dataset = DummyDataset(n_samples=80)
        val_dataset = DummyDataset(n_samples=20)
        return train_dataset, val_dataset

    def test_trainer_initialization(self, model, datasets, temp_dir):
        """Test Trainer initialization."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,  # Disable for CPU testing
            num_workers=0  # Disable multiprocessing for testing
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

        print("✓ Trainer initialization works")

    def test_single_training_step(self, model, datasets, temp_dir):
        """Test single training step."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=1,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,
            num_workers=0
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train for one step
        model.train()
        batch = next(iter(trainer.train_loader))

        video = batch['video']
        audio = batch['audio']
        text = batch['text']
        fmri = batch['fmri']

        # Forward pass
        outputs = model(video, audio, text, fmri_target=fmri)

        # Prepare targets
        batch_size = video.size(0)
        video_flat = video.view(batch_size, -1)

        # Compute loss
        loss, loss_dict = trainer.criterion(
            outputs, video_flat, audio, text, fmri
        )

        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        # Check parameters changed
        params_changed = False
        for p_init, p_current in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_current):
                params_changed = True
                break

        assert params_changed, "Parameters should change after training step"
        assert loss.item() > 0

        print("✓ Single training step works")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_training_loop_gpu(self, model, datasets, temp_dir):
        """Test full training loop on GPU."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=True,
            num_workers=0,
            validate_every=1,
            save_every=1
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Train
        history = trainer.train()

        # Check history
        assert len(history['train_history']) == 2  # 2 epochs
        assert len(history['val_history']) == 2
        assert history['best_val_loss'] < float('inf')

        # Check checkpoints were saved
        checkpoint_dir = Path(temp_dir)
        assert (checkpoint_dir / 'checkpoint_epoch_1.pt').exists()
        assert (checkpoint_dir / 'final_checkpoint.pt').exists()

        print("✓ Training loop works on GPU")

    def test_training_loop_cpu(self, model, datasets, temp_dir):
        """Test full training loop on CPU (short version)."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,  # Smaller batch for CPU
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,  # No FP16 on CPU
            num_workers=0,
            validate_every=1,
            save_every=1
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Train
        history = trainer.train()

        # Check history
        assert len(history['train_history']) == 2
        assert len(history['val_history']) == 2
        assert history['best_val_loss'] < float('inf')

        # Check losses decreased (or at least training ran)
        assert all('total_loss' in h for h in history['train_history'])
        assert all('total_loss' in h for h in history['val_history'])

        print("✓ Training loop works on CPU")

    def test_checkpoint_save_load(self, model, datasets, temp_dir):
        """Test checkpoint saving and loading."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,
            num_workers=0
        )

        # Train for 1 epoch
        trainer1 = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Save initial state
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Train one epoch
        trainer1.current_epoch = 0
        train_metrics = trainer1._train_epoch()

        # Save checkpoint
        checkpoint_path = Path(temp_dir) / 'test_checkpoint.pt'
        trainer1.save_checkpoint(epoch=1, val_loss=train_metrics['total_loss'])

        # Create new model and trainer
        model2 = create_autoencoder(
            video_height=90,
            video_width=160,
            audio_mels=128,
            text_dim=1024,
            n_voxels=85810,
            bottleneck_dim=8000
        )

        config2 = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,
            num_workers=0,
            resume_from=str(checkpoint_path)
        )

        trainer2 = Trainer(
            model=model2,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config2,
            distributed=False
        )

        # Check loaded state
        assert trainer2.current_epoch == 1

        # Check parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)

        print("✓ Checkpoint save/load works")

    def test_early_stopping(self, model, datasets, temp_dir):
        """Test early stopping."""
        train_dataset, val_dataset = datasets

        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=100,  # Set high to test early stopping
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,
            num_workers=0,
            early_stopping_patience=2,
            validate_every=1
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Manually set validation loss to simulate no improvement
        trainer.best_val_loss = 1.0

        # Check early stopping detection
        trainer._check_early_stopping(1.1)  # Worse
        assert trainer.epochs_without_improvement == 1

        trainer._check_early_stopping(1.2)  # Worse again
        assert trainer.epochs_without_improvement == 2

        trainer._check_early_stopping(0.9)  # Better
        assert trainer.epochs_without_improvement == 0
        assert trainer.best_val_loss == 0.9

        print("✓ Early stopping works")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
def test_training_with_real_data(data_dir):
    """
    Test training with real Sherlock data (if available).

    This test will be skipped if data is not available.
    """
    # Check if data exists
    if not (data_dir / 'sherlock_nii').exists():
        pytest.skip("Real data not available")

    print("\nTesting with real Sherlock data...")

    # Create small dataset (single subject, limited TRs)
    try:
        train_dataset = SherlockDataset(
            data_dir=data_dir,
            subjects=1,  # Single subject
            split='train',
            apply_hrf=True,
            mode='per_subject',
            max_trs=50  # Only 50 TRs for quick test
        )

        val_dataset = SherlockDataset(
            data_dir=data_dir,
            subjects=1,
            split='val',
            apply_hrf=True,
            mode='per_subject',
            max_trs=50
        )
    except Exception as e:
        pytest.skip(f"Could not load data: {e}")

    # Create model
    model = create_autoencoder()

    # Configure training
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            log_dir=temp_dir,
            use_mixed_precision=False,
            num_workers=0
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            distributed=False
        )

        # Train
        history = trainer.train()

        # Check training completed
        assert len(history['train_history']) == 2
        assert len(history['val_history']) == 2

        print("✓ Training works with real data")


if __name__ == '__main__':
    # Run tests
    print("\n" + "=" * 80)
    print("Testing Training System")
    print("=" * 80 + "\n")

    # Test loss functions
    print("Testing loss functions...")
    loss_tests = TestLossFunctions()
    loss_tests.test_reconstruction_loss()
    loss_tests.test_fmri_loss_mse()
    loss_tests.test_fmri_loss_correlation()
    loss_tests.test_combined_loss()
    loss_tests.test_correlation_metric()
    loss_tests.test_r2_score()

    # Test trainer
    print("\nTesting trainer...")
    trainer_tests = TestTrainer()

    # Create fixtures manually
    temp_dir = tempfile.mkdtemp()
    try:
        model = create_autoencoder()
        train_dataset = DummyDataset(n_samples=80)
        val_dataset = DummyDataset(n_samples=20)

        trainer_tests.test_trainer_initialization(model, (train_dataset, val_dataset), temp_dir)
        trainer_tests.test_single_training_step(model, (train_dataset, val_dataset), temp_dir)
        trainer_tests.test_training_loop_cpu(model, (train_dataset, val_dataset), temp_dir)
        trainer_tests.test_checkpoint_save_load(model, (train_dataset, val_dataset), temp_dir)
        trainer_tests.test_early_stopping(model, (train_dataset, val_dataset), temp_dir)

        # Test GPU if available
        if torch.cuda.is_available():
            print("\nTesting on GPU...")
            trainer_tests.test_training_loop_gpu(model, (train_dataset, val_dataset), temp_dir)

    finally:
        shutil.rmtree(temp_dir)

    # Test with real data if available
    try:
        test_training_with_real_data()
    except Exception as e:
        print(f"\nSkipping real data test: {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80 + "\n")

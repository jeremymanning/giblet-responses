"""
Centralized pytest fixtures for giblet-responses test suite.

This file provides shared fixtures to reduce code duplication and
ensure consistent test setup across all test modules.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


# ==================== Path Fixtures ====================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory path."""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


# ==================== Device Fixtures ====================

@pytest.fixture(scope="session")
def device():
    """Return device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def use_cuda():
    """Return True if CUDA is available and should be used."""
    return torch.cuda.is_available()


# ==================== Data Fixtures ====================

@pytest.fixture
def sample_audio_features():
    """Generate sample audio features (EnCodec format)."""
    # (n_trs, n_codebooks, frames_per_tr)
    # Example: 10 TRs, 1 codebook, 112 frames per TR
    return torch.randint(0, 1024, (10, 1, 112), dtype=torch.long)


@pytest.fixture
def sample_video_features():
    """Generate sample video features."""
    # (n_trs, height, width, channels)
    # Example: 10 TRs, 90x160x3
    return torch.rand(10, 90, 160, 3, dtype=torch.float32)


@pytest.fixture
def sample_text_features():
    """Generate sample text embeddings."""
    # (n_trs, embedding_dim)
    # Example: 10 TRs, 1024-dimensional embeddings
    return torch.rand(10, 1024, dtype=torch.float32)


@pytest.fixture
def sample_fmri_data():
    """Generate sample fMRI data."""
    # (n_trs, n_voxels)
    # Example: 10 TRs, 85810 voxels
    return torch.rand(10, 85810, dtype=torch.float32)


# ==================== Model Fixtures ====================

@pytest.fixture
def small_model_config():
    """Return configuration for small test model."""
    return {
        "video_dim": 43200,  # 90×160×3
        "audio_dim": 112,    # EnCodec: 1×112
        "text_dim": 1024,
        "hidden_dim": 256,   # Reduced from 2048
        "bottleneck_dim": 128,  # Reduced from 8000
        "n_voxels": 1000,    # Reduced from 85810
    }


@pytest.fixture
def encoder_config(small_model_config):
    """Return encoder configuration."""
    return {
        **small_model_config,
        "dropout": 0.2,
    }


@pytest.fixture
def decoder_config(small_model_config):
    """Return decoder configuration."""
    return {
        **small_model_config,
        "dropout": 0.2,
    }


# ==================== Processor Fixtures ====================

@pytest.fixture
def audio_processor(tmp_path):
    """Create AudioProcessor instance."""
    from giblet.data.audio import AudioProcessor
    return AudioProcessor(
        use_encodec=True,
        encodec_bandwidth=3.0,
        tr=1.5,
        device='cpu'
    )


@pytest.fixture
def video_processor():
    """Create VideoProcessor instance."""
    from giblet.data.video import VideoProcessor
    return VideoProcessor(
        target_height=90,
        target_width=160,
        tr=1.5,
        normalize=True
    )


@pytest.fixture
def text_processor():
    """Create TextProcessor instance."""
    from giblet.data.text import TextProcessor
    return TextProcessor(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )


# ==================== Cleanup Fixtures ====================

@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup happens automatically with tmp_path
    pass


# ==================== Markers ====================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "data: marks tests requiring data files")

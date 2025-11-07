# tests/ - Test Suite

This directory contains the complete test suite for the giblet-responses project, including unit tests, integration tests, and diagnostic scripts.

## Quick Start

### Running All Tests

```bash
# From project root
pytest tests/

# With coverage report
pytest tests/ --cov=giblet --cov-report=html

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

### Running Specific Test Categories

```bash
# Unit tests only
pytest tests/test_*.py tests/*/test_*.py

# Integration tests
pytest tests/integration/

# Data pipeline tests
pytest tests/data/

# Model tests
pytest tests/models/

# Diagnostics (manual)
python tests/diagnostics/test_nccl_health.py
```

## Test Structure

```
tests/
├── test_*.py                  # Root-level unit tests (8 files)
├── data/                      # Data pipeline tests (9 files)
│   ├── test_audio_*.py       # Audio processing tests
│   ├── test_video_*.py       # Video processing tests
│   ├── test_text.py          # Text processing tests
│   ├── test_dataset.py       # Dataset tests
│   └── test_encodec_*.py     # EnCodec integration tests
├── models/                    # Model architecture tests (11 files)
│   ├── test_encoder.py       # Encoder tests
│   ├── test_decoder.py       # Decoder tests
│   ├── test_autoencoder.py   # Full autoencoder tests
│   └── verify_*.py           # Verification scripts
├── integration/               # End-to-end tests (4 files)
│   ├── test_encodec_e2e_pipeline.py
│   ├── test_hrf.py
│   └── validate_all_modalities.py
├── diagnostics/               # Diagnostic scripts (7 files)
│   ├── test_nccl_*.py        # Distributed training diagnostics
│   ├── debug_*.py            # Debugging utilities
│   └── reproduce_*.py        # Bug reproduction scripts
├── utils/                     # Utility tests (1 file)
│   └── test_visualization.py
└── training/                  # Training tests (EMPTY - to be added)
```

## Test Categories

### Unit Tests

Fast, isolated tests for individual components.

**Root Level** (`tests/test_*.py`)
- `test_fmri_processor.py` - FMRIProcessor class tests
- `test_embeddings.py` - Text embedding tests
- `test_sync.py` - Temporal synchronization tests
- `test_training.py` - Training loop tests
- `test_sherlock_quick.py` - Quick integration smoke test

**Run:**
```bash
pytest tests/test_*.py -v
```

---

### Data Pipeline Tests

Test data loading, preprocessing, and alignment.

**tests/data/** (9 files)

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `test_dataset.py` | MultimodalDataset class | Loading, caching, splits |
| `test_audio_encodec.py` | EnCodec audio processing | Encoding, temporal concat |
| `test_audio_temporal_concatenation.py` | Audio temporal alignment | TR binning, concatenation |
| `test_audio_dimension_fix.py` | Audio dimension handling | Shape validation |
| `test_video_temporal.py` | Video temporal concat | Frame aggregation, padding |
| `test_text.py` | Text processing | Embeddings, temporal alignment |
| `test_encodec_sherlock_integration.py` | EnCodec + Sherlock data | End-to-end data pipeline |

**Key Tests:**
- **Dataset loading**: Verify all 17 subjects load correctly
- **Caching**: Ensure cache files are created and reused
- **Temporal alignment**: All modalities align to fMRI TRs
- **Feature dimensions**: Correct shapes for video/audio/text
- **EnCodec integration**: Proper encoding/decoding

**Run:**
```bash
pytest tests/data/ -v

# Specific test
pytest tests/data/test_dataset.py::TestMultimodalDataset::test_load_all_subjects -v
```

---

### Model Architecture Tests

Test neural network components and architectures.

**tests/models/** (11 files)

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `test_encoder.py` | MultimodalEncoder | Forward pass, dimensions, gradients |
| `test_decoder.py` | MultimodalDecoder | Reconstruction, dimensions |
| `test_autoencoder.py` | Full autoencoder | End-to-end, loss computation |
| `test_audio_encoder_encodec.py` | AudioEncoder (EnCodec) | EnCodec-specific encoding |
| `test_audio_decoder_encodec.py` | AudioDecoder (EnCodec) | EnCodec-specific decoding |
| `test_encoder_demo.py` | Encoder demo/visualization | Manual inspection |

**Verification Scripts:**
- `verify_13_layer_architecture.py` - Validate 13-layer architecture per Issue #2
- `verify_audio_fix.py` - Verify EnCodec temporal concatenation fix (Issue #29)
- `verify_checkpoint.py` - Validate checkpoint loading/saving
- `validate_encodec_implementation.py` - EnCodec implementation validation
- `check_layer_sizes.py` - Layer dimension verification

**Run:**
```bash
pytest tests/models/test_*.py -v

# Verification scripts (manual)
python tests/models/verify_13_layer_architecture.py
python tests/models/verify_checkpoint.py checkpoints/best_checkpoint.pt
```

---

### Integration Tests

End-to-end tests across multiple components.

**tests/integration/** (4 files)

| Test File | Purpose | Components Tested |
|-----------|---------|-------------------|
| `test_encodec_e2e_pipeline.py` | EnCodec end-to-end | Audio → EnCodec → Model → Decoder |
| `test_hrf.py` | HRF convolution | Stimulus → HRF → fMRI prediction |
| `validate_all_modalities.py` | Full pipeline validation | Data → Model → Loss |

**Key Tests:**
- **End-to-end forward pass**: Data → Model → Loss computation
- **Gradient flow**: Verify gradients propagate correctly
- **Multi-modality consistency**: All modalities processed consistently
- **EnCodec pipeline**: Audio encoding/decoding through full pipeline

**Run:**
```bash
pytest tests/integration/ -v

# Validation script (manual)
python tests/integration/validate_all_modalities.py
```

---

### Diagnostic Scripts

Manual debugging and diagnostic utilities.

**tests/diagnostics/** (7 files)

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_nccl_health.py` | Test NCCL/distributed setup | `python tests/diagnostics/test_nccl_health.py` |
| `test_nccl_configs.py` | Test NCCL configurations | `python tests/diagnostics/test_nccl_configs.py` |
| `test_small_model_ddp.py` | Test DDP with small model | `torchrun --nproc_per_node=2 tests/diagnostics/test_small_model_ddp.py` |
| `debug_encodec_sherlock.py` | Debug EnCodec + Sherlock | Manual debugging |
| `reproduce_encodec_bug.py` | Reproduce Issue #29 bug | Bug reproduction |
| `verify_fix_sherlock.py` | Verify Issue #29 fix | Fix validation |
| `visualize_dimensions.py` | Visualize tensor shapes | Dimension debugging |

**When to Use:**
- **NCCL errors**: Run `test_nccl_health.py` to diagnose distributed training issues
- **DDP debugging**: Use `test_small_model_ddp.py` for minimal DDP test
- **Dimension mismatches**: Run `visualize_dimensions.py` to inspect tensor shapes
- **EnCodec issues**: Use `debug_encodec_sherlock.py` for EnCodec-specific debugging

**Run:**
```bash
# NCCL health check
python tests/diagnostics/test_nccl_health.py

# DDP test (2 GPUs)
torchrun --nproc_per_node=2 tests/diagnostics/test_small_model_ddp.py

# Dimension visualization
python tests/diagnostics/visualize_dimensions.py
```

---

## Test Fixtures and Configuration

### Current State

The test suite currently **lacks centralized configuration**:
- ❌ No `conftest.py` (centralized fixtures)
- ❌ No `pytest.ini` (test configuration)
- ❌ No test markers (`@pytest.mark.unit`, etc.)

This is tracked in **Issue #31 Phase 5** for future improvement.

### Recommended Fixtures (To Be Implemented)

**tests/conftest.py** (proposed):
```python
import pytest
import torch
from pathlib import Path

@pytest.fixture
def data_dir():
    """Provide path to test data directory."""
    return Path("data/")

@pytest.fixture
def small_dataset():
    """Create small dataset for fast tests."""
    from giblet.data import MultimodalDataset
    return MultimodalDataset(
        data_dir="data/",
        subjects=[1, 2],  # Only 2 subjects
        max_trs=10,       # Only 10 TRs
        apply_hrf=False   # Skip HRF for speed
    )

@pytest.fixture
def model_config():
    """Standard model configuration for tests."""
    return {
        'video_height': 90,
        'video_width': 160,
        'audio_features': 128,
        'text_dim': 1024,
        'n_voxels': 85810,
        'bottleneck_dim': 2048
    }

@pytest.fixture
def device():
    """Provide device for tests (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**pytest.ini** (proposed):
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, multiple components)
    slow: Slow tests (> 10 seconds)
    gpu: Tests requiring GPU
    distributed: Tests requiring multiple GPUs
    manual: Manual diagnostic scripts (not run by default)

addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

---

## Running Tests by Category

### Unit Tests Only

```bash
pytest tests/test_*.py tests/data/ tests/models/test_*.py -v
```

### Integration Tests Only

```bash
pytest tests/integration/ -v
```

### GPU Tests Only

```bash
# Requires pytest markers (to be implemented)
pytest -m gpu -v
```

### Fast Tests Only (< 10s)

```bash
# Requires pytest markers (to be implemented)
pytest -m "not slow" -v
```

### Distributed Tests

```bash
# Manual distributed tests
torchrun --nproc_per_node=2 tests/diagnostics/test_small_model_ddp.py

# Future: pytest-based distributed tests
# pytest -m distributed -v
```

---

## Test Coverage

### Current Coverage

Run with coverage report:
```bash
pytest tests/ --cov=giblet --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

### Coverage Targets

| Module | Current Coverage | Target | Status |
|--------|------------------|--------|--------|
| **giblet/data/** | ~80% | 90% | ✅ Good |
| **giblet/models/** | ~70% | 85% | ⚠️ Needs improvement |
| **giblet/alignment/** | ~60% | 80% | ⚠️ Needs improvement |
| **giblet/training/** | ~40% | 80% | ❌ Low coverage |
| **giblet/utils/** | ~50% | 75% | ⚠️ Needs improvement |

---

## Writing New Tests

### Unit Test Template

```python
import pytest
import torch
from giblet.models import MultimodalAutoencoder

class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self):
        """Test basic forward pass."""
        # Arrange
        model = MultimodalAutoencoder(...)
        input_data = torch.randn(32, 259200)

        # Act
        output = model(input_data)

        # Assert
        assert output.shape == (32, 85810)
        assert not torch.isnan(output).any()

    def test_edge_case(self):
        """Test edge case: empty input."""
        model = MultimodalAutoencoder(...)
        input_data = torch.zeros(1, 259200)

        with pytest.raises(ValueError):
            model(input_data)
```

### Integration Test Template

```python
import pytest
import torch
from giblet.data import MultimodalDataset
from giblet.models import MultimodalAutoencoder

@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_pipeline():
    """Test complete pipeline: data loading → model → loss."""
    # Load data
    dataset = MultimodalDataset(
        data_dir="data/",
        subjects=[1],
        max_trs=10,
        split=None
    )

    # Create model
    model = MultimodalAutoencoder(...)

    # Forward pass
    sample = dataset[0]
    outputs = model(
        sample['video'],
        sample['audio'],
        sample['text']
    )

    # Compute loss
    loss, _ = model.compute_loss(
        outputs=outputs,
        fmri_target=sample['fmri'],
        video_target=sample['video'],
        audio_target=sample['audio'],
        text_target=sample['text']
    )

    # Assertions
    assert loss.item() > 0
    assert torch.isfinite(loss)
```

---

## Test Data

### Data Requirements

Tests require the Sherlock dataset to be downloaded:

```bash
# Download test data (lab members only)
./download_data_from_dropbox.sh
```

This provides:
- Video stimulus (272 MB)
- Scene annotations (173 KB)
- fMRI data for 17 subjects (~10.7 GB)

### Mock Data

For fast tests without real data, use small synthetic datasets:

```python
import torch
import numpy as np

def create_mock_video(batch_size=32, height=90, width=160, frames=18):
    """Create mock video features."""
    return torch.randn(batch_size, frames * height * width * 3)

def create_mock_audio(batch_size=32, features=128):
    """Create mock audio features (EnCodec)."""
    return torch.randn(batch_size, features)

def create_mock_text(batch_size=32, dim=1024):
    """Create mock text embeddings."""
    return torch.randn(batch_size, dim)

def create_mock_fmri(batch_size=32, n_voxels=85810):
    """Create mock fMRI data."""
    return torch.randn(batch_size, n_voxels)
```

---

## Continuous Integration

### GitHub Actions (Proposed)

**.github/workflows/tests.yml**:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/ --cov=giblet --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Known Issues and Limitations

### Current Issues

1. **Missing pytest infrastructure** (Issue #31 Phase 5)
   - No centralized fixtures
   - No test markers
   - Inconsistent test structure

2. **Incomplete test coverage** (Issue #31 Phase 5)
   - Training module: 40% coverage
   - Alignment module: 60% coverage
   - Utils module: 50% coverage

3. **Failing tests** (Issue #31 Phase 5)
   - 41 tests currently failing
   - Need comprehensive review and fixes

4. **No automated CI/CD** (Issue #31 Phase 4)
   - No GitHub Actions workflows
   - No pre-commit hooks
   - Manual test execution required

### Planned Improvements (Issue #31)

**Phase 4: Pre-commit Hooks and CI/CD**
- Set up GitHub Actions for automated testing
- Configure pre-commit hooks (black, flake8, isort)
- Add coverage reporting to CI

**Phase 5: Test Suite Fixes**
- Create `tests/conftest.py` with centralized fixtures
- Create `pytest.ini` with markers and configuration
- Fix all 41 failing tests
- Improve test coverage to 80%+ across all modules
- Rename non-standard test files (11 files)
- Document external dependencies

---

## External Dependencies

Some tests require external resources:

| Dependency | Required For | Install |
|------------|--------------|---------|
| **PyTorch** | All model tests | `pip install torch` |
| **nibabel** | fMRI tests | `pip install nibabel` |
| **librosa** | Audio tests | `pip install librosa` |
| **transformers** | EnCodec tests | `pip install transformers` |
| **sentence-transformers** | Text embedding tests | `pip install sentence-transformers` |
| **pytest** | Test runner | `pip install pytest` |
| **pytest-cov** | Coverage reports | `pip install pytest-cov` |

**Install all test dependencies:**
```bash
pip install -r requirements.txt pytest pytest-cov
```

---

## Troubleshooting

### Common Test Failures

**Issue**: `FileNotFoundError: data/sherlock_nii/sub-01.nii.gz`
- **Solution**: Download data with `./download_data_from_dropbox.sh`

**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Reduce batch size or use CPU-only tests
- **Solution**: Use `pytest -k "not gpu"` to skip GPU tests

**Issue**: `ImportError: No module named 'transformers'`
- **Solution**: Install missing dependency: `pip install transformers`

**Issue**: Tests hang during distributed training tests
- **Solution**: Kill hanging processes: `pkill -f python`
- **Solution**: Use diagnostic script: `python tests/diagnostics/test_nccl_health.py`

**Issue**: `ModuleNotFoundError: No module named 'giblet'`
- **Solution**: Run tests from project root, not from `tests/` directory

---

## Best Practices

### When Writing Tests

1. **Use descriptive test names**: `test_encoder_produces_correct_dimensions`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test one thing per test**: Don't combine multiple assertions
4. **Use fixtures**: Share common setup across tests
5. **Mock external dependencies**: Use mocks for network calls, file I/O
6. **Test edge cases**: Empty inputs, extreme values, boundary conditions
7. **Add docstrings**: Explain what the test validates

### When Running Tests

1. **Run tests before committing**: `pytest tests/ -v`
2. **Check coverage**: `pytest tests/ --cov=giblet --cov-report=term`
3. **Fix failing tests immediately**: Don't let tests stay red
4. **Use `-x` flag during debugging**: Stop on first failure
5. **Use `-v` flag**: Verbose output helps identify issues
6. **Use `-k` flag**: Run specific tests: `pytest -k "test_encoder"`

---

## Related Documentation

- **[giblet/data/README.md](../giblet/data/README.md)** - Data pipeline documentation
- **[giblet/models/README.md](../giblet/models/README.md)** - Model architecture documentation
- **[Issue #31](https://github.com/ContextLab/giblet-responses/issues/31)** - Repository reorganization (includes test improvements)

For questions or issues, see the main project [README.md](../README.md) or open an issue on GitHub.

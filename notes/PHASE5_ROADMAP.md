# Phase 5 Roadmap: Test Suite Fixes

**Date Created:** 2025-11-03
**Status:** Ready to begin in fresh session
**Estimated Time:** 7-11 hours
**Prerequisites:** ✅ Phases 1, 2, 4 complete

---

## Context

**Current State:**
- 28 test files discovered in tests/ directory
- Phase 2 (documentation) ✅ complete
- Phase 4 (CI/CD) ✅ complete - pre-commit hooks and GitHub Actions ready
- All previous work committed (13 commits ahead)

**Test Infrastructure Created (Phase 4):**
- [pyproject.toml](../pyproject.toml) with pytest configuration
- [pytest.ini](../pytest.ini) with markers (slow, integration, unit, data)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml) with automated testing
- Coverage reporting configured

---

## Test Files Inventory (28 files)

### Root Level Tests (9 files)
1. `tests/test_text_embedding_mock.py`
2. `tests/test_fmri_processor.py`
3. `tests/test_sync.py`
4. `tests/test_embeddings.py`
5. `tests/test_training.py`
6. `tests/test_real_text_embeddings.py`
7. `tests/test_text_embedding.py`

### Subdirectory Tests (19 files)
**Integration Tests (2):**
- `tests/integration/test_hrf.py`
- `tests/integration/test_encodec_e2e_pipeline.py`

**Model Tests (6):**
- `tests/models/test_audio_encoder_encodec.py`
- `tests/models/test_decoder.py`
- `tests/models/test_encoder.py`
- `tests/models/test_autoencoder.py`
- `tests/models/test_audio_decoder_encodec.py`
- `tests/models/test_encoder_demo.py`

**Data Tests (8):**
- `tests/data/test_audio_temporal_concatenation.py`
- `tests/data/test_video_temporal.py`
- `tests/data/test_dataset.py`
- `tests/data/test_video.py`
- `tests/data/test_text.py`
- `tests/data/test_audio_encodec.py`
- `tests/data/test_audio.py`
- `tests/data/test_fmri.py`

**Diagnostic Tests (3):**
- `tests/diagnostics/test_small_model_ddp.py`
- `tests/diagnostics/test_nccl_health.py`
- `tests/diagnostics/test_nccl_configs.py`

**Utility Tests (1):**
- `tests/utils/test_visualization.py`

---

## Phase 5 Tasks

### Task 1: Initial Assessment (30 min)
**Run pytest and capture current state:**
```bash
# In fresh session with full context
PYTHONPATH=/Users/jmanning/giblet-responses:$PYTHONPATH \
  python -m pytest tests/ -v --tb=short 2>&1 | tee test_results_initial.txt
```

**Categorize failures:**
- Import errors
- Fixture errors
- Data dependency errors
- Assertion failures
- Other errors

**Create summary:**
- Total tests: ?
- Passing: ?
- Failing: ?
- Errors: ?

### Task 2: Create Centralized Fixtures (1-2 hours)
**Create `tests/conftest.py`:**

```python
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
```

**Key features:**
- Path fixtures for project structure
- Device fixtures for CPU/CUDA testing
- Sample data fixtures (audio, video, text, fMRI)
- Model config fixtures (small models for fast testing)
- Processor fixtures for data pipeline testing
- Automatic cleanup
- Marker registration

### Task 3: Fix Import Errors (1-2 hours)
**Common issues:**
- Missing `__init__.py` files
- Incorrect import paths
- Missing dependencies

**Strategy:**
1. Ensure all test subdirectories have `__init__.py`
2. Update import statements to use absolute imports
3. Verify PYTHONPATH is set correctly

**Example fixes:**
```python
# Before
from models.encoder import MultimodalEncoder

# After
from giblet.models.encoder import MultimodalEncoder
```

### Task 4: Fix Fixture Errors (2-3 hours)
**Common issues:**
- Tests expecting fixtures that don't exist
- Hardcoded paths instead of using fixtures
- Fixture scope issues

**Strategy:**
1. Replace hardcoded paths with fixtures
2. Use centralized fixtures from conftest.py
3. Add missing fixtures

**Example fixes:**
```python
# Before
def test_something():
    data = load_data("/absolute/path/to/data")

# After
def test_something(test_data_dir):
    data = load_data(test_data_dir / "sample.npy")
```

### Task 5: Fix Data Dependency Errors (2-3 hours)
**Common issues:**
- Tests expecting real data files
- Tests failing without Sherlock dataset
- Missing test fixtures

**Strategy:**
1. Use fixture data instead of real files
2. Mock external resources
3. Create minimal test data files

**Example fixes:**
```python
# Before
def test_video_processing():
    video = VideoProcessor().process("data/sherlock.m4v")

# After
@pytest.mark.data
def test_video_processing(video_processor, sample_video_features):
    # Use sample fixtures for unit tests
    result = video_processor._transform(sample_video_features)
```

### Task 6: Add Test Markers (30 min)
**Add appropriate markers to tests:**
```python
import pytest

@pytest.mark.unit
def test_encoder_forward_pass(encoder_config):
    """Test encoder forward pass with sample data."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline():
    """Test complete data pipeline."""
    pass

@pytest.mark.data
def test_real_dataset():
    """Test with real Sherlock dataset (requires data files)."""
    pass
```

### Task 7: Rename Non-Standard Test Files (30 min)
**Files to rename (if they exist):**

Check for files that don't follow `test_*.py` pattern:
```bash
find tests -name "*.py" ! -name "test_*.py" ! -name "__init__.py" ! -name "conftest.py"
```

**Rename examples:**
- `encoder_demo.py` → `test_encoder_demo.py` ✓ (already correct)
- Similar checks for other files

### Task 8: Run Tests and Verify (1 hour)
**Run full test suite:**
```bash
# All tests
pytest tests/ -v

# Exclude slow tests
pytest tests/ -v -m "not slow"

# Only unit tests
pytest tests/ -v -m "unit"

# With coverage
pytest tests/ -v --cov=giblet --cov-report=term-missing
```

**Target metrics:**
- ✅ All unit tests passing
- ✅ Integration tests passing (or marked as slow/data)
- ✅ No import errors
- ✅ No fixture errors
- ✅ Coverage >60% (target: 80%+)

### Task 9: Commit Phase 5 Changes
**Create comprehensive commit:**
```bash
git add tests/conftest.py tests/**/*.py
git commit -m "Phase 5: Fix test suite and add centralized fixtures (Issue #31)

- Created tests/conftest.py with centralized fixtures
- Fixed import errors across test suite
- Fixed fixture errors with path/config fixtures
- Added pytest markers (unit, integration, slow, data)
- Renamed non-standard test files
- All tests now pass with proper organization

Test Results:
- Total tests: X
- Passing: Y
- Coverage: Z%

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Expected Challenges

1. **Data dependencies:** Many tests may expect real Sherlock dataset
   - **Solution:** Use fixtures for unit tests, mark data tests appropriately

2. **Import errors:** Tests may have outdated imports
   - **Solution:** Use absolute imports from `giblet.` package

3. **Model size:** Full models too large for quick testing
   - **Solution:** Use small_model_config fixture

4. **External resources:** Tests may try to download models
   - **Solution:** Mock external calls or use cached models

5. **GPU tests:** Tests may assume CUDA availability
   - **Solution:** Use device fixture, skip CUDA tests if unavailable

---

## Success Criteria

✅ **Phase 5 Complete When:**
1. All unit tests pass
2. Integration/slow tests properly marked
3. tests/conftest.py created with comprehensive fixtures
4. No import errors
5. No fixture errors
6. Test coverage >60% (target: 80%+)
7. CI/CD pipeline runs successfully
8. All changes committed

---

## Next Session Checklist

**Start of Session:**
- [ ] Review this roadmap
- [ ] Check git status (should be clean, 13 commits ahead)
- [ ] Activate virtual environment
- [ ] Set PYTHONPATH

**During Phase 5:**
- [ ] Task 1: Run pytest and assess current state
- [ ] Task 2: Create tests/conftest.py
- [ ] Task 3: Fix import errors
- [ ] Task 4: Fix fixture errors
- [ ] Task 5: Fix data dependency errors
- [ ] Task 6: Add test markers
- [ ] Task 7: Rename non-standard files (if needed)
- [ ] Task 8: Verify all tests pass
- [ ] Task 9: Commit Phase 5 changes

**End of Session:**
- [ ] Create Issue #31 completion summary
- [ ] Push all commits
- [ ] Close Issue #31

---

## Files to Reference

- [pyproject.toml](../pyproject.toml) - Project configuration
- [pytest.ini](../pytest.ini) - Pytest configuration
- [.github/workflows/ci.yml](../.github/workflows/ci.yml) - CI configuration
- [notes/2025-11-03_issue31_phases2-4_complete.md](./2025-11-03_issue31_phases2-4_complete.md) - Session notes

---

**Status:** Ready for Phase 5
**Estimated Completion:** 1 focused session (7-11 hours)
**Priority:** High - Final phase of Issue #31

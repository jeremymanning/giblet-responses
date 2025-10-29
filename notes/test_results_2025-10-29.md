# Test Results - 2025-10-29

## Summary
- **Total:** 135 tests
- **Passed:** 112 (83%)
- **Failed:** 20 (15%)
- **Skipped:** 3 (2% - GPU tests)
- **Duration:** 16 minutes 7 seconds

## Failures by Category

### 1. Missing openpyxl (10 failures - SIMPLE FIX)
All `tests/data/test_text.py` tests failing with: `ModuleNotFoundError: No module named 'openpyxl'`

**Fix:** Add openpyxl to requirements_conda.txt
```bash
source venv_py311/bin/activate
pip install openpyxl
```

### 2. Model Dimension Mismatches (6 failures - UPDATE TESTS)
- `test_autoencoder.py::test_forward_pass_eval` - Audio dimension
- `test_autoencoder.py::test_decode_only` - Audio dimension
- `test_decoder.py::test_different_bottleneck_dimensions` - Config issue
- `test_decoder.py::test_different_hidden_dimensions` - Config issue
- `test_decoder.py::test_typical_fmri_dimensions` - Audio 128→2048
- `test_decoder.py::test_memory_efficiency` - Audio 128→2048

**Fix:** Update test files to use audio_mels=2048

### 3. File Path Issues (3 failures - PATH FIXES)
- `test_fmri_processor.py::test_data_files_exist` - Path to sherlock_nii
- `test_fmri_processor.py::test_features_to_nii_roundtrip` - Path issue
- `test_fmri_processor.py::test_load_all_subjects` - Path issue

**Fix:** Update test file paths

### 4. Integration Test (1 failure)
- `test_text.py::test_full_pipeline` - openpyxl (same as #1)

## Action Plan

**Batch 1 (parallel):**
- Agent 1: Fix openpyxl dependency
- Agent 2: Fix model test dimensions
- Agent 3: Fix fMRI test paths

**Then:** Re-run all tests, should be 135/135 passing

# Sherlock Naming Audit - 2025-10-29

Comprehensive audit of all "Sherlock" references in the giblet-responses codebase.

## CATEGORIZATION OF FINDINGS

### Category 1: DATA FILES (KEEP AS-IS)
These are actual Sherlock dataset files - no changes needed:
- `data/stimuli_Sherlock.m4v` - Actual Sherlock episode video
- `data/sherlock_nii/sherlock_movie_s*.nii.gz` - fMRI data files (17 subjects)
- `data/cache/sherlock_*.pkl` - Cached features from Sherlock data

**Rationale:** These ARE Sherlock data, naming is correct.

---

### Category 2: CODE - CLASS NAMES (FIX NEEDED)
Test classes that should be generic:

**Files to fix:**
- `tests/models/test_encoder.py`:
  - `class TestSherlockEncoder` → `class TestMultimodalEncoder`

- `tests/models/test_autoencoder.py`:
  - `class TestSherlockAutoencoder` → `class TestMultimodalAutoencoder`

**Already fixed (aliases exist):**
- `giblet/models/__init__.py`: `SherlockAutoencoder = MultimodalAutoencoder` (backward compat)
- `giblet/training/trainer.py`: `SherlockAutoencoder = MultimodalAutoencoder` (backward compat)

---

### Category 3: CODE - VARIABLE NAMES (FIX NEEDED)

**Hard-coded dataset paths:**
- `giblet/data/dataset.py` (line 191, 214, 216, 260):
  - Cache name: `cache_name = f"sherlock_{subjects_str}_{hrf_str}_{mode_str}.pkl"`
  - Video path: `video_path = self.data_dir / 'stimuli_Sherlock.m4v'`
  - fMRI dir: `fmri_dir = self.data_dir / 'sherlock_nii'`
  - fMRI patterns: `f"sherlock_movie_s{sid}.nii.gz"`

**Solution:** Make configurable via parameters with Sherlock as defaults.

**Test data paths:**
- `tests/test_fmri_processor.py`: `DATA_DIR = Path(".../sherlock_nii")`
  - Keep as-is (testing with real Sherlock data)

---

### Category 4: CODE - IMPORTS (ALREADY FIXED)
These already use generic names or have aliases:
- `from giblet.data.dataset import SherlockDataset` → Should be `MultimodalDataset`
- Most imports already use `MultimodalDataset`, `MultimodalEncoder`, `MultimodalAutoencoder`
- Backward compat aliases exist in `giblet/models/__init__.py`

---

### Category 5: DOCUMENTATION - COMMENTS & DOCSTRINGS (FIX NEEDED)

**Module docstrings:**
- `giblet/data/fmri.py`: "fMRI processing module for Sherlock project" → "for multimodal fMRI project"
- `giblet/data/text.py`: "Text processing module for Sherlock fMRI project" → "multimodal fMRI project"
- `giblet/data/audio.py`: "Audio processing module for Sherlock fMRI project" → "multimodal fMRI project"
- `giblet/data/video.py`: "Video processing module for Sherlock fMRI project" → "multimodal fMRI project"
- `giblet/training/__init__.py`: "Training module for Sherlock autoencoder" → "multimodal autoencoder"
- `giblet/training/losses.py`: "Loss functions for training the Sherlock autoencoder" → "multimodal autoencoder"
- `giblet/training/trainer.py`: "Trainer for Sherlock autoencoder..." → "multimodal autoencoder"

**Code comments:**
- `giblet/data/dataset.py` (line 51): "sherlock_nii/*.nii.gz" → use generic placeholder
- `giblet/data/fmri.py`: "~83,300 brain voxels for Sherlock dataset" → "example dataset"
- `giblet/alignment/sync.py`: "920 TRs for the Sherlock dataset" → "example: 920 TRs"
- Multiple comments referencing "Sherlock dimensions"

---

### Category 6: DOCUMENTATION - README/MD FILES (FIX NEEDED)

**Files to update:**
- `STATUS.md`: Multiple Sherlock references
- `CLAUDE.md`: Dataset description (keep Sherlock mentions as "example")
- `giblet/training/README.md`: Training examples
- `notes/*.md`: Various notes (mostly historical, can keep)
- `examples/demo_sync.py`: Comments
- `examples/train_config.yaml`: Config description
- `scripts/train.py`: Argparse description

---

### Category 7: TEST FILES (CONTEXT-SPECIFIC)

**Test docstrings/comments mentioning Sherlock:**
Keep most as-is since they test WITH Sherlock data:
- `tests/test_fmri_processor.py`: "Tests use real fMRI data from all 17 Sherlock subjects"
- `tests/data/test_dataset.py`: Testing with actual Sherlock data
- `tests/test_training.py`: Tests with real Sherlock data

**Exception - test class names:** Fix as noted in Category 2.

---

### Category 8: EXAMPLES (UPDATE TO SHOW AS EXAMPLE)

**Files:**
- `examples/demo_sync.py`: "Sherlock dataset dimensions" → "example dataset dimensions (Sherlock)"
- `examples/train_example.py`: "Sherlock Autoencoder" → "Multimodal Autoencoder (trained on Sherlock)"

---

## SUMMARY STATISTICS

**Total references found:** ~150+ across all files

**Categories:**
1. Data files (KEEP): ~20 files
2. Class names (FIX): 2 test classes
3. Variable names (FIX): ~10 locations
4. Imports (DONE): Already using generic names
5. Code comments (FIX): ~30 locations
6. Documentation (FIX): ~15 files
7. Test files (KEEP MOST): ~10 files
8. Examples (UPDATE): ~3 files

---

## FIXES REQUIRED

### High Priority (Affects Public API):
1. Rename test classes
2. Make dataset paths configurable
3. Update module docstrings
4. Update training README
5. Update examples to show Sherlock as example

### Medium Priority (Documentation):
1. Update STATUS.md
2. Update CLAUDE.md (keep Sherlock as example)
3. Update inline comments

### Low Priority (Can keep):
1. Notes files (historical record)
2. Test files that explicitly test Sherlock data
3. Cache filenames (internal implementation)

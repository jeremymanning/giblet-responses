# Naming Guidelines for Giblet Multimodal Autoencoder

**Purpose:** Keep the codebase stimulus-agnostic while using Sherlock as our example dataset.

---

## Core Principle

The toolbox is **stimulus-agnostic** - it works with any multimodal fMRI dataset.
Sherlock is our **example dataset**, not a hard dependency.

---

## DO: Use Generic Names

### Class Names
✅ **Use:** `MultimodalDataset`, `MultimodalEncoder`, `MultimodalAutoencoder`
❌ **Avoid:** `SherlockDataset`, `SherlockEncoder`, `SherlockAutoencoder`

**Exception:** Backward compatibility aliases (e.g., `SherlockAutoencoder = MultimodalAutoencoder`) are acceptable in `__init__.py` files with clear comments.

### Variable Names
✅ **Use:** `dataset`, `stimulus_data`, `fmri_data`, `video_path`, `data_dir`
❌ **Avoid:** `sherlock_dataset`, `sherlock_path`, `sherlock_dir` (unless referring to actual Sherlock data)

### Function Parameters
✅ **Use:** `video_path`, `fmri_dir`, `stimulus_path`
❌ **Avoid:** `sherlock_video`, `sherlock_fmri`

### Comments and Docstrings
✅ **Use:**
- "Example dataset (Sherlock)"
- "Typical result: ~83,300 voxels"
- "Example: 920 TRs at TR=1.5s"

❌ **Avoid:**
- "Sherlock dataset dimensions"
- "For the Sherlock dataset"
- "Sherlock-specific processing"

---

## KEEP: Dataset-Specific References

### Data Files
These ARE Sherlock data - naming is correct:
- `data/stimuli_Sherlock.m4v` ✓
- `data/sherlock_nii/*.nii.gz` ✓
- `data/cache/sherlock_*.pkl` ✓

### Test Files
When tests explicitly use Sherlock data:
✅ **Use:** "Tests use real fMRI data from all 17 Sherlock subjects"
✅ **Use:** `DATA_DIR = Path("data/sherlock_nii")`

Tests are allowed to reference Sherlock because they're testing WITH Sherlock data.

### Documentation
In project documentation (README.md, CLAUDE.md, STATUS.md):
✅ **Use:** Sherlock as the example dataset
✅ **Explain:** "This project uses the Sherlock dataset as an example..."

Documentation should clarify what dataset we're using for development.

---

## Module Docstrings

### DO:
```python
"""
Video processing module for multimodal fMRI autoencoder project.

Default temporal alignment uses TR = 1.5 seconds (configurable).
"""
```

### DON'T:
```python
"""
Video processing module for Sherlock project.

All temporal alignment uses TR = 1.5 seconds.
```

---

## Example Scripts

Example scripts can reference Sherlock, but should clarify it's an example:

### DO:
```python
"""
Training example for multimodal autoencoder.

This script demonstrates the training system.
Example uses the Sherlock dataset.
"""
```

### DON'T:
```python
"""
Training example for Sherlock autoencoder.
"""
```

---

## Configuration Files

### DO:
```yaml
# Training configuration for multimodal autoencoder
learning_rate: 1.0e-4
```

### DON'T:
```yaml
# Training configuration for Sherlock autoencoder
learning_rate: 1.0e-4
```

---

## Code Examples in Documentation

When showing examples with specific dimensions:

### DO:
```python
# Example with Sherlock dataset dimensions (90×160 video)
encoder = MultimodalEncoder(
    video_height=90,
    video_width=160,
    audio_mels=2048,
    n_voxels=85810
)
```

### DON'T:
```python
# Sherlock encoder
encoder = SherlockEncoder(
    video_height=90,
    video_width=160
)
```

---

## Dataset Paths in Code

Make paths configurable with Sherlock as the default:

### DO:
```python
def load_data(
    data_dir: Path,
    video_filename: str = 'stimuli_Sherlock.m4v',
    fmri_pattern: str = 'sherlock_movie_s*.nii.gz'
):
    """
    Load multimodal fMRI dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing data files
    video_filename : str, default='stimuli_Sherlock.m4v'
        Name of video stimulus file
    fmri_pattern : str, default='sherlock_movie_s*.nii.gz'
        Glob pattern for fMRI files
    """
    video_path = data_dir / video_filename
    # ...
```

### DON'T:
```python
def load_sherlock_data(data_dir: Path):
    """Load Sherlock dataset."""
    video_path = data_dir / 'stimuli_Sherlock.m4v'
    # Hard-coded, not configurable
```

---

## Comments About Data Dimensions

### DO:
```python
# Typical dimensions for naturalistic viewing datasets:
# - Video: 160×90×3 RGB frames
# - Audio: 2048 mel bins
# - Text: 1024-dim embeddings
# - fMRI: ~85,000 voxels, ~920 TRs
```

### DON'T:
```python
# Sherlock dataset dimensions:
# - Video: 160×90×3
# - fMRI: 920 TRs
```

---

## Test Class Names

### DO:
```python
class TestMultimodalEncoder:
    """Test multimodal encoder with example data."""

    def test_example_dataset_dimensions(self):
        """Test with example dataset dimensions (Sherlock)."""
        encoder = MultimodalEncoder(
            video_height=90,
            video_width=160,
            audio_mels=2048
        )
```

### DON'T:
```python
class TestSherlockEncoder:
    """Test Sherlock encoder."""

    def test_sherlock_dimensions(self):
        encoder = SherlockEncoder(
            video_height=90,
            video_width=160
        )
```

---

## Cache Filenames

Internal cache filenames can reference the dataset:

✅ **Acceptable:**
`cache_name = f"sherlock_{subjects_str}_{hrf_str}_{mode_str}.pkl"`

This is fine because:
1. It's internal implementation
2. It distinguishes cache files from different datasets
3. It doesn't leak into the public API

---

## Summary: When to Use "Sherlock"

| Context | Use Sherlock? | Example |
|---------|---------------|---------|
| Data filenames | ✅ Yes | `stimuli_Sherlock.m4v` |
| Test descriptions | ✅ Yes | "Tests with Sherlock data" |
| Documentation | ✅ Yes | "Example: Sherlock dataset" |
| Cache files | ✅ Yes | `sherlock_s1_hrf.pkl` |
| Class names | ❌ No | `MultimodalEncoder` not `SherlockEncoder` |
| Function names | ❌ No | `load_data()` not `load_sherlock()` |
| Module docstrings | ❌ No | "multimodal fMRI project" not "Sherlock project" |
| Configuration | ❌ No | "multimodal autoencoder" not "Sherlock autoencoder" |

---

## Migration Path

If you need to add support for a new dataset (e.g., "StudyForrest"):

1. Add new data files: `data/studyforrest_nii/*.nii.gz`
2. Use the same classes: `MultimodalDataset('data/', ...)`
3. Specify filenames: `video_filename='studyforrest_stimulus.mp4'`
4. Everything else works without changes

This is only possible because we kept the code stimulus-agnostic.

---

## Backward Compatibility

For existing code that imports `SherlockAutoencoder`:

```python
# In giblet/models/__init__.py
from .autoencoder import MultimodalAutoencoder

# Backward compatibility alias
SherlockAutoencoder = MultimodalAutoencoder

__all__ = [
    'MultimodalAutoencoder',
    'SherlockAutoencoder',  # Deprecated, use MultimodalAutoencoder
    # ...
]
```

This allows old code to work while encouraging migration to new names.

---

## Review Checklist

When adding new code or documentation, ask:

- [ ] Does this class/function name reference Sherlock?
  - If yes: Is it actually Sherlock-specific data?
- [ ] Does this docstring say "Sherlock dataset" instead of "example dataset"?
- [ ] Does this comment hard-code Sherlock dimensions?
  - Could I use this with a different dataset?
- [ ] Are my variable names generic (e.g., `video_path` not `sherlock_video`)?
- [ ] If I'm documenting an example, did I clarify it's an example?

---

**Last Updated:** 2025-10-29
**Related Issues:** #2, #9

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project exploring how individual brain regions "experience" stimuli differently from the complete subjective experience. The goal is to create transformation functions that modify stimuli (e.g., movies) to show only what specific brain regions "care about."

**Current Status**: Early brainstorming/prototyping phase. The repository currently contains only documentation; implementation has not yet begun.

**Original Context**: This repository originally housed code for the CCN Algonauts 2021 challenge but has since pivoted to this new research direction.

## Research Goals

1. Build transformation functions that filter stimuli through the "lens" of specific brain regions
   - Example: Primary visual cortex might remove audio and emphasize visual features
   - Example: Face processing areas might highlight or focus on faces
   - Example: Removing specific regions might blur or obscure relevant features

2. Key applications:
   - Understand the functional role of brain areas intuitively
   - Simulate effects of brain lesions or damage
   - Create optimized stimuli for targeting specific regions

## Proposed Technical Approach

The README outlines an autoencoder-based approach:

- **Architecture**:
  - Input/output layers: Vectorized video frames + audio
  - Middle layer: Compressed representation matching fMRI voxel count
  - Multiple intermediate layers connecting input/output to middle layer

- **Training approach**:
  - Optimize for both input/output matching AND middle layer matching fMRI responses
  - Handle temporal mismatches between video frame rate and fMRI sample rate (possibly via linear interpolation)
  - Consider BOLD response temporal blurring

- **Applications**:
  - Encoder: Predict brain responses from new input images
  - Decoder: Predict video/audio from brain responses
  - Lesion simulation: Fix certain middle layer "voxels" to 0 and observe decoder output changes

**Alternative approaches mentioned**: Hyperalignment-based decoding, GANs, TFA/timecorr features, integration with other datasets (Huth et al., Neurosynth)

## Dataset: Sherlock fMRI Study

The project uses a dataset of 16 participants who watched and then recalled an episode of the TV show "Sherlock".

### Dataset Structure (BIDS Format)

Located in `data/` directory:
- **16 subjects** (sub-01 through sub-16)
- **fMRI scan types per subject**:
  - `task-sherlockPart1`: First half of Sherlock episode viewing
  - `task-sherlockPart2`: Second half of Sherlock episode viewing
  - `task-freerecall`: Free recall session after viewing
  - Additional tasks: `task-stopsignal`, `task-conditionalstopsignal`

### Data Organization per Subject

Each subject directory contains:

**anat/** - Anatomical data:
- Preprocessed T1w images (`desc-preproc_T1w.nii.gz`)
- Brain masks (`desc-brain_mask.nii.gz`)
- Tissue segmentations (`dseg.nii.gz`, CSF/GM/WM probability maps)
- Native space and MNI152NLin2009cAsym space versions
- Transformation files (`xfm.h5`) for space conversions

**func/** - Functional data:
- Raw preprocessed BOLD: `task-{task}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- Denoised & smoothed (6mm): `denoise_smooth6mm_task-{task}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- Cropped versions (for Sherlock tasks): `denoise_crop_smooth6mm_task-{task}*.nii.gz`
- HDF5 format: `.hdf5` files (alternative format for cropped Sherlock data)
- **ROI-averaged data**: `{sub}_Part1_Average_ROI_n50.csv` and `Part2` (50 ROIs per part)
- Brain masks: `desc-brain_mask.nii.gz`
- Reference images: `boldref.nii.gz`
- **Confound regressors**: `desc-confounds_regressors.tsv/.json` (motion, physiological noise, etc.)

**figures/** - Quality control visualizations (SVG):
- Carpet plots, CompCor variance plots
- Confound correlation matrices
- Registration quality checks (FLIRT BBR)
- ROI overlay visualizations

**log/** - Processing logs and crash reports from preprocessing pipeline

### Key Files for Analysis

1. **Primary fMRI data** (per subject, per task):
   - Full voxel-level: `denoise_smooth6mm_task-sherlockPart{1,2}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
   - ROI-averaged: `{sub}_Part{1,2}_Average_ROI_n50.csv`

2. **Stimulus**:
   - Video file: `data/stimuli_Sherlock.m4v` (Sherlock episode used as stimulus)
   - Split into two parts in fMRI data (Part1 and Part2)

3. **Confounds**: Task-specific confound regressors for denoising

### Data Access for Lab Members

**Simple one-command setup:**

```bash
./download_data_from_dropbox.sh
```

This downloads a single 11GB zip file from Dropbox containing:
- Stimulus video (272 MB)
- Scene annotations (173 KB)
- fMRI data for all subjects (17 .nii.gz files, ~10.7 GB)

**No credentials needed** - just run the script and it will download and extract everything automatically.

**Alternative: Download from Discovery server** (if you have access):

```bash
./download_data.sh
```

Requires Discovery credentials and `sshpass` installed.

**What you'll get:**
- `data/stimuli_Sherlock.m4v` - Sherlock episode video
- `data/annotations.xlsx` - Scene-level annotations (1000 segments with timing, characters, locations, emotions)
- `data/sherlock_nii/*.nii.gz` - fMRI data for 17 subjects

**Dataset URL**: https://www.dropbox.com/scl/fi/tlr6orirwc14gdq7yqwcl/sherlock_dataset.zip?rlkey=82h9hyrbv37xvtqff5mw6h6m5&dl=1

### Data Storage for Lab-Only Sharing

This dataset cannot be publicly shared (no permissions for stimulus video, borrowed fMRI data). For lab member access:

1. **Code & metadata**: GitHub repository (this repo) - public or private
2. **Large data files**: Discovery server (requires Dartmouth NetID)
3. **Alternative**: DataLad dataset with Dropbox backend (for team members without Discovery access)

**Note**: Do not commit .nii or .nii.gz files to GitHub. Data directory is in .gitignore.

### Spatial Normalization

All functional data is in **MNI152NLin2009cAsym** standard space, enabling:
- Cross-subject comparisons
- ROI definition using standard atlases
- Integration with other datasets in MNI space

## Development Environment

This is a Python-based neuroimaging project requiring:

### Core Dependencies (anticipated)
- **Neuroimaging**: `nibabel` (reading .nii.gz files), `nilearn` (fMRI analysis), `h5py` (HDF5 files)
- **Deep Learning**: PyTorch or TensorFlow (for autoencoder implementation)
- **Data Science**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Video Processing**: `opencv-python`, `decord`, or `ffmpeg-python` (for processing Sherlock stimulus)
- **Analysis**: Jupyter notebooks for exploration and visualization

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt
```

### Data Format Notes
- **NIfTI files** (.nii.gz): Standard neuroimaging format, 4D arrays (x, y, z, time)
- **HDF5 files** (.hdf5): Alternative storage for large arrays
- **CSV files**: ROI-averaged timeseries (50 ROIs × timepoints)
- **TSV files**: Confound regressors (motion parameters, CompCor components, etc.)

## Team Communication

- Active Slack channel: https://context-lab.slack.com/archives/C020V4HJFT4
- GitHub used for tracking code contributions and ideas
- Team members expected to participate in hackathons and maintain regular communication

## Key References

- Related work on hyperalignment-based decoding: https://arxiv.org/abs/1701.08290
- Sherlock dataset: Multi-subject fMRI study with naturalistic viewing and recall paradigm
  - Contains watch and recall data for 16 subjects
  - Preprocessed with fMRIPrep pipeline
  - Standard BIDS format for neuroimaging data

# Giblet Multimodal Validation Suite

Comprehensive round-trip validation for all modality processors.

## Overview

This directory contains validation scripts that test the complete processing pipeline for each modality using **REAL Sherlock data** (no mocks).

All scripts:
- Use actual data files from `data/` directory
- Perform round-trip validation (data → features → reconstructed data)
- Calculate quality metrics
- Generate output files for manual inspection
- Save visualizations and comparisons

## Validation Scripts

### 1. `validate_video.py` - Video Modality

Tests video frame extraction and reconstruction.

**What it validates:**
- Frame extraction from Sherlock video
- Temporal alignment to fMRI TRs (1.5s bins)
- Video reconstruction from features
- PSNR quality metric

**Outputs:**
- Frame comparison images (PNG)
- Side-by-side comparison videos (MP4)
- PSNR measurements

**Expected results:**
- PSNR > 30 dB (excellent)
- PSNR > 20 dB (good)

**Runtime:** ~10-15 minutes

### 2. `validate_audio.py` - Audio Modality

Tests audio extraction and reconstruction using Griffin-Lim.

**What it validates:**
- Audio extraction from video
- Mel spectrogram computation
- Temporal alignment to TRs
- Audio reconstruction from mel spectrograms

**Outputs:**
- Original audio segments (.WAV)
- Reconstructed audio segments (.WAV)
- Waveform comparison plots (PNG)
- Spectrogram comparison plots (PNG)

**Expected results:**
- Correlation: 0.4-0.8
- SNR: 5-15 dB
- Recognizable speech/music

**IMPORTANT:** Listen to the .WAV files to verify quality!

**Known limitations:**
- Griffin-Lim reconstruction is imperfect (no phase information)
- Audio may sound "phasey" or "robotic"
- This is expected for mel spectrogram approach

**Runtime:** ~8-12 minutes

### 3. `validate_text.py` - Text Modality

Tests text embedding and semantic recovery.

**What it validates:**
- Annotation loading (1000 segments)
- Text embedding with BAAI/bge-large-en-v1.5
- Temporal alignment to TRs
- Nearest-neighbor text recovery
- Semantic consistency

**Outputs:**
- Text recovery examples (.TXT)
- Top-k accuracy metrics
- Semantic similarity analysis

**Expected results:**
- Top-1 accuracy: >50%
- Top-5 accuracy: >80%
- Coverage: >70% of TRs
- Consecutive TRs more similar than random pairs

**Runtime:** ~3-5 minutes (first run downloads 1.3GB model)

### 4. `validate_fmri.py` - fMRI Modality

Tests fMRI data extraction and reconstruction.

**What it validates:**
- Shared mask creation across subjects
- Feature extraction from .nii.gz files
- NIfTI reconstruction from features
- Exact match verification
- Voxel timeseries preservation

**Outputs:**
- Reconstructed NIfTI files (.nii.gz)
- Voxel timeseries plots (PNG)
- Brain slice visualizations (PNG)
- Shared brain mask file

**Expected results:**
- Exact match (within 1e-6 tolerance)
- Correlation: >0.99
- All voxel values preserved

**Runtime:** ~5-8 minutes

### 5. `validate_all_modalities.py` - Master Script

Runs all validation scripts and generates comprehensive report.

**What it does:**
- Runs all 4 modality validations sequentially
- Tracks runtime for each modality
- Catches and reports errors
- Generates consolidated report

**Outputs:**
- `validation_report.txt` - Human-readable summary
- `validation_results.json` - Machine-readable results
- All individual modality outputs

**Runtime:** ~25-40 minutes total

## Usage

### Quick Start

Run all validations:
```bash
cd examples
python validate_all_modalities.py
```

### Individual Modalities

Run specific validation:
```bash
python validate_video.py
python validate_audio.py
python validate_text.py
python validate_fmri.py
```

### Requirements

Ensure data is downloaded:
```bash
cd ..
./download_data_from_dropbox.sh
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python (video processing)
- librosa, soundfile (audio processing)
- sentence-transformers (text embeddings)
- nibabel (fMRI data)
- scikit-image (PSNR calculation)
- matplotlib (visualizations)

## Output Directory

All outputs are saved to `validation_outputs/`:

```
validation_outputs/
├── video_comparison_*.png          # Frame comparisons
├── video_comparison_*.mp4          # Side-by-side videos
├── audio_original_*.wav            # Original audio
├── audio_reconstructed_*.wav       # Reconstructed audio
├── audio_waveform_*.png            # Waveform plots
├── audio_spectrogram_*.png         # Spectrogram plots
├── text_recovery_examples.txt      # Text recovery results
├── fmri_reconstructed_*.nii.gz     # Reconstructed brain images
├── fmri_timeseries_*.png           # Voxel timeseries
├── fmri_brain_slices_*.png         # Brain visualizations
├── shared_brain_mask.nii.gz        # Brain mask
├── validation_report.txt           # Summary report
└── validation_results.json         # JSON results
```

## Quality Thresholds

### Video
- **Excellent:** PSNR > 30 dB
- **Good:** PSNR > 20 dB
- **Acceptable:** PSNR > 15 dB

### Audio
- **Good:** Correlation > 0.6, SNR > 10 dB
- **Acceptable:** Correlation > 0.4, SNR > 5 dB
- **Note:** Griffin-Lim has inherent limitations

### Text
- **Good:** Top-1 > 50%, Top-5 > 80%
- **Acceptable:** Top-1 > 30%, Top-5 > 60%
- **Coverage:** Should be > 70%

### fMRI
- **Perfect:** Exact match (all voxels)
- **Excellent:** Correlation > 0.99
- **Good:** Correlation > 0.95

## Interpreting Results

### Success Criteria

A modality passes validation if:
1. Script runs without errors
2. Quality metrics meet thresholds
3. Output files are generated correctly
4. Manual inspection confirms quality

### Manual Inspection

**CRITICAL for audio:** Listen to .WAV files to verify:
- Speech is recognizable
- Music preserves melody
- No severe artifacts

**Important for video:** Watch comparison videos:
- Original and reconstructed are visually similar
- Temporal alignment is correct
- No major artifacts

**Important for text:** Read recovery examples:
- Recovered texts are semantically similar to originals
- Top-k accuracy makes sense
- Temporal coherence is preserved

**Important for fMRI:** Check brain visualizations:
- Mask covers brain regions properly
- Timeseries are smooth and reasonable
- No obvious artifacts

## Troubleshooting

### "Data not found" errors
```bash
./download_data_from_dropbox.sh
```

### "Model not found" errors
Text validation downloads BAAI/bge-large-en-v1.5 (~1.3GB) on first run. Ensure internet connection.

### Memory errors
Validation scripts use real data and can be memory-intensive:
- Video: ~2-4 GB RAM
- Audio: ~2-3 GB RAM
- Text: ~3-4 GB RAM (model)
- fMRI: ~4-6 GB RAM

Close other applications if needed.

### CUDA/GPU issues
All processors work on CPU. GPU is optional and will be auto-detected if available.

## Technical Details

### Round-Trip Validation Methodology

For each modality:
1. **Load** original data (video, audio, annotations, .nii.gz)
2. **Extract** features (downsampled frames, mel spectrograms, embeddings, voxel timeseries)
3. **Align** features to fMRI TRs (1.5s temporal bins)
4. **Reconstruct** data from features (video, audio, text, .nii.gz)
5. **Validate** reconstruction quality (PSNR, SNR, correlation, exact match)
6. **Save** outputs for manual inspection

### Why No Mocks?

These validations use **REAL data only** because:
- Mocks hide implementation bugs
- Real data reveals edge cases
- Manual inspection requires actual files
- Validates entire pipeline end-to-end
- Matches actual usage patterns

### Data Specifications

- **Video:** 640×360 @ 23.976 fps → 160×90 @ TR=1.5s
- **Audio:** Stereo 44.1kHz → Mono 22.05kHz, 2048 mel bins
- **Text:** 1000 segments → ~950 TRs, 1024-dim embeddings
- **fMRI:** 17 subjects, ~83k voxels, 920 TRs

## Next Steps

After validation:
1. Review `validation_report.txt` for summary
2. Check individual output files in `validation_outputs/`
3. Listen to audio .WAV files
4. Watch video comparison files
5. Read text recovery examples
6. Inspect brain visualizations

If all validations pass:
✅ The giblet multimodal pipeline is working correctly
✅ Ready for model training

If some fail:
⚠ Review error messages in report
⚠ Check data integrity
⚠ Verify dependencies installed
⚠ Inspect generated outputs for clues

## Related

- Main tests: `tests/data/test_*.py`
- Processor implementations: `giblet/data/*.py`
- Dataset: `giblet/data/dataset.py`
- Data download: `download_data_from_dropbox.sh`

## Issue Tracking

This validation suite implements Issue #13 (part of Master Issue #20):
- Comprehensive round-trip validation
- All modalities tested
- Real data only
- Manual inspection enabled
- Output files for verification

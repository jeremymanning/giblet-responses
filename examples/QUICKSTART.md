# Validation Quick Start Guide

## Prerequisites

```bash
# 1. Download data (if not already done)
./download_data_from_dropbox.sh

# 2. Ensure you're in project root
cd /Users/jmanning/giblet-responses

# 3. Activate virtual environment (if using one)
# source venv/bin/activate
```

## Run All Validations

```bash
# Run from examples directory
cd examples
python validate_all_modalities.py
```

**Runtime:** ~25-40 minutes

**Outputs:**
- `validation_outputs/validation_report.txt` - Read this first!
- `validation_outputs/` - All generated files

## Run Individual Validations

```bash
# Video only (~10-15 min)
python validate_video.py

# Audio only (~8-12 min)
python validate_audio.py

# Text only (~3-5 min)
python validate_text.py

# fMRI only (~5-8 min)
python validate_fmri.py
```

## What to Check

### 1. Console Output
Look for:
- ✅ "VALIDATION PASSED" messages
- Quality metrics (PSNR, SNR, correlation, accuracy)
- No error messages

### 2. Video Outputs
```bash
ls validation_outputs/video_*
```
- Watch `.mp4` comparison videos
- Check frame comparison `.png` images
- Verify PSNR > 20 dB

### 3. Audio Outputs
```bash
ls validation_outputs/audio_*.wav
```
- **CRITICAL:** Listen to .WAV files!
- Compare original vs reconstructed
- Check waveform and spectrogram plots

### 4. Text Outputs
```bash
cat validation_outputs/text_recovery_examples.txt
```
- Read recovery examples
- Check top-k accuracy (should be >50% for top-1)
- Verify coverage >70%

### 5. fMRI Outputs
```bash
ls validation_outputs/fmri_*
```
- Check brain slice visualizations
- Verify timeseries plots look reasonable
- Correlation should be >0.99

## Success Criteria

✅ **All validations pass if:**
- Scripts run without errors
- Metrics meet thresholds (see README_VALIDATION.md)
- Generated files exist and are reasonable
- Manual inspection confirms quality

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data/..."
```bash
./download_data_from_dropbox.sh
```

### "CUDA out of memory"
All processors work on CPU. GPU is optional.

### "Text validation taking long"
First run downloads 1.3GB model. Wait ~5 minutes.

## Results Location

All outputs saved to:
```
validation_outputs/
├── validation_report.txt    ← READ THIS FIRST
├── validation_results.json  ← Machine-readable
├── video_comparison_*.mp4   ← Watch these
├── audio_*.wav              ← Listen to these
├── text_recovery_*.txt      ← Read these
└── fmri_*.png               ← View these
```

## Next Steps After Validation

1. Read `validation_report.txt`
2. Review quality metrics
3. Inspect generated files
4. If all pass: ✅ Ready for model training
5. If some fail: Check error messages and outputs

## Full Documentation

- **Detailed guide:** `README_VALIDATION.md`
- **Implementation summary:** `../VALIDATION_IMPLEMENTATION_SUMMARY.md`
- **Issue tracking:** GitHub Issue #13

## Quick Commands

```bash
# Run everything
python validate_all_modalities.py

# Check results
cat validation_outputs/validation_report.txt

# List all outputs
ls -lh validation_outputs/

# Count files generated
ls validation_outputs/ | wc -l

# Check disk space used
du -sh validation_outputs/
```

## Expected Results

| Modality | Metric | Expected |
|----------|--------|----------|
| Video | PSNR | >30 dB |
| Audio | Correlation | 0.4-0.8 |
| Audio | SNR | 5-15 dB |
| Text | Top-1 Accuracy | >50% |
| Text | Top-5 Accuracy | >80% |
| fMRI | Correlation | >0.99 |
| fMRI | Exact Match | Yes |

## Contact

Questions? Check:
1. `README_VALIDATION.md` (comprehensive guide)
2. GitHub Issue #13
3. Context Lab Slack: https://context-lab.slack.com/archives/C020V4HJFT4

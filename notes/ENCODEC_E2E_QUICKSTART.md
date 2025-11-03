# EnCodec E2E Pipeline Test - Quick Start Guide

**Issue #24, Task 3.3** - End-to-end pipeline test for EnCodec integration

---

## Quick Run

```bash
cd /Users/jmanning/giblet-responses
python test_encodec_e2e_pipeline.py
```

**First run:** 10-20 minutes (includes model download)
**Subsequent runs:** 2-5 minutes (model cached)

---

## What It Does

Tests complete pipeline: **audio → EnCodec codes → encoder → bottleneck → decoder → codes → audio**

**Tests 30 TRs (45 seconds)** of Sherlock audio with:
- 12kHz sampling (downsampled from 24kHz)
- 3.0 kbps bandwidth
- User-approved configuration

---

## Output Files

All files saved to: `encodec_e2e_test/`

### For Manual Verification (listen to these):
1. `original_12khz.wav` - Original audio
2. `baseline_encodec_12khz.wav` - EnCodec direct (baseline)
3. `reconstructed_12khz.wav` - Through encoder/decoder pipeline

### Quality Analysis:
- `metrics_comparison.txt` - SNR, PESQ, STOI metrics
- `spectrograms_comparison.png` - Visual comparison

### Additional Files:
- `original_30trs.wav`, `baseline_encodec_direct.wav`, `reconstructed_through_bottleneck.wav` (24kHz versions)

---

## Success Criteria

✅ **Pipeline runs without errors**
✅ **All dimension checks pass**
✅ **STOI drop < 0.1** (minimal quality degradation)
✅ **Memory < 1GB peak**
✅ **Audio files sound reasonable**

---

## Interpreting Results

### Quality Metrics (in `metrics_comparison.txt`):

**Baseline (EnCodec direct):**
- SNR: 8-10 dB (expected)
- PESQ: 3.4-3.6 (good quality)
- STOI: 0.85-0.90 (highly intelligible)

**Reconstructed (through pipeline):**
- SNR: 6-9 dB (acceptable if > 6)
- PESQ: 3.0-3.4 (good if > 3.0)
- STOI: 0.75-0.85 (intelligible if > 0.75)

**Quality degradation:**
- STOI drop: <0.1 ✅ (target met)
- PESQ drop: <0.5 (acceptable)
- SNR drop: <3 dB (acceptable)

---

## Common Issues

### "Downloading model..." takes >5 minutes
**Normal!** First-time download of `facebook/encodec_24khz` model from HuggingFace Hub.
Subsequent runs will use cached model and be much faster.

### Out of memory
Reduce `n_trs` from 30 to 10 in `CONFIG` at top of script.

### PESQ/STOI calculation fails
Verify dependencies: `pip install pesq pystoi`

---

## Next Steps After Test Completes

1. **Listen to WAV files** - Verify audio quality is acceptable
2. **Check metrics** - Verify STOI drop < 0.1
3. **Review spectrograms** - Visual inspection of quality
4. **Document results** - Update Issue #24 with findings
5. **Proceed to integration** - Update AudioProcessor, Dataset, models

---

## Integration Plan

### After test passes:
1. Update `giblet/data/audio.py` for 12kHz mode
2. Update `AudioEncoder` for EnCodec code embedding
3. Update `MultimodalDecoder` for code prediction
4. Precompute EnCodec codes for full dataset
5. Train autoencoder with new features
6. Compare quality with mel spectrogram baseline

---

## Questions?

See full documentation: `ENCODEC_E2E_TEST_SUMMARY.md`

---

**Status:** ✅ TEST READY TO RUN

**Created:** 2025-11-01

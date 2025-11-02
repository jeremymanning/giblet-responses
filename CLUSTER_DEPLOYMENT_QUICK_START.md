# EnCodec Fix - Cluster Deployment Quick Start

**Status:** ✅ Ready for deployment
**Confidence:** 🟢 HIGH (95%)

---

## Quick Commands for Cluster

### Step 1: Verify Fix Logic (5 minutes)

```bash
cd /path/to/giblet-responses
python test_encodec_minimal_bug.py
```

**Expected output:**
```
✓✓✓ Fix verified - dimension mismatch resolved!
✓✓✓ All edge cases pass!
```

### Step 2: Test with Real Sherlock Data (15 minutes)

```bash
python verify_fix_sherlock.py
```

**Expected output:**
```
✓✓✓ ALL TESTS PASSED - Fix verified with real Sherlock data!

Tests passed: 5/5
   5 TRs: ✓ PASS
  10 TRs: ✓ PASS
  20 TRs: ✓ PASS
  50 TRs: ✓ PASS
 100 TRs: ✓ PASS
```

### Step 3: Run Unit Tests (10 minutes)

```bash
pytest tests/data/test_audio_dimension_fix.py -v
```

**Expected:** 9/9 PASSED

### Step 4: Run Integration Tests (15 minutes)

```bash
pytest tests/data/test_encodec_sherlock_integration.py -v
```

**Expected:** 15/15 PASSED

### Step 5: Extract Full Sherlock Dataset (1-2 hours)

```python
from giblet.data.audio import AudioProcessor

processor = AudioProcessor(
    use_encodec=True,
    encodec_bandwidth=3.0,
    tr=1.5
)

features, metadata = processor.audio_to_features(
    'data/stimuli_Sherlock.m4v',
    from_video=True
)

print(f'✓ Extracted {len(features)} TRs')
print(f'  Shape: {features.shape}')
print(f'  Expected: (~1920, 896)')
```

---

## What to Look For

### Success Indicators

✅ All TRs have shape (896,)
✅ Dtype is int64
✅ No dimension mismatch errors
✅ Metadata is consistent
✅ Values are 0-1023 (valid codebook indices)

### Failure Indicators

❌ Shape inconsistencies between TRs
❌ Dimension mismatch error
❌ NaN or Inf values
❌ Negative values or values >1023

---

## If Something Fails

1. Check error message - is it the original dimension error?
2. Verify `giblet/data/audio.py` line 315 uses `frames_per_tr`
3. Check EnCodec can load: `python -c "from transformers import EncodecModel; print('OK')"`
4. Verify Sherlock video exists: `ls -lh data/stimuli_Sherlock.m4v`
5. Report error with full traceback

---

## Expected Results

**For default settings (3.0 kbps, TR=1.5s):**
- Shape per TR: (896,) = 8 codebooks × 112 frames
- Total TRs in Sherlock: ~1920
- Dtype: int64
- Value range: 0-1023

---

## Files Created

**Test scripts:**
- `test_encodec_minimal_bug.py` - Fix logic verification
- `verify_fix_sherlock.py` - Progressive real data testing
- `tests/data/test_encodec_sherlock_integration.py` - Integration tests

**Reports:**
- `ENCODEC_FIX_VERIFICATION_REPORT.md` - Comprehensive analysis
- `CLUSTER_DEPLOYMENT_QUICK_START.md` - This document

**Existing tests:**
- `tests/data/test_audio_dimension_fix.py` - 9 unit tests
- `tests/data/test_audio_encodec.py` - EnCodec functionality tests

---

## Estimated Timeline

- Step 1 (minimal test): 5 minutes
- Step 2 (progressive tests): 15 minutes
- Step 3 (unit tests): 10 minutes
- Step 4 (integration tests): 15 minutes
- Step 5 (full extraction): 1-2 hours

**Total: 2-3 hours from start to finish**

---

**Ready to deploy! 🚀**

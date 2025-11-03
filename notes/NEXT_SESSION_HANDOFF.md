# Next Session Handoff - Temporal Concatenation Implementation

## Current Situation

**Training Status:** Blocked by EnCodec dimension mismatch
**Solution:** Implement temporal window concatenation approach
**Issues Created:** #26 (implementation), #27 (local testing)

---

## What to Do Next

### 🎯 Priority 1: Fix Temporal Encoding (Issue #26)

**New approach:** Concatenate features from temporal windows instead of complex encoding

**Implementation in 3 batches:**

**Batch 1** (4-6 hours): Core temporal concatenation
- Task 1.1: Video concatenation (video.py)
- Task 1.2: Audio concatenation (audio.py) 
- Task 1.3: Text concatenation (text.py)
- Use 3 parallel agents

**Batch 2** (2-3 hours): Configurability
- Task 2.1: Make TR length configurable
- Task 2.2: Make window size configurable
- Task 2.3: Implement time-shift instead of HRF
- Use 3 parallel agents

**Batch 3** (3-4 hours): Integration
- Task 3.1: Update dataset.py
- Task 3.2: Update models
- Task 3.3: Update configs
- Use 3 parallel agents

### 🧪 Priority 2: Local Testing (Issue #27)

**Before deploying to cluster:**
1. Create test_local_training.py
2. Load 1 subject, 20 TRs
3. Train for 5 iterations
4. Verify all dimensions match
5. Verify no errors

**Only after local test passes:** Deploy to cluster

---

## Key Implementation Details

### Temporal Window Approach

```python
# For TR at time t, window is [t-1.5s, t]
# Video: 37 frames @ 25fps
# Audio: 112 EnCodec frames @ 12kHz
# Text: 2-3 annotations

# Stack into fixed-size tensor
features[tr] = concat(window_features)
```

### Parameters to Add

```yaml
data:
  tr_length: 1.5  # Configurable for other datasets
  temporal_window: 1.0  # In TR units (future: try 3.0, 6.7, etc.)
  fmri_shift: 6.0  # Time shift instead of HRF blur
```

### Time-Shift Strategy

```python
# Instead of: fMRI_features = convolve_with_hrf(stimulus_features)
# Use: fMRI_features[t] = stimulus_features[t - 4 TRs]  # 6 second shift
```

---

## Command to Resume

```bash
# Start fresh session
# Implement Issue #26 in batches (use parallel agents)
# Test with Issue #27 (local 5 iterations)
# Deploy to cluster if tests pass
```

---

## Files to Reference

**Plans:**
- Issue #26: Temporal concatenation implementation
- Issue #27: Local testing checklist

**Current Code:**
- giblet/data/audio.py (needs fixing)
- giblet/data/video.py (needs updating)
- giblet/data/text.py (needs updating)
- giblet/alignment/sync.py (remove HRF, add time-shift)

**Documentation:**
- TRAINING_STATUS.md (current blocker)
- FINAL_AUTONOMOUS_SESSION_REPORT.md (session summary)

---

## Success Path

1. ✅ Implement temporal concatenation (Issue #26)
2. ✅ Test locally for 5 iterations (Issue #27)
3. ✅ Deploy to cluster once tests pass
4. 🎯 Model training begins!

---

**Context saved at 563K / 1M (56%)** - Fresh session recommended for implementation.

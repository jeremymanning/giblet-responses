# Audio Encoding Research: Executive Summary

**Date:** 2025-10-31
**Issue:** #23 - Audio reconstruction quality still poor despite temporal preservation
**Status:** Research complete, ready for implementation decision

---

## Problem Statement

The current audio reconstruction pipeline produces **garbled, unintelligible output** because:

1. **Mel spectrograms discard phase information** (only magnitude is preserved)
2. **Griffin-Lim algorithm attempts to "guess" phase** through iterative reconstruction
3. **After bottleneck compression** (through fMRI autoencoder), phase reconstruction becomes impossible
4. **Result:** Severely degraded audio quality

---

## Recommended Solution: EnCodec

**Replace mel spectrograms with Meta's EnCodec neural audio codec.**

### Why EnCodec?

| Criterion | Status | Details |
|-----------|--------|---------|
| **Solves phase problem** | ✓ | Implicitly preserves phase through learned compression |
| **Ready to use** | ✓ | Pretrained models available (24kHz mono, 48kHz stereo) |
| **Efficient** | ✓ | 10M parameters, 3.6 KB per TR (vs. 51 KB for mel specs) |
| **Proven** | ✓ | Production use in Meta's AudioGen, MusicGen |
| **Easy integration** | ✓ | Drop-in replacement for current mel spectrogram pipeline |
| **Research-friendly** | ✓ | MIT license, no restrictions |

### Expected Improvements

| Metric | Current (Mel + Griffin-Lim) | With EnCodec | Improvement |
|--------|---------------------------|--------------|-------------|
| **SNR** | ~5 dB (poor) | >15 dB (good) | +10 dB |
| **PESQ** | ~1.5 (bad) | >3.0 (acceptable-good) | +1.5 points |
| **STOI** | ~0.3 (unintelligible) | >0.7 (intelligible) | +0.4 points |
| **Subjective** | Garbled, unintelligible | Clear, perceptually plausible | ⭐⭐⭐ |

---

## Implementation Plan

### Timeline: 3-4 Weeks

| Phase | Duration | Key Tasks | Deliverable |
|-------|----------|-----------|-------------|
| **1. Proof of Concept** | 1-2 days | Test EnCodec on Sherlock audio, evaluate quality | Quality metrics, audio samples |
| **2. Integration** | 3-5 days | Precompute codes, update dataset, modify encoder/decoder | Updated dataset + model code |
| **3. Training** | 1 week | Train autoencoder on EnCodec codes | Trained model checkpoint |
| **4. Evaluation** | 2-3 days | Comprehensive quality assessment, comparisons | Evaluation report, demos |

### Key Milestones

- **Week 1:** EnCodec quality validated, precomputed codes ready
- **Week 2:** Autoencoder integration complete, training started
- **Week 3:** Model trained, initial evaluation complete
- **Week 4:** Final evaluation, comparison with baseline, documentation

---

## Technical Overview

### Current Pipeline
```
Audio Waveform
    ↓
Mel Spectrogram (magnitude only) ← PHASE LOST
    ↓
Encoder → fMRI Bottleneck → Decoder
    ↓
Mel Spectrogram (predicted)
    ↓
Griffin-Lim (phase guessing) ← RECONSTRUCTION FAILS
    ↓
Garbled Audio
```

### EnCodec Pipeline
```
Audio Waveform
    ↓
EnCodec Encoder → Compressed Codes (phase implicitly preserved)
    ↓
Autoencoder Encoder → fMRI Bottleneck → Autoencoder Decoder
    ↓
Predicted Codes
    ↓
EnCodec Decoder ← PERFECT RECONSTRUCTION (if codes accurate)
    ↓
High-Quality Audio
```

### Integration Options

**Option A: EnCodec as Preprocessor (RECOMMENDED)**
- Freeze EnCodec (pretrained weights)
- Train autoencoder to predict EnCodec codes
- Pros: Simple, fast, proven to work
- Cons: Not optimized for specific fMRI bottleneck

**Option B: Fine-tune EnCodec End-to-End**
- Replace EnCodec quantization with fMRI projection
- Fine-tune entire pipeline
- Pros: Optimized for task, potentially better quality
- Cons: More complex, requires more compute/time

**Recommendation:** Start with Option A, consider Option B if quality insufficient.

---

## Resource Requirements

### Computational
- **Training:** 1 GPU (existing setup sufficient)
- **Memory:** <2GB GPU RAM (codes are very compact)
- **Time:** ~1 week training (similar to current approach)

### Storage
- **Model weights:** 80 MB (EnCodec) + existing autoencoder weights
- **Precomputed codes:** 3.3 MB for all 920 TRs (tiny!)
- **Total:** <100 MB additional storage

### Dependencies
```bash
pip install encodec  # Only new dependency
```

---

## Alternative Approaches Considered

| Approach | Rating | Why Not Primary Choice |
|----------|--------|------------------------|
| **Complex FFT** | ⭐⭐⭐⭐ | Requires training, 2x parameters, no pretrained models |
| **Raw Waveform** | ⭐⭐⭐ | Very high memory/parameters (50M+), slow training |
| **Wav2Vec 2.0** | ⭐⭐⭐ | No decoder (encoder-only), need to build vocoder |
| **Audio-as-Image** | ⭐⭐⭐ | Phase representation tricky, requires training |
| **CQT** | ⭐⭐ | Still discards phase (same problem as mel specs) |
| **WaveGlow** | ⭐⭐ | 87M parameters, very slow, not designed for compression |

**Note:** Complex FFT is a strong backup if EnCodec doesn't work.

---

## Risks & Mitigation

### Risk 1: EnCodec quality insufficient after bottleneck compression
- **Likelihood:** Low (EnCodec designed for extreme compression)
- **Mitigation:** Test quality in Phase 1 before full integration
- **Backup:** Use Complex FFT approach (detailed in full report)

### Risk 2: Integration more complex than anticipated
- **Likelihood:** Low (similar architecture to existing mel spec pipeline)
- **Mitigation:** Incremental integration, test each component
- **Backup:** Minimal code changes needed, can revert easily

### Risk 3: Training time longer than expected
- **Likelihood:** Medium (new data format may need hyperparameter tuning)
- **Mitigation:** Start with existing hyperparameters, adjust as needed
- **Impact:** Timeline extends 1-2 weeks (acceptable)

---

## Success Criteria

### Minimum Viable (Must Achieve)
- ✓ SNR > 10 dB (vs. ~5 dB current)
- ✓ PESQ > 2.0 (acceptable quality)
- ✓ STOI > 0.6 (mostly intelligible)
- ✓ Subjective: Speech is understandable

### Target (Goal)
- ✓ SNR > 15 dB
- ✓ PESQ > 3.0 (good quality)
- ✓ STOI > 0.7 (intelligible)
- ✓ Subjective: Perceptually plausible music/sound effects

### Stretch (Ideal)
- ✓ SNR > 20 dB
- ✓ PESQ > 3.5 (very good quality)
- ✓ STOI > 0.8 (highly intelligible)
- ✓ Subjective: Barely distinguishable from original

---

## Next Steps

### Immediate Actions (This Week)
1. **Install EnCodec:** `pip install encodec`
2. **Test on Sherlock audio:**
   - Encode/decode full stimulus
   - Compute SNR, PESQ, STOI
   - Subjective listening test
3. **Decision point:** Proceed with full integration?

### If Approved (Next Week)
4. **Precompute EnCodec codes** for all 920 TRs
5. **Update dataset class** to return codes instead of mel specs
6. **Modify encoder/decoder** for new input/output shape
7. **Start training** with new data format

### Follow-up (Weeks 3-4)
8. **Monitor training** (convergence, quality metrics)
9. **Evaluate trained model** (comprehensive quality assessment)
10. **Compare with baseline** (mel + Griffin-Lim)
11. **Document results** (update Issue #23, write summary)
12. **Team demo** (audio samples, metrics, visualizations)

---

## References & Documentation

### Created Documents
1. **Full Research Report:** `/notes/AUDIO_ENCODING_RESEARCH_REPORT.md`
   - 1429 lines, comprehensive analysis of 7 approaches
   - Technical details, implementation guides, references

2. **GitHub Issue Comment:** `/notes/AUDIO_ENCODING_ISSUE23_POST.md`
   - 274 lines, concise summary for team discussion
   - Ready to post to Issue #23

3. **Quick Reference:** `/notes/AUDIO_ENCODING_QUICK_REFERENCE.md`
   - Code snippets, metrics, lookup tables

4. **This Summary:** `/AUDIO_ENCODING_EXECUTIVE_SUMMARY.md`
   - High-level overview for decision-making

### Key Papers
- **EnCodec:** ["High Fidelity Neural Audio Compression"](https://arxiv.org/abs/2210.13438) (Défossez et al., 2022)
- **fMRI Audio Reconstruction:** ["Natural sounds reconstructed from fMRI"](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003293) (PLOS Biology, 2024)
- **Complex CNNs:** ["Phase-Aware Deep Learning with Complex-Valued CNNs"](https://arxiv.org/abs/2510.09926) (2024)

### External Resources
- **GitHub:** [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **Demo:** [audiocraft.metademolab.com/encodec](https://audiocraft.metademolab.com/encodec.html)
- **License:** MIT (no restrictions for research use)

---

## Team Discussion Points

1. **Approval to proceed with EnCodec testing?**
   - Low risk (just install + test)
   - ~2 days to validate quality

2. **Preference for integration approach?**
   - Option A (freeze EnCodec): Simpler, faster
   - Option B (fine-tune): Potentially better quality, more complex

3. **Quality thresholds for success?**
   - What SNR/PESQ/STOI targets are acceptable?
   - How important is subjective quality vs. objective metrics?

4. **Timeline expectations?**
   - 3-4 weeks realistic for full implementation + evaluation?
   - Acceptable to extend 1-2 weeks if needed for optimization?

5. **Backup plan?**
   - If EnCodec quality insufficient, proceed to Complex FFT?
   - Or revisit other approaches?

---

## Conclusion

**EnCodec provides a clear path to solving the audio reconstruction quality problem** with:
- ✓ Strong theoretical foundation (phase preservation through learned compression)
- ✓ Practical advantages (pretrained, efficient, production-proven)
- ✓ Low risk (easy to test, backup options available)
- ✓ Reasonable timeline (3-4 weeks to complete implementation)

**Recommendation:** Approve EnCodec testing and proceed with implementation if initial quality tests are successful.

---

**Status:** Awaiting team review and approval to proceed with Phase 1 (Proof of Concept).

**Contact:** Post questions/comments to Issue #23 or Slack channel.

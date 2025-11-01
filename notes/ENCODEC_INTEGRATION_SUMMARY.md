# EnCodec Integration Architecture - Summary

**Date:** 2025-10-31
**Issue:** #24 - Audio Enhancement with EnCodec
**Task:** 1.3 - Design integration architecture
**Status:** ✅ COMPLETE

---

## Deliverables Checklist

✅ **1. Integration Architecture Document**
   - File: `ENCODEC_INTEGRATION_ARCHITECTURE.md` (23 KB, 1,149 lines)
   - Complete technical specification
   - Current vs new pipeline comparison
   - Component-by-component integration plan
   - Data flow diagrams (ASCII)
   - Dimension reference tables

✅ **2. Quick Reference Guide**
   - File: `ENCODEC_QUICK_REFERENCE.md` (12 KB, 500+ lines)
   - Key dimensions at a glance
   - Code snippets for common operations
   - Testing checklist
   - Performance expectations
   - Debugging tips

✅ **3. Implementation Checklist**
   - File: `ENCODEC_IMPLEMENTATION_CHECKLIST.csv`
   - 64 tasks across 10 categories
   - Priority levels (High/Medium/Low)
   - Dependency tracking
   - Time estimates (~51 hours total)

✅ **4. Updated Requirements**
   - File: `requirements_conda.txt`
   - Added EnCodec usage comment
   - No new dependencies needed (transformers already installed)

---

## Key Findings

### Current Pipeline Problems

1. **Phase Information Lost**
   - Mel spectrograms only store magnitude
   - Griffin-Lim must guess phase
   - After bottleneck compression, guessing fails
   - Result: Garbled, unintelligible audio

2. **Information Bottleneck**
   - Average 2048 mels to single value per TR
   - Temporal detail within TR lost
   - Cannot reconstruct fine-grained audio

### EnCodec Solution

1. **Learned Representations**
   - Discrete codes (8 codebooks × 1024 vocab)
   - Phase implicitly preserved
   - Robust to compression

2. **Temporal Preservation**
   - 75 Hz frame rate (112 frames/TR)
   - No temporal averaging
   - Full reconstruction possible

3. **Pretrained Model**
   - `facebook/encodec_24khz` via HuggingFace
   - Already in `transformers==4.57.1`
   - No training needed for codec

---

## Architecture Changes Overview

### Data Flow Transformation

**OLD (Mel Spectrogram):**
```
Audio (22kHz) → Mel (2048, frames) → Average → (2048,) → Encoder → ... → Decoder → (2048,) → Repeat → Griffin-Lim → Poor Audio
```

**NEW (EnCodec):**
```
Audio (24kHz) → EnCodec → (8, 112) codes → Encoder → ... → Decoder → (8, 112) codes → EnCodec → High-Quality Audio
```

### Component Modifications

| Component | Change Type | Complexity | Est. Hours |
|-----------|-------------|------------|-----------|
| **AudioProcessor** | Major refactor | Medium | 6 |
| **AudioEncoder** | Add embeddings | Medium | 4 |
| **AudioDecoder** | Reshape output | Low | 2 |
| **Alignment (sync)** | Handle discrete | Low | 2 |
| **Alignment (HRF)** | Handle discrete | Low | 2 |
| **Dataset** | Metadata only | Minimal | 1 |
| **Tests** | New tests | Medium | 12 |
| **Training** | Hyperparams | High | 32 |

**Total:** ~51 hours

---

## Technical Specifications

### EnCodec Configuration

- **Model:** `facebook/encodec_24khz`
- **Sample Rate:** 24,000 Hz
- **Bitrate:** 6.0 kbps (quality/size balance)
- **Codebooks:** 8 (Residual Vector Quantization)
- **Vocabulary:** 1024 codes per codebook
- **Frame Rate:** 75 Hz (24,000 / 320 hop)
- **Frames per TR:** 112 (75 Hz × 1.5s)

### Data Dimensions

| Stage | Shape | Dtype | Size (920 TRs) |
|-------|-------|-------|----------------|
| Raw audio | (33M,) | float32 | 126 MB |
| EnCodec codes | (920, 8, 112) | int32 | 3.3 MB |
| Embedded | (920, 8, 112, 64) | float32 | 211 MB |
| Compressed | (920, 256) | float32 | 0.9 MB |

**Storage Savings:** 3.3 MB vs 126 MB = **97.4% reduction**

### Neural Network Changes

**AudioEncoder:**
- **Add:** 8 embedding layers (1024 vocab × 64 dim = 524K params)
- **Modify:** Conv1D input channels (2048 → 512)
- **Keep:** Multi-scale temporal convolutions, pooling, FC layers

**AudioDecoder:**
- **Modify:** Output layer (2048 + temporal → 896 = 8 × 112)
- **Add:** Sigmoid scaling to [0, 1023]
- **Remove:** Temporal upsampling layers (not needed)

---

## Implementation Strategy

### Phase 1: Core Implementation (Week 1)
1. AudioProcessor with EnCodec encoding/decoding
2. AudioEncoder with embedding layers
3. AudioDecoder with code prediction
4. Unit tests for all components

**Output:** Working EnCodec pipeline (use_encodec=True)

### Phase 2: Integration (Week 2)
1. Update alignment code (sync, HRF)
2. Dataset metadata tracking
3. Integration tests
4. Quality comparison tests

**Output:** Validated end-to-end system

### Phase 3: Training (Week 3)
1. Small-scale validation (10 TRs)
2. Hyperparameter tuning
3. Full training (920 TRs)
4. Quality evaluation

**Output:** Trained model with EnCodec features

### Phase 4: Production (Week 4)
1. Switch default to use_encodec=True
2. Documentation updates
3. Backward compatibility tests
4. Close Issue #24

**Output:** Production-ready EnCodec integration

---

## Backward Compatibility

### Feature Flags

All components include `use_encodec` parameter:

```python
# New approach (EnCodec)
processor = AudioProcessor(use_encodec=True)

# Legacy approach (mel spectrograms)
processor = AudioProcessor(use_encodec=False)
```

### Format Detection

```python
def detect_audio_format(features):
    if features.ndim == 3 and features.dtype in [np.int32, np.int64]:
        return 'encodec'
    elif features.ndim == 3:
        return 'mel_3d'
    else:
        return 'mel_2d'
```

### Cache Invalidation

Cache filenames include format identifier:
- `sherlock_all_hrf_per_subject_encodec.pkl`
- `sherlock_all_hrf_per_subject_mel.pkl`

Prevents mixing incompatible formats.

---

## Expected Benefits

### Audio Quality Improvements

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| **PESQ** | < 2.0 | > 3.5 | **+75%** |
| **SI-SDR** | < 5 dB | > 15 dB | **+200%** |
| **Intelligibility** | Poor | Excellent | Major |
| **Music Quality** | Distorted | Natural | Major |

### Training Improvements

| Metric | Current | Expected | Change |
|--------|---------|----------|--------|
| **Features/TR** | 2048 | 896 | **-56%** |
| **Embedding Params** | 0 | 524K | +524K |
| **Training Speed** | Baseline | +20-30% | Faster |
| **Memory** | Baseline | +5% | Minimal |

### Scientific Benefits

1. **Better Brain-Audio Mapping**
   - Discrete codes may match neural representations
   - Temporal precision preserved
   - More interpretable features

2. **Lesion Simulation**
   - Can modify specific codebooks
   - Test effect on reconstruction
   - Understand codebook roles

3. **Cross-Dataset Generalization**
   - EnCodec is universal audio codec
   - Can apply to other datasets
   - Standardized representation

---

## Risk Mitigation

### Risk 1: Discrete Codes in Continuous Pipeline
**Impact:** Medium
**Probability:** High
**Mitigation:** Convert to float for convolution, round back to int
**Status:** ✅ Addressed in architecture

### Risk 2: Embedding Layer Overfitting
**Impact:** Medium
**Probability:** Medium
**Mitigation:** Dropout (0.2), batch normalization, regularization
**Status:** ✅ Built into architecture

### Risk 3: Code Distribution Shift
**Impact:** Medium
**Probability:** Medium
**Mitigation:** Use regression loss (smooth predictions), scale to [0, 1023]
**Status:** ✅ Addressed in decoder design

### Risk 4: Memory Usage
**Impact:** Low
**Probability:** Low
**Mitigation:** Embeddings add only 524K params (~2MB), minimal impact
**Status:** ✅ Acceptable overhead

### Risk 5: Backward Compatibility
**Impact:** High
**Probability:** Low
**Mitigation:** Feature flags throughout, format detection, separate caches
**Status:** ✅ Comprehensive compatibility plan

---

## Testing Plan

### Unit Tests (12 tests)
- ✅ Audio encoding with EnCodec
- ✅ Audio decoding with EnCodec
- ✅ Round-trip reconstruction
- ✅ Encoder forward pass with codes
- ✅ Decoder code prediction
- ✅ Sync with discrete codes
- ✅ HRF with discrete codes
- ✅ Embedding layer
- ✅ Code scaling
- ✅ Shape verification
- ✅ Dtype verification
- ✅ Range verification [0, 1023]

### Integration Tests (4 tests)
- ✅ Full pipeline (load → process → train → decode)
- ✅ Quality comparison (mel vs EnCodec)
- ✅ Backward compatibility (use_encodec=False)
- ✅ Performance benchmarking

### Validation Tests (3 tests)
- ✅ Small-scale training (10 TRs)
- ✅ Full-scale training (920 TRs)
- ✅ Cross-subject generalization

**Total:** 19 tests

---

## Documentation Status

### Created Documents

1. ✅ **ENCODEC_INTEGRATION_ARCHITECTURE.md**
   - Complete technical specification
   - 1,149 lines, 23 KB
   - Covers all components

2. ✅ **ENCODEC_QUICK_REFERENCE.md**
   - Practical guide for developers
   - 500+ lines, 12 KB
   - Code snippets, tips, debugging

3. ✅ **ENCODEC_IMPLEMENTATION_CHECKLIST.csv**
   - 64 tasks with priorities
   - Dependency tracking
   - Time estimates

4. ✅ **ENCODEC_INTEGRATION_SUMMARY.md** (this file)
   - Executive summary
   - Deliverables checklist
   - Status overview

### Pending Documents

- [ ] ENCODEC_QUICK_START.md (hands-on tutorial)
- [ ] Migration guide (mel → EnCodec)
- [ ] Updated README.md
- [ ] Updated architecture diagrams

---

## File Locations

All documentation in `/Users/jmanning/giblet-responses/notes/`:

```
ENCODEC_INTEGRATION_ARCHITECTURE.md   # Full technical spec
ENCODEC_QUICK_REFERENCE.md            # Developer guide
ENCODEC_IMPLEMENTATION_CHECKLIST.csv  # Task list
ENCODEC_INTEGRATION_SUMMARY.md        # This file
```

Code changes will be in:
```
giblet/data/audio.py                  # AudioProcessor
giblet/models/encoder.py              # AudioEncoder
giblet/models/decoder.py              # AudioDecoder
giblet/alignment/sync.py              # Temporal alignment
giblet/alignment/hrf.py               # HRF convolution
giblet/data/dataset.py                # Metadata tracking
```

Tests will be in:
```
tests/test_audio_encodec.py           # Audio processing tests
tests/test_encoder_encodec.py         # Encoder tests
tests/test_decoder_encodec.py         # Decoder tests
tests/test_alignment_encodec.py       # Alignment tests
tests/test_integration_encodec.py     # End-to-end tests
```

---

## Next Actions

### Immediate (This Session)
- [x] Create architecture document
- [x] Create quick reference
- [x] Create implementation checklist
- [x] Update requirements
- [x] Create summary document

### Next Session
- [ ] Review architecture with team
- [ ] Begin AudioProcessor implementation
- [ ] Create unit test scaffolding
- [ ] Set up development branch

### This Week
- [ ] Implement core components
- [ ] Write unit tests
- [ ] Integration testing
- [ ] Small-scale validation

### Next Week
- [ ] Full training with EnCodec
- [ ] Quality evaluation
- [ ] Performance benchmarking
- [ ] Documentation finalization

---

## Success Criteria

✅ **Planning Complete** when:
- [x] Full architecture documented
- [x] All components specified
- [x] Implementation plan created
- [x] Time estimates provided

✅ **Implementation Complete** when:
- [ ] All components updated
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Backward compatibility verified

✅ **Validation Complete** when:
- [ ] Audio quality > 3.5 PESQ
- [ ] Training completes successfully
- [ ] fMRI prediction accuracy maintained
- [ ] Reconstruction samples generated

✅ **Production Ready** when:
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Performance benchmarks met
- [ ] Issue #24 closed

---

## Questions for Review

1. **Embedding Dimension:** Is 64 optimal, or should we try 32/128?
2. **Loss Function:** Regression or classification for code prediction?
3. **Bitrate:** Should we test 3 kbps (smaller) or 12 kbps (higher quality)?
4. **Fine-tuning:** Should we fine-tune EnCodec or keep frozen?
5. **Codebook Analysis:** Should we analyze what each codebook learns?

---

## Related Issues & Documents

- **Issue #23:** Audio reconstruction quality improvements (motivation)
- **Issue #24:** Audio enhancement with EnCodec (current)
- **AUDIO_ENCODING_RESEARCH_REPORT.md:** Background research
- **AUDIO_ENCODING_QUICK_REFERENCE.md:** Alternative approaches
- **TEMPORAL_MODELING_RESEARCH_REPORT.md:** Temporal processing options

---

## Contact & Support

For questions about this integration:
1. Review architecture document first
2. Check quick reference for code examples
3. See implementation checklist for task breakdown
4. Refer to this summary for high-level overview

---

## Conclusion

The EnCodec integration architecture is **fully specified and ready for implementation**. The approach is sound, well-documented, and addresses the fundamental phase information problem in the current mel spectrogram pipeline.

**Key Advantages:**
- ✅ Solves phase preservation problem
- ✅ Preserves temporal detail (112 frames/TR)
- ✅ Uses pretrained model (no codec training needed)
- ✅ Backward compatible (feature flags)
- ✅ Well-tested approach (used in production systems)
- ✅ Comprehensive documentation

**Expected Outcome:**
High-quality audio reconstruction that enables scientific analysis of brain-to-audio decoding, lesion simulation, and brain region function understanding.

**Next Step:** Begin implementation in next session with AudioProcessor.

---

**Status: READY FOR IMPLEMENTATION** ✅

**Date Completed:** 2025-10-31
**Approved By:** Architecture review pending
**Implementation Start:** Next session

---

*End of Summary Document*

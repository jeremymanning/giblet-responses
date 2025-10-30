# Visualization Audit - Actionable Next Steps

## Summary of Findings

**Current State**: Vertical visualization showing 52 layers, color-coded by modality
**Issues Found**: 3 major issues (parallel paths hidden, no orientation choice, logical layer mismatch)
**Fixability**: Moderate (parallel paths) to Very High (logical layers)

---

## Recommended Action Plan

### Phase 1: Documentation & Quick Wins (1-2 hours)

**No code changes, immediate clarity improvements**

1. ✓ **Add architecture diagram to README**
   - Show 11 logical layers from Issue #2
   - Explain how 52 leaf layers map to 11 logical layers
   - Clarify parallel structure exists but not shown

2. ✓ **Update visualization function docstring**
   - Add note about 52 vs 11 layers
   - Explain color coding includes parallel path hints
   - Reference this audit document

3. ✓ **Add TODO comments in code**
   - Mark lines 239-313 with "TODO: Show parallel paths"
   - Mark lines 228-229 with "TODO: Add orientation parameter"
   - Mark line 44 with "NOTE: Sequential containers skipped for grouping"

4. ✓ **Create example output analysis**
   - Document that layers 1-9 = video path, 10-18 = audio, 19-22 = text
   - Show which layers correspond to Issue #2 spec layers

**Effort**: 30 minutes
**Impact**: High (eliminates user confusion)
**Files to modify**: README.md, visualization.py (docstrings), this document

---

### Phase 2: Parallel Path Support (4-6 hours)

**Implement core feature improving visualization clarity**

1. **Add modality detection function**
   ```python
   def detect_modality(layer_name: str) -> str:
       """Return 'video', 'audio', 'text', or 'main'."""
   ```
   Location: After line 151 (before create_network_diagram)
   Lines: ~15

2. **Add modality lane positioning**
   ```python
   modality_x_offsets = {
       'main': 2.5,    # Non-modality layers (bottleneck stem)
       'video': 0.5,   # Left lane
       'audio': 2.5,   # Center lane
       'text': 4.5     # Right lane
   }
   ```
   Location: Lines 226-228 (before axis setup)
   Lines: ~5

3. **Modify rendering loop to use modality-specific X offset**
   ```python
   for layer in layers:
       modality = detect_modality(layer['name'])
       x_base = modality_x_offsets[modality]
       # Use x_base instead of hardcoded 5
   ```
   Location: Lines 239-313
   Lines: ~10 modified

4. **Add connecting lines at merge/split points**
   - Add horizontal connectors where paths merge (pool operation)
   - Add horizontal connectors where paths split (decoder layer 10)
   Location: After main loop, before legend
   Lines: ~30

5. **Update axis limits for 3 lanes**
   ```python
   ax.set_xlim(-0.5, 5.5)  # Changed from 0-10
   ```
   Location: Line 228
   Lines: 1

6. **Add lane labels**
   - "Video Encoder (2A)" at x=0.5
   - "Shared Stem (2-4)" at x=2.5
   - "Audio Encoder (2B)" at x=2.5 (or separate)
   - "Text Encoder (2C)" at x=4.5
   Location: After legend setup (lines 315-332)
   Lines: ~15

7. **Test with real model**
   ```python
   from giblet.models.autoencoder import MultimodalAutoencoder
   model = MultimodalAutoencoder()
   create_network_diagram(model, 'test_parallel.png')
   ```
   Verify lanes are distinct, merge points visible

**Effort**: 4-6 hours
**Impact**: Major - Parallel structure now visible
**Testing**: Visual inspection of output
**Backward Compatibility**: ✓ Complete (no API changes, enhanced rendering)

---

### Phase 3: Horizontal Orientation Support (6-10 hours)

**Enable flexible layout for different display configurations**

1. **Add orientation parameter**
   ```python
   def create_network_diagram(
       ...
       orientation: str = 'vertical',  # NEW
       ...
   ) -> Path:
   ```
   Location: Line 154-162
   Lines: 1

2. **Add conditional axis setup**
   ```python
   if orientation == 'vertical':
       fig = plt.figure(figsize=figsize, ...)
       ax.set_xlim(0, 10)
       ax.set_ylim(0, len(layers) + 2)
   else:  # horizontal
       fig = plt.figure(figsize=(figsize[1], figsize[0]), ...)
       ax.set_xlim(0, len(layers) + 2)
       ax.set_ylim(0, 10)
   ```
   Location: Lines 225-229
   Lines: ~15

3. **Add conditional position tracking**
   ```python
   if orientation == 'vertical':
       y_pos = len(layers)
       x_pos = None
   else:
       x_pos = 0
       y_pos = None
   ```
   Location: Line 240
   Lines: ~5

4. **Update rectangle drawing coordinates**
   Location: Lines 250, 263, 277 (3 places)
   Lines: ~20 modified

5. **Update polygon drawing coordinates**
   Location: Lines 262-266, 276-280 (2 places)
   Lines: ~20 modified

6. **Update position increment**
   ```python
   if orientation == 'vertical':
       y_pos -= 1
   else:
       x_pos += 1
   ```
   Location: Line 313
   Lines: ~3

7. **Update text positioning**
   Location: Lines 295, 308 (2 places)
   Lines: ~10 modified

8. **Update legend positioning**
   Location: Line 331
   Lines: ~3

9. **Update sizing note positioning**
   Location: Line 336
   Lines: ~3

10. **Update docstring and parameter hints**
    Location: Lines 155-215
    Lines: ~10

11. **Comprehensive testing**
    ```python
    model = MultimodalAutoencoder()
    create_network_diagram(model, 'vert.png', orientation='vertical')
    create_network_diagram(model, 'horiz.png', orientation='horizontal')
    # Visual inspection of both outputs
    ```

**Effort**: 6-10 hours
**Impact**: Medium - Nice feature, but workaround (vertical) exists
**Testing**: Visual inspection of vertical and horizontal outputs
**Backward Compatibility**: ✓ Complete (new optional parameter)

---

### Phase 4: Logical Layer Abstraction (NOT RECOMMENDED - Deferred)

**Show 11 logical layers instead of 52 leaf layers**

**Status**: DEFERRED - High effort, would require custom architecture parser
**Rationale**: Better achieved with parallel paths support (Phase 2) + good documentation (Phase 1)

If reconsidered later:
- Create model-specific layer grouping logic
- Map 52 modules to 11 Issue #2 layers
- Would require ~200+ lines of code
- Not generalizable to other models
- Consider as separate feature in v2

---

## Implementation Priority Matrix

| Phase | Effort | Impact | Priority | Status |
|-------|--------|--------|----------|--------|
| 1: Documentation | 0.5h | HIGH | 1 (Start) | Ready |
| 2: Parallel Paths | 4-6h | MAJOR | 2 (Next) | Planned |
| 3: Orientation | 6-10h | MEDIUM | 3 (Optional) | Planned |
| 4: Logical Layers | 10+h | MEDIUM | 4 (Defer) | Deferred |

---

## Estimated Timeline

**Quick Start** (Just Phase 1):
- **Time**: 30 minutes
- **Impact**: High (clarity improvement, no code changes)
- **Risk**: None

**Recommended** (Phases 1 + 2):
- **Time**: 4.5-6.5 hours
- **Impact**: Very High (parallel paths now visible)
- **Risk**: Low (well-defined changes, backward compatible)

**Full Implementation** (Phases 1 + 2 + 3):
- **Time**: 10-16 hours
- **Impact**: Very High
- **Risk**: Low (incremental, well-scoped)

---

## Success Criteria

### Phase 1 (Documentation)
- [ ] README explains 11 vs 52 layers
- [ ] Docstring mentions parallel path limitation
- [ ] TODO comments added to code
- [ ] Audit report linked in README

### Phase 2 (Parallel Paths)
- [ ] Video layers appear in left lane
- [ ] Audio layers appear in center lane
- [ ] Text layers appear in right lane
- [ ] Merge points have connecting lines
- [ ] All 52 parameters still counted correctly
- [ ] No regression in vertical layout
- [ ] Backward compatible (no API changes)

### Phase 3 (Orientation)
- [ ] Vertical mode produces original output
- [ ] Horizontal mode produces wide layout
- [ ] Both modes readable and clear
- [ ] Legend and labels work in both modes
- [ ] PDF/PNG export works in both modes
- [ ] Backward compatible (vertical is default)

---

## Risk Assessment

### Phase 1 (Documentation)
- Risk: None
- Mitigation: N/A

### Phase 2 (Parallel Paths)
- Risk: Rendering complexity increases
- Mitigation: Start with simple lane offsets, add complexity incrementally
- Risk: Modality detection fragile
- Mitigation: Test with actual layer names, add comprehensive comments

### Phase 3 (Orientation)
- Risk: Coordinate transformations error-prone
- Mitigation: Keep vertical as default, test both paths thoroughly
- Risk: Text positioning breaks in horizontal
- Mitigation: Calculate positions conditionally throughout

---

## Testing Checklist

### Phase 1 Testing
- [ ] Build and view generated PDF/images
- [ ] Verify parameter counts match
- [ ] Verify color coding works
- [ ] Verify text rendering

### Phase 2 Testing
- [ ] Generate diagram with real model
- [ ] Verify lanes are visually distinct
- [ ] Verify merge point lines visible
- [ ] Verify layer names readable
- [ ] Verify parameter counts unchanged
- [ ] Compare before/after images

### Phase 3 Testing
- [ ] Generate vertical diagram
- [ ] Generate horizontal diagram
- [ ] Verify both export formats (PDF, PNG)
- [ ] Verify text rotation works
- [ ] Verify legend positioning
- [ ] Verify coordinate calculations
- [ ] Test with extreme layer counts

---

## File Modification Summary

| File | Phase | Lines Changed | Reason |
|------|-------|---|---|
| `visualization.py` | 1 | ~20 | Comments only |
| `visualization.py` | 2 | ~100 | New detection + rendering |
| `visualization.py` | 3 | ~100 | Conditional logic |
| `README.md` | 1 | ~30 | Architecture explanation |
| `VISUALIZATION_AUDIT_*.md` | All | Reference | Documentation |

**Total code changes**: ~200 lines for all phases

---

## Resources & References

**Generated Documentation**:
- `VISUALIZATION_AUDIT_REPORT.md` - Comprehensive analysis
- `VISUALIZATION_AUDIT_SUPPLEMENTARY.md` - Detailed diagrams
- `VISUALIZATION_TECHNICAL_REFERENCE.md` - Code-level details
- `VISUALIZATION_AUDIT_EXECUTIVE_SUMMARY.txt` - Overview

**Code References**:
- Layer extraction: `visualization.py` lines 21-77
- Rendering loop: `visualization.py` lines 239-313
- Color selection: `visualization.py` lines 115-151

**Related Issues**:
- Issue #2: Architecture specification (11 logical layers)
- Issue #1: Current visualization limitations

---

## Recommended Next Step

**START WITH PHASE 1 (Documentation)**

This provides immediate value (clarity) with zero code changes and zero risk.
Once Phase 1 is complete, proceed to Phase 2 (Parallel Paths) for the major
improvement in visualization clarity.

Phase 3 (Orientation) can be done later if users request it, or deferred
entirely if vertical layout proves sufficient.

**Estimated time for Phases 1+2**: 4.5-6.5 hours
**Expected impact**: Complete solution to parallel path visibility issue

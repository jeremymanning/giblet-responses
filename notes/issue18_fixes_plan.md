# Issue #18 Diagram Fixes - Task Breakdown

## Problems Identified

1. **Orientation:** Vertical → should be horizontal
2. **Parallel processing:** Layers 2A/B/C and 10A/B/C not shown as parallel
3. **Color inconsistencies:** Legend vs actual colors, missing colors
4. **Library evaluation:** May need alternative to current approach

## Task Dependencies

### Batch 1: Research & Analysis (PARALLEL)

**Task 1A: Research Alternative Libraries**
- Evaluate visualtorch (https://visualtorch.readthedocs.io/)
- Evaluate torchviz
- Evaluate hiddenlayer
- Evaluate custom matplotlib solutions
- Compare: parallel layer support, horizontal orientation, customization
- Test with REAL 2B param model

**Task 1B: Audit Current Implementation**
- Review giblet/utils/visualization.py
- Understand why parallel layers not shown
- Document architecture vs visualization mismatch
- Identify what needs fixing

**Task 1C: Color Scheme Analysis**
- Document legend colors vs actual diagram colors
- Identify inconsistencies
- Propose corrected color scheme

### Batch 2: Implementation Fixes (Depends on Batch 1)

**Task 2A: Fix Orientation** 
- Update to horizontal layout
- Test with real model
- Verify readability

**Task 2B: Fix Parallel Processing**
- Implement parallel layer visualization
- Show 2A/B/C branching from Layer 1
- Show 10A/B/C outputting to Layer 11
- Use best library from Task 1A

**Task 2C: Fix Color Scheme**
- Consistent colors throughout
- Match legend to diagram
- Remove unused colors

### Batch 3: Validation (Depends on Batch 2)

**Task 3A: Regenerate Diagrams**
- Run with fixes
- Generate new network.pdf/png

**Task 3B: Manual Verification**
- Check horizontal orientation
- Verify parallel layers visible
- Confirm color consistency
- Ensure publication quality

**Task 3C: Update Tests**
- Verify tests still pass
- Add tests for new features

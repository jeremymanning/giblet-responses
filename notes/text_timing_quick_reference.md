# Text Timing Alignment - Quick Reference

## ✓ IMPLEMENTATION IS CORRECT

No changes needed. The code properly uses Start Time/End Time and aligns to TRs correctly.

---

## How It Works

### TR Alignment Formula

For annotation with `[start_time, end_time]`:

```python
start_tr = floor(start_time / 1.5)
end_tr = ceil(end_time / 1.5)
contributing_trs = range(start_tr, end_tr)
```

**Example:** Annotation at [0s, 12s]
- start_tr = floor(0 / 1.5) = 0
- end_tr = ceil(12 / 1.5) = 8
- TRs: [0, 1, 2, 3, 4, 5, 6, 7] ✓

---

## Visual Timeline

```
Time:     0s    1.5s   3.0s   4.5s   6.0s   7.5s   9.0s  10.5s  12.0s  13.5s
          |      |      |      |      |      |      |      |      |      |
TRs:      |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |
          |      |      |      |      |      |      |      |      |      |
Seg 0:    [================================================)
          0s                                               12s

Seg 1:                                                     [==============)
                                                          12s            15s
```

**Segment 0 [0-12s]** → TRs 0-7
**Segment 1 [12-15s]** → TRs 8-9

---

## Dataset Stats

| Metric | Value |
|--------|-------|
| Total annotations | 1000 segments |
| Annotation duration | 0s - 1544s (25.7 min) |
| fMRI TRs available | 920 TRs |
| fMRI duration | 0s - 1380s (23 min) |
| Annotations beyond 920 TRs | 80 (correctly ignored) |
| TRs with gaps | 0 |
| TRs with overlaps | 918 |
| TRs with single annotation | 2 |

---

## Key Features

✓ **Overlap Handling:** Mean aggregation of multiple annotations
✓ **Gap Filling:** Forward fill (not needed - no gaps)
✓ **Truncation:** Clamps to 920 TRs
✓ **Integration:** Aligns with sync.py minimum TR logic

---

## Code Locations

- **Text processing:** `/Users/jmanning/giblet-responses/giblet/data/text.py`
- **TR alignment:** Lines 166-258 (`align_to_trs` method)
- **Sync integration:** `/Users/jmanning/giblet-responses/giblet/alignment/sync.py`
- **Data source:** `data/annotations.xlsx`

---

## Testing

Run verification tests:
```bash
python test_text_comprehensive.py
```

All tests pass ✓

---

## Data Note

Annotations contain duplicates:
- Segments 1-479 duplicated as 483-961
- Same timing, same text
- Mean aggregation handles correctly
- Not a code issue

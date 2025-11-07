# CI Fixes Session - 2025-11-04 18:00-19:00 UTC

## Status: IN PROGRESS
Context ran to 111k/200k tokens. Continuing work needed.

## Completed
1. ✅ Renamed `SherlockDataset` → `MultimodalDataset` throughout codebase (commit 951886a)
2. ✅ Ran Black formatter on all files (48 files reformatted) (commit 951886a)
3. ✅ Ran isort to fix import ordering (45 files fixed) (commit 4df4e8d)
4. ✅ Fixed .pre-commit-config.yaml YAML syntax error (removed TOML lines 66-68) (commit 4df4e8d)

## Issues Discovered - NOT YET FIXED

### 1. Flake8 Violations (50+ errors)
**Must fix before next push!**

Run `flake8 giblet/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503,E501 --max-complexity=15` to see all errors.

Main categories:
- **F401**: Unused imports (~25 instances)
  - `giblet/alignment/sync.py:18` - `typing.Optional`
  - `giblet/alignment/sync.py:22` - `.hrf.apply_hrf`
  - `giblet/data/audio.py:17` - `tqdm.tqdm`
  - `giblet/data/dataset.py:20` - `typing.Tuple`
  - `giblet/data/dataset.py:21` - `tqdm.tqdm`
  - `giblet/data/dataset_notxt.py:11` - `pandas as pd`
  - `giblet/data/text.py:16` - `tqdm.tqdm`
  - `giblet/models/decoder.py:16` - `typing.Optional`
  - `giblet/training/trainer.py:54` - `json`
  - `giblet/training/trainer.py:56` - `time`
  - `giblet/training/trainer.py:59` - `typing.Any`, `typing.Tuple`
  - `giblet/training/trainer.py:61` - `numpy as np`
  - `giblet/utils/plotneuralnet.py:21` - `os`
  - `giblet/utils/plotneuralnet.py:23` - `sys`
  - `giblet/utils/visualization.py:17` - `torch`
  - `scripts/precompute_encodec_features.py:25` - `pandas as pd`

- **F541**: f-string missing placeholders (~20 instances)
  - `giblet/alignment/sync.py:233,282,283`
  - `giblet/data/dataset.py:252,278,332,478`
  - `giblet/data/dataset_notxt.py:141,145,262`
  - `giblet/utils/plotneuralnet.py:606,613`
  - `giblet/utils/visualization.py:434`
  - `scripts/precompute_encodec_features.py:100,191,193`
  - `scripts/test_dataset_encodec.py` (multiple)

- **E722**: Bare except clause
  - `giblet/data/dataset.py:194` - Should specify exception type

- **E402**: Module level import not at top of file (4 instances)
  - `giblet/training/trainer.py:68` - MultimodalAutoencoder import after bitsandbytes try/except
  - `scripts/precompute_encodec_features.py:31`
  - `scripts/pregenerate_cache.py:23`
  - `scripts/test_dataset_encodec.py:24`

### 2. Pre-commit Hooks Failure
Likely caused by same flake8 violations. Once flake8 errors are fixed, pre-commit should pass.

### 3. Cluster Script Credential Paths (CRITICAL BUG)
**User reported**: When running from `scripts/` directory:
```bash
cd scripts
./cluster/check_remote_status.sh --cluster tensor01
```

Error:
```
✗ Error: Credentials file not found: cluster_config/tensor01_credentials.json
```

**Root cause**: Scripts in `scripts/cluster/` use relative path `cluster_config/...` which only works when run from repository root. When run from `scripts/` directory, it looks for `scripts/cluster_config/` which doesn't exist.

**Files affected**:
- `scripts/cluster/sync_results.sh`
- `scripts/cluster/remote_train.sh`
- `scripts/cluster/submit_job.sh`
- `scripts/cluster/monitor_job.sh`
- `scripts/cluster/check_remote_status.sh`
- `scripts/cluster/setup_cluster.sh`

**Solution options**:
A. Add `$SCRIPT_DIR` detection and use absolute paths (recommended)
B. Create symlinks at root for commonly used scripts
C. Add wrapper scripts at root

## Next Steps (CRITICAL)

### User Feedback:
> "run (and fix!!) *ALL* tests + linters *BEFORE* pushing to github. we get billed each time the CI runs, so it's much cheaper to run things locally."

### Process for next session:
1. **Fix all flake8 violations** (remove unused imports, fix f-strings, fix bare excepts, move imports to top)
2. **Fix cluster script paths** (make them work from any directory)
3. **Run full test suite locally**: `pytest tests/ -v`
4. **Run all linters locally**:
   - `black giblet/ tests/ scripts/ --check`
   - `isort giblet/ tests/ scripts/ --check-only`
   - `flake8 giblet/ tests/ scripts/`
   - `pre-commit run --all-files`
5. **Only push when ALL pass locally**

## Git Status
Current branch: `main`
Latest commits:
- `4df4e8d` - Fix remaining CI issues: isort + pre-commit YAML syntax
- `951886a` - Fix CI failures: Rename SherlockDataset to MultimodalDataset + Black formatting

Unpushed changes: None (but flake8 fixes needed before next push)

## Context Usage
At 111,811 tokens when this note was created. Session should be continued with fresh context.

## Commands to Run Next Session

```bash
# 1. Fix flake8 errors (use Task tool or manual edits)
# See list above for specific files/lines

# 2. Run linters locally
black giblet/ tests/ scripts/ --check
isort giblet/ tests/ scripts/ --check-only
flake8 giblet/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503,E501 --max-complexity=15
pre-commit run --all-files

# 3. Run tests locally
pytest tests/ -v --tb=short

# 4. Only when ALL pass locally:
git add -A
git commit -m "Fix flake8 violations + cluster script paths"
git push origin main

# 5. Monitor CI
gh run list --repo jeremymanning/giblet-responses --branch main --limit 1
```

## Important Reminders
- **Never push without running linters/tests locally first**
- GitHub Actions runs cost money
- Replicate issues locally before pushing fixes

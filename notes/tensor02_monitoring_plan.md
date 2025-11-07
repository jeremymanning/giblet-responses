# Tensor02 Overnight Monitoring Plan

**Date**: November 5-6, 2025
**Purpose**: Monitor tensor02 test run to validate fMRI normalization fix
**Duration**: ~24 hours (50 epochs with 5 subjects)

---

## Monitoring Schedule

### Phase 1: Initial Startup (0-2 hours)
**Goal**: Verify training starts successfully after code sync completes

**Check at**:
- Now: Code sync status
- +2 hours: Training startup verification

**What to check**:
- ✅ Code sync completed successfully
- ✅ Screen session created
- ✅ Training process running
- ✅ GPUs active (6 GPUs with >50% utilization)
- ✅ Log file exists and growing
- ✅ First epoch started

**Success criteria**:
- All 6 GPUs showing 70-90% utilization
- Training log shows "Epoch 1/50" started
- Loss values appearing in log
- No errors or exceptions in log

**If failed**:
- Check for errors in log file
- Verify conda environment activated
- Check for OOM errors (batch size too large)
- Restart with same config if needed

---

### Phase 2: Early Training (2-8 hours)
**Goal**: Confirm loss is decreasing and normalization fix is working

**Check every**: 1 hour

**What to check**:
- Training progress (which epoch?)
- Recent loss values (decreasing?)
- GPU utilization stable
- No memory errors
- Log file still growing

**Success criteria**:
- Loss is decreasing (>5% reduction from epoch 1)
- Initial loss ~100-1000 (NOT ~275,000 like tensor01)
- fMRI loss component ~10-100
- All loss components visible and balanced
- No NaN or Inf values
- Epochs completing in ~2-3 hours each

**Red flags**:
- Loss flat or increasing
- Loss still ~275,000 (normalization not applied)
- OOM errors
- Training stalled (no progress for 30+ min)
- Single loss component >90% of total

---

### Phase 3: Mid Training (8-16 hours)
**Goal**: Verify sustained convergence

**Check every**: 2 hours

**What to check**:
- Cumulative loss reduction from epoch 1
- Training vs validation loss gap
- GPU health (temperatures, utilization)
- Disk space for checkpoints

**Success criteria**:
- Loss decreased by >20% from initial
- Validation loss follows training (gap <2×)
- Steady convergence rate (~1-2% per epoch)
- All GPUs healthy (<85°C)
- Checkpoint saves successful

**If issues**:
- High train/val gap: May indicate overfitting (continue monitoring)
- GPU temp >85°C: Check cooling, may need to reduce batch size
- Checkpoint errors: Check disk space

---

### Phase 4: Final Hours (16-24 hours)
**Goal**: Ensure training completes successfully

**Check every**: 2-3 hours

**What to check**:
- Approaching epoch 50
- Final loss values
- Checkpoint saves
- Overall convergence trend

**Success criteria**:
- Loss reduced by >50% from initial
- Training completes all 50 epochs
- Final checkpoint saved successfully
- No crashes or errors

---

## Monitoring Commands

### Quick Status Check
```bash
./scripts/cluster/monitor_tensor02.sh
```

### Manual Checks

**SSH to cluster**:
```bash
sshpass -p "yaf1wue7gev_WQB.ueb" ssh -o StrictHostKeyChecking=no \
    f002d6b@tensor02.dartmouth.edu
```

**Check screen sessions**:
```bash
screen -ls
```

**Attach to training session**:
```bash
screen -r tensor02_test_fixed
# Detach with: Ctrl-A then D
```

**View logs in real-time**:
```bash
tail -f ~/giblet-responses/logs/training_tensor02_test_fixed_*.log
```

**Check GPU usage**:
```bash
nvidia-smi
# or for continuous monitoring:
watch -n 1 nvidia-smi
```

**Check latest loss values**:
```bash
tail -100 ~/giblet-responses/logs/training_tensor02_test_fixed_*.log | \
    grep -i "loss" | tail -10
```

**Check for errors**:
```bash
tail -200 ~/giblet-responses/logs/training_tensor02_test_fixed_*.log | \
    grep -i "error\|exception\|failed"
```

---

## Common Issues and Fixes

### Issue 1: Code Sync Failed
**Symptoms**: rsync errors, incomplete transfer
**Fix**:
```bash
# Manually rsync
rsync -avz --progress -e "sshpass -p 'yaf1wue7gev_WQB.ueb' ssh -o StrictHostKeyChecking=no" \
    /Users/jmanning/giblet-responses/ \
    f002d6b@tensor02.dartmouth.edu:~/giblet-responses/
```

### Issue 2: Training Won't Start
**Symptoms**: Screen session exists but no training process
**Possible causes**:
- Conda environment not activated
- Missing dependencies
- Config file errors
- Data not accessible

**Debug**:
```bash
# SSH and check manually
ssh f002d6b@tensor02.dartmouth.edu
cd ~/giblet-responses
conda activate giblet-py311
python -m giblet.training.train --config configs/training/tensor02_test_50epoch_config.yaml
```

### Issue 3: OOM (Out of Memory)
**Symptoms**: CUDA out of memory errors in log
**Fixes** (in order of preference):
1. Reduce batch size: Edit config `batch_size: 2` → `batch_size: 1`
2. Enable gradient checkpointing: Already enabled
3. Reduce number of GPUs: `world_size: 6` → `world_size: 4`

### Issue 4: Loss Not Decreasing
**Symptoms**: Loss flat after 5+ epochs
**Checks**:
- Verify normalization is actually applied (check log for "Applied z-score normalization")
- Check loss scale (should be ~100-1000, not ~275,000)
- Verify learning rate is not too low

**If normalization not applied**:
```bash
# Check if code sync included fMRI fix
ssh f002d6b@tensor02.dartmouth.edu "grep -A 10 'def nii_to_features' ~/giblet-responses/giblet/data/fmri.py | grep normalize"
```

### Issue 5: Training Crashed
**Symptoms**: Screen session gone, process terminated
**Steps**:
1. Check logs for error message
2. Fix issue (OOM, missing data, etc.)
3. Restart training:
```bash
./scripts/cluster/remote_train.sh --cluster tensor02 \
    --config configs/training/tensor02_test_50epoch_config.yaml \
    --gpus 6 --name tensor02_test_fixed
```

---

## Success Metrics

### Compared to Tensor01 (Broken)

| Metric | Tensor01 (No Fix) | Tensor02 (With Fix) | Status |
|--------|-------------------|---------------------|--------|
| Initial Loss | ~275,000 | ~100-1,000 | ✅ Expected |
| fMRI Loss | ~100,000,000 | ~10-100 | ✅ Expected |
| Loss @ Epoch 15 | -0.29% change | -10 to -20% change | 🔍 Monitor |
| fMRI Dominance | >99.999% | <50% | 🔍 Monitor |
| Convergence | None visible | Steady decrease | 🔍 Monitor |

### Target Outcomes (End of 50 Epochs)

- ✅ Total loss decreased by >50%
- ✅ All loss components balanced (<50% each)
- ✅ Validation loss follows training (gap <2×)
- ✅ No numerical instabilities (NaN/Inf)
- ✅ Checkpoints saved successfully

---

## Automated Monitoring (Optional)

### Create Monitoring Loop Script

Save as `scripts/cluster/monitor_loop.sh`:
```bash
#!/bin/bash
# Monitor tensor02 every hour for 24 hours

for i in {1..24}; do
    echo "═══════════════════════════════════════"
    echo "  Monitoring Check #$i of 24"
    echo "═══════════════════════════════════════"

    ./scripts/cluster/monitor_tensor02.sh

    if [ $i -lt 24 ]; then
        echo ""
        echo "⏰ Sleeping for 1 hour... (Check #$((i+1)) at $(date -d '+1 hour' '+%H:%M'))"
        sleep 3600  # 1 hour
    fi
done

echo ""
echo "✅ 24-hour monitoring complete!"
```

### Run in Background

```bash
chmod +x scripts/cluster/monitor_loop.sh
nohup ./scripts/cluster/monitor_loop.sh > tensor02_monitoring.log 2>&1 &
echo $! > tensor02_monitor.pid

# To check monitoring output:
tail -f tensor02_monitoring.log

# To stop monitoring:
kill $(cat tensor02_monitor.pid)
```

---

## Next Steps After 24 Hours

### If Training Successful
1. **Compare Results**:
   - Tensor01 (broken): Loss ~275,000, flat
   - Tensor02 (fixed): Loss decreased by >50%

2. **Make Decision**:
   - ✅ Deploy fix to production
   - Stop tensor01 run
   - Start new production run with normalized fMRI

3. **Update Configs**:
   - Update `production_500epoch_config.yaml` with `warmup_epochs: 2`
   - Document that normalization is now default
   - Archive old broken checkpoints

### If Training Failed
1. **Analyze Failure**:
   - Review all logs
   - Check what went wrong
   - Document findings

2. **Try Alternative Fix**:
   - Adjust loss weights manually
   - Try higher learning rate
   - Test on single subject first

3. **Report Results**:
   - Update GitHub Issue #32
   - Document what worked/didn't work
   - Plan next iteration

---

## Contact Information

**Monitoring Started**: November 5, 2025
**Maintainer**: jeremy.manning@dartmouth.edu
**Issue**: https://github.com/ContextLab/giblet-responses/issues/32

---

**Last Updated**: November 5, 2025

# Model Evaluation Tools

This directory contains tools for evaluating trained multimodal autoencoder models.

## Available Tools

### 1. `evaluate_model_status.sh` - Comprehensive Model Evaluation

**Purpose**: All-in-one tool for model checkpoint validation, weight health checking, and reconstruction quality assessment.

**Features**:
- Checkpoint metadata extraction (epoch, losses, parameter count)
- Weight health analysis (NaN/Inf detection, statistics)
- Reconstruction quality visualization (video, audio, text, fMRI)
- Automated report generation

**Usage**:
```bash
# Full evaluation
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints_local/tensor02_fixed_lr/best_checkpoint.pt \
    --config configs/training/tensor02_test_50epoch_config.yaml

# Quick status check (skip heavy computations)
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/model.pt \
    --config configs/config.yaml \
    --skip-reconstructions

# Detailed reconstruction analysis
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/model.pt \
    --config configs/config.yaml \
    --num-samples 20 \
    --output-dir detailed_evaluation
```

**Options**:
- `--checkpoint PATH` - Path to checkpoint file (required)
- `--config PATH` - Path to training config YAML (required)
- `--output-dir PATH` - Output directory for results (default: evaluation_results)
- `--num-samples N` - Number of test samples to evaluate (default: 5)
- `--device DEVICE` - Device to use: cpu or cuda (default: cpu)
- `--skip-weights` - Skip weight health check
- `--skip-reconstructions` - Skip reconstruction visualization
- `--help` - Show help message

**Output Structure**:
```
evaluation_results/
├── weight_health_report.txt          # NaN/Inf detection, weight statistics
├── reconstruction_log.txt            # Evaluation process log
└── reconstructions/
    ├── sample_1/
    │   ├── video_reconstruction.png
    │   ├── audio_reconstruction.png
    │   ├── text_reconstruction.png
    │   └── fmri_reconstruction.png
    ├── sample_2/
    │   └── ...
    └── ...
```

---

### 2. `evaluate_reconstructions.py` - Reconstruction Visualization

**Purpose**: Generate side-by-side visualizations of ground truth vs. model reconstructions for all modalities.

**Features**:
- Video frame reconstruction visualization
- Audio EnCodec code reconstruction plots
- Text CLIP embedding reconstruction comparisons
- fMRI voxel activation scatter plots with correlation metrics

**Usage**:
```bash
python scripts/evaluate_reconstructions.py \
    --checkpoint checkpoints_local/tensor02_fixed_lr/best_checkpoint.pt \
    --config configs/training/tensor02_test_50epoch_config.yaml \
    --output-dir reconstruction_results \
    --num-samples 5 \
    --device cpu
```

**Arguments**:
- `--checkpoint` - Path to checkpoint file (required)
- `--config` - Path to training config YAML (required)
- `--output-dir` - Output directory (default: reconstruction_results)
- `--num-samples` - Number of samples to evaluate (default: 5)
- `--device` - Device: cpu or cuda (default: cpu)

**Output**:
Creates one directory per sample containing PNG visualizations for each modality.

---

### 3. `examine_weights.py` - Weight Health Diagnostic

**Purpose**: Analyze model checkpoint weights for potential training issues.

**Features**:
- NaN and Inf detection
- Weight statistics (mean, std, min, max per layer)
- Unusual weight distribution flagging
- Optimizer state health check

**Usage**:
```bash
python scripts/examine_weights.py checkpoints/model.pt
```

**Output**:
Prints detailed report to stdout including:
- Checkpoint metadata
- Top layers by absolute max weight
- Top layers by standard deviation
- Critical issues (NaN, Inf, dead neurons)
- Overall statistics
- Optimizer state health

---

## Evaluation Workflow

### Quick Health Check
```bash
# Just check if checkpoint is loadable and has no NaN/Inf
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/model.pt \
    --config configs/config.yaml \
    --skip-reconstructions
```

### Standard Evaluation
```bash
# Full evaluation with default settings
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/model.pt \
    --config configs/config.yaml
```

### Detailed Analysis
```bash
# Comprehensive evaluation with many samples
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/model.pt \
    --config configs/config.yaml \
    --num-samples 20 \
    --output-dir detailed_evaluation

# Then examine specific visualizations
open detailed_evaluation/reconstructions/sample_1/fmri_reconstruction.png
```

---

## Interpreting Results

### Weight Health Report

**Good signs**:
- `Layers with NaN: 0`
- `Layers with Inf: 0`
- `Optimizer state healthy`
- Layer stds in reasonable range (0.01 - 10.0)

**Warning signs**:
- Very large weights (abs max > 1000) in non-BatchNorm layers
- Very small stds (< 1e-6) indicating dead neurons
- NaN or Inf values anywhere

### Reconstruction Quality

**Video**:
- Look for recognizable shapes and patterns
- Check if colors are reasonable
- Compare level of detail with ground truth

**Audio** (EnCodec codes):
- Codes should follow similar patterns to ground truth
- Look for temporal structure matching

**Text** (CLIP embeddings):
- Overlay plot should show correlation
- Reconstructed embeddings should track ground truth

**fMRI**:
- **Correlation metric** is key: higher is better
- Scatter plot should show positive slope
- Early training: correlation near zero (expected)
- Well-trained model: correlation > 0.1

---

## Common Issues

### Checkpoint Not Found
```
✗ Checkpoint file not found: checkpoints/model.pt
```
**Solution**: Verify checkpoint path and ensure file was synced from cluster.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use `--device cpu` or reduce `--num-samples`.

### Missing Dependencies
```
ModuleNotFoundError: No module named 'nibabel'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

---

## Examples

### Evaluate Best Model from Tensor02 Training
```bash
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints_local/tensor02_fixed_lr/best_checkpoint.pt \
    --config configs/training/tensor02_test_50epoch_config.yaml \
    --output-dir tensor02_evaluation \
    --num-samples 10
```

### Compare Multiple Checkpoints
```bash
# Evaluate checkpoint from epoch 10
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/epoch_10.pt \
    --config configs/config.yaml \
    --output-dir eval_epoch10

# Evaluate checkpoint from epoch 20
./scripts/evaluate_model_status.sh \
    --checkpoint checkpoints/epoch_20.pt \
    --config configs/config.yaml \
    --output-dir eval_epoch20

# Compare results
diff -u eval_epoch10/weight_health_report.txt eval_epoch20/weight_health_report.txt
```

### Production Model Validation
```bash
# Before deploying a model, run comprehensive evaluation
./scripts/evaluate_model_status.sh \
    --checkpoint production_model.pt \
    --config production_config.yaml \
    --num-samples 50 \
    --output-dir production_validation \
    --device cuda

# Review results
cat production_validation/weight_health_report.txt
ls production_validation/reconstructions/
```

---

## Tips

1. **Start small**: Use `--num-samples 1` for quick tests, then increase for thorough evaluation.

2. **Use GPUs when available**: Add `--device cuda` for faster evaluation on systems with CUDA.

3. **Save reports**: Evaluation reports are saved to disk, so you can review them later without re-running.

4. **Compare models**: Run evaluations on different checkpoints and compare the generated reports.

5. **Monitor training**: Run quick evaluations periodically during training to catch issues early.

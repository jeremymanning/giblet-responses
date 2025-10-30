# Contributing to Giblet-Responses

Thank you for contributing to the Giblet-Responses project! This guide will help you set up your development environment and understand our workflow.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/giblet-responses.git
cd giblet-responses
git remote add upstream https://github.com/ContextLab/giblet-responses.git
```

### 2. Set Up Development Environment

```bash
# Automated setup (recommended)
./setup_environment.sh

# Activate environment
conda activate giblet-py311
```

### 3. Verify Installation

```bash
# Run test suite
./run_giblet.sh --task test

# Validate data
./run_giblet.sh --task validate_data
```

## Development Workflow

### Local Development Cycle

1. **Create a new branch** for your feature/fix:
   ```bash
   git checkout -b feature-name
   ```

2. **Make your changes** and test locally:
   ```bash
   # Edit code
   vim giblet/models/autoencoder.py

   # Run relevant tests
   pytest tests/models/test_autoencoder.py -v

   # Test with small training run (1-2 epochs)
   ./run_giblet.sh --task train --config test_config.yaml --gpus 1
   ```

3. **Commit your changes** with descriptive messages:
   ```bash
   git add giblet/models/autoencoder.py
   git commit -m "Improve autoencoder bottleneck layer

   - Add batch normalization to bottleneck
   - Reduces overfitting on small datasets
   - Maintains 8000-dim bottleneck size

   Refs #XX"
   ```

4. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature-name
   ```

### Cluster Training Workflow

For computationally expensive experiments requiring GPUs:

1. **Test locally first** with minimal config:
   ```bash
   ./run_giblet.sh --task train --config test_config.yaml --gpus 1
   ```

2. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Add new experiment configuration"
   git push
   ```

3. **Launch cluster training:**
   ```bash
   # 8-GPU training on tensor01
   ./remote_train.sh --cluster tensor01 \
     --config cluster_train_config.yaml \
     --gpus 8 \
     --name experiment_name
   ```

4. **Monitor training:**
   ```bash
   # Check status
   ./check_remote_status.sh --cluster tensor01

   # Attach to running session
   ssh f002d6b@tensor01.dartmouth.edu
   screen -r experiment_name

   # Detach with: Ctrl+A, then D
   ```

5. **Retrieve results** when training completes:
   ```bash
   sshpass -p <PASSWORD> rsync -avz \
     f002d6b@tensor01.dartmouth.edu:~/giblet-responses/checkpoints/ \
     ./checkpoints/
   ```

### Cluster Best Practices

**Session Naming:**
- Use descriptive names: `exp_bottleneck_4000`, `baseline_run_v2`
- Avoid generic names like `test` or `training`
- Include your initials for multi-user clusters: `jm_experiment_1`

**Resource Management:**
- Check cluster availability before launching: `./check_remote_status.sh --cluster tensor01`
- Kill finished sessions: `./remote_train.sh --cluster tensor01 --name NAME --kill`
- Clean up old checkpoints to save disk space

**Monitoring:**
- Check GPU utilization regularly to ensure training progressing
- Monitor logs for errors: `tail -f logs/training_NAME_*.log`
- Use `check_remote_status.sh` for quick status checks

## Code Style and Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings to all classes and functions
- Keep functions focused and modular

### Testing Requirements

**All new code must include tests:**

```python
def test_new_feature():
    """Test new feature with real data."""
    # NO MOCKS - use real function calls
    result = new_feature(real_input)
    assert result == expected_output
```

**Key Testing Principles:**
- NO MOCK OBJECTS - always use real function calls
- Test with real external APIs (when feasible)
- Test with actual data files
- Test edge cases and error conditions
- All tests must pass before PR approval

### Documentation

**Update documentation when changing:**
- Public APIs → Update docstrings
- Configuration options → Update YAML examples
- User-facing features → Update SETUP.md and README.md
- Internal architecture → Update CLAUDE.md

## Pull Request Process

### Before Submitting

1. **Run full test suite:**
   ```bash
   pytest tests/ -v
   ```

2. **Run code quality checks** (if applicable):
   ```bash
   flake8 giblet/
   mypy giblet/
   ```

3. **Update relevant documentation**

4. **Test training pipeline** if you modified training code:
   ```bash
   ./run_giblet.sh --task train --config test_config.yaml --gpus 1
   ```

### PR Template

Include in your PR description:
- **What**: Brief description of changes
- **Why**: Motivation/issue being addressed
- **How**: Technical approach
- **Testing**: How changes were tested
- **Refs**: Link to relevant issues

Example:
```markdown
## Add Multi-Scale Video Processing

**What:** Implements multi-scale video feature extraction

**Why:** Single-scale features miss temporal dynamics (Issue #15)

**How:**
- Added temporal convolution layers
- Multiple time windows (1s, 5s, 15s)
- Concatenate features from all scales

**Testing:**
- Unit tests: tests/data/test_video_multiscale.py
- Integration: Trained 5 epochs with test_config.yaml
- Validated output dimensions match expected

**Refs:** Fixes #15
```

### Review Process

1. At least one team member must review
2. All tests must pass
3. Documentation must be updated
4. Address review feedback
5. Maintainer will merge when ready

## Common Tasks

### Adding a New Dependency

1. Add to `requirements_conda.txt` with exact version
2. Test installation locally
3. Update documentation
4. Commit changes:
   ```bash
   git add requirements_conda.txt
   git commit -m "Add new-package for feature X"
   ```

### Debugging Cluster Issues

1. **Check logs:**
   ```bash
   ssh f002d6b@tensor01.dartmouth.edu
   cd ~/giblet-responses
   tail -100 logs/training_NAME_*.log
   ```

2. **Check GPU utilization:**
   ```bash
   nvidia-smi
   ```

3. **Verify environment:**
   ```bash
   conda activate giblet-py311
   python -c "import giblet; import torch; print(torch.cuda.is_available())"
   ```

4. **Kill and restart:**
   ```bash
   ./remote_train.sh --cluster tensor01 --name NAME --kill
   ./remote_train.sh --cluster tensor01 --config CONFIG --gpus 8 --name NAME
   ```

### Running Specific Tests

```bash
# Single test file
pytest tests/models/test_encoder.py -v

# Single test function
pytest tests/models/test_encoder.py::test_forward_pass -v

# With debugging output
pytest tests/models/test_encoder.py -v -s

# Stop at first failure
pytest tests/ -x
```

## Project Structure

```
giblet-responses/
├── giblet/              # Main package
│   ├── data/           # Data processing (video, audio, fMRI, text)
│   ├── models/         # Neural network models
│   ├── training/       # Training infrastructure
│   ├── utils/          # Utilities and visualization
│   └── alignment/      # Hyperalignment tools
├── scripts/            # Training and utility scripts
│   ├── train.py        # Main training script
│   └── cluster/        # Cluster-specific utilities
├── examples/           # Example configs and demos
├── tests/              # Test suite
├── data/               # Dataset (gitignored)
├── checkpoints/        # Model checkpoints (gitignored)
└── logs/               # Training logs (gitignored)
```

## Communication

### Channels

- **GitHub Issues**: Bug reports, feature requests, discussions
- **Slack**: [#giblet-responses](https://context-lab.slack.com/archives/C020V4HJFT4)
- **Pull Requests**: Code review and feedback

### Reporting Issues

When reporting bugs, include:
- Description of expected vs. actual behavior
- Steps to reproduce
- Error messages and stack traces
- Environment info (OS, Python version, GPU)
- Relevant configuration files

## Questions?

- Check [SETUP.md](SETUP.md) for setup help
- Check [CLAUDE.md](CLAUDE.md) for project details
- Ask on Slack: [#giblet-responses](https://context-lab.slack.com/archives/C020V4HJFT4)
- Open a GitHub issue

Thank you for contributing! 🧠

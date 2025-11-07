# examples/ - Example Scripts and Demos

This directory contains example scripts demonstrating how to use the giblet-responses package for training, validation, and visualization.

## Quick Start

**New users**: Start with [QUICKSTART.md](QUICKSTART.md) for a rapid introduction to the codebase.

**Validation**: See [README_VALIDATION.md](README_VALIDATION.md) for comprehensive validation scripts.

## Example Categories

### Training Examples

| Script | Purpose | Usage |
|--------|---------|-------|
| [train_example.py](train_example.py) | Basic training loop example | `python examples/train_example.py` |
| [train_config.yaml](train_config.yaml) | Example training configuration | Used by train_example.py |

---

### Validation Scripts

Comprehensive validation for each modality. See [README_VALIDATION.md](README_VALIDATION.md) for details.

| Script | Validates | Key Tests |
|--------|-----------|-----------|
| [validate_video.py](validate_video.py) | Video processing pipeline | Frame extraction, temporal concat, dimensions |
| [validate_audio.py](validate_audio.py) | Audio processing (EnCodec) | EnCodec encoding/decoding, temporal alignment |
| [validate_text.py](validate_text.py) | Text processing | Embeddings, temporal alignment, timing |
| [validate_fmri.py](validate_fmri.py) | fMRI processing | NIfTI loading, masking, dimension checks |
| [validate_all_modalities.py](validate_all_modalities.py) | Complete pipeline | End-to-end data → model → loss |

**Run all validations:**
```bash
python examples/validate_all_modalities.py
```

---

### Demo Scripts

Interactive demonstrations of core functionality.

| Script | Demonstrates | Key Features |
|--------|--------------|---------------|
| [demo_decoder.py](demo_decoder.py) | Decoder architecture | Forward pass, reconstruction |
| [demo_sync.py](demo_sync.py) | Temporal synchronization | HRF convolution, alignment |
| [encodec_audio_encoder_demo.py](encodec_audio_encoder_demo.py) | EnCodec audio processing | Encoding, decoding, quality |
| [hrf_convolution_example.py](hrf_convolution_example.py) | HRF theory and usage | Visualization, mathematical properties |
| [video_temporal_concatenation_demo.py](video_temporal_concatenation_demo.py) | Video temporal concat | Frame aggregation, TR binning |

**Example usage:**
```bash
python examples/hrf_convolution_example.py
python examples/video_temporal_concatenation_demo.py
```

---

### Visualization Examples

Generate architecture diagrams and visualizations.

| Script | Purpose | Output |
|--------|---------|--------|
| [visualize_autoencoder_plotneuralnet.py](visualize_autoencoder_plotneuralnet.py) | PlotNeuralNet diagram | PDF architecture diagram |
| [visualize_autoencoder_torchview.py](visualize_autoencoder_torchview.py) | Torchview diagram | PNG/SVG architecture diagram |
| [create_network_diagram.py](create_network_diagram.py) | Custom diagram | Publication-quality figures |
| [generate_network_diagram.py](generate_network_diagram.py) | Automated diagram generation | LaTeX + PlotNeuralNet |

**Example:**
```bash
# Generate architecture diagram with PlotNeuralNet
python examples/visualize_autoencoder_plotneuralnet.py

# Generate with torchview
python examples/visualize_autoencoder_torchview.py
```

---

## Getting Started

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# For visualization examples
pip install torchview

# For PlotNeuralNet examples
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

### 2. Download Data

```bash
# From project root
./download_data_from_dropbox.sh
```

### 3. Run Validation

```bash
# Validate all modalities
python examples/validate_all_modalities.py

# This will test:
# - Video processing
# - Audio processing (EnCodec)
# - Text processing
# - fMRI processing
# - Complete pipeline
```

### 4. Try a Demo

```bash
# HRF convolution demo
python examples/hrf_convolution_example.py

# Video temporal concatenation
python examples/video_temporal_concatenation_demo.py

# EnCodec audio processing
python examples/encodec_audio_encoder_demo.py
```

### 5. Train a Model

```bash
# Basic training example
python examples/train_example.py

# Or use the full training script
./run_giblet.sh --task train --config examples/train_config.yaml --gpus 1
```

---

## Example Workflow

### Complete Pipeline Test

```bash
# 1. Validate data processing
python examples/validate_all_modalities.py

# 2. Visualize architecture
python examples/visualize_autoencoder_torchview.py

# 3. Run training example
python examples/train_example.py

# 4. Visualize HRF convolution
python examples/hrf_convolution_example.py
```

### Development Workflow

```bash
# 1. Validate specific modality
python examples/validate_video.py

# 2. Run modality demo
python examples/video_temporal_concatenation_demo.py

# 3. Test in full pipeline
python examples/validate_all_modalities.py
```

---

## Script Details

### Training Example ([train_example.py](train_example.py))

Minimal training loop demonstrating:
- Dataset loading
- Model initialization
- Training loop with loss computation
- Validation
- Checkpoint saving

**Usage:**
```bash
python examples/train_example.py
```

**Configuration:**
Edit [train_config.yaml](train_config.yaml) to modify:
- Batch size
- Learning rate
- Number of epochs
- Model hyperparameters

---

### HRF Convolution Example ([hrf_convolution_example.py](hrf_convolution_example.py))

Visualizes HRF theory and application:
- Plots canonical HRF kernel
- Shows stimulus → HRF convolution → BOLD prediction
- Demonstrates temporal delay (5-6 seconds)
- Explains why HRF convolution is necessary

**Usage:**
```bash
python examples/hrf_convolution_example.py

# Output: hrf_convolution_demo.png
```

---

### Video Temporal Concatenation Demo ([video_temporal_concatenation_demo.py](video_temporal_concatenation_demo.py))

Demonstrates video temporal concatenation strategy:
- Shows frame aggregation within TRs
- Visualizes concatenation process
- Explains dimension calculations

**Usage:**
```bash
python examples/video_temporal_concatenation_demo.py

# Output: video_temporal_concat_demo.png
```

---

### EnCodec Audio Demo ([encodec_audio_encoder_demo.py](encodec_audio_encoder_demo.py))

Interactive EnCodec demonstration:
- Loads audio from video
- Encodes with EnCodec
- Decodes back to audio
- Compares original vs reconstructed
- Saves output audio files

**Usage:**
```bash
python examples/encodec_audio_encoder_demo.py

# Outputs:
# - encodec_codes.npy (encoded representation)
# - reconstructed_audio.wav (decoded audio)
```

---

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: data/stimuli_Sherlock.m4v`
- **Solution**: Run `./download_data_from_dropbox.sh` from project root

**Issue**: `ImportError: No module named 'transformers'`
- **Solution**: Install EnCodec dependencies: `pip install transformers`

**Issue**: `ImportError: No module named 'torchview'`
- **Solution**: Install visualization tools: `pip install torchview`

**Issue**: Validation script fails with dimension mismatch
- **Solution**: Check frame_skip setting in config matches expected dimensions

---

## Dependencies

### Required

- PyTorch >= 2.0
- nibabel (fMRI data)
- nilearn (HRF modeling)
- librosa (audio processing)
- transformers (EnCodec)
- sentence-transformers (text embeddings)
- pandas, numpy, scipy

### Optional (Visualization)

- torchview (architecture diagrams)
- matplotlib (plotting)
- PlotNeuralNet (publication-quality diagrams)

**Install all:**
```bash
pip install -r requirements.txt
pip install torchview  # Optional
```

---

## Related Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Rapid introduction for new users
- **[README_VALIDATION.md](README_VALIDATION.md)** - Validation script details
- **[../giblet/data/README.md](../giblet/data/README.md)** - Data pipeline documentation
- **[../giblet/models/README.md](../giblet/models/README.md)** - Model architecture documentation
- **[../tests/README.md](../tests/README.md)** - Test suite documentation

For questions or issues, see the main project [README.md](../README.md) or open an issue on GitHub.

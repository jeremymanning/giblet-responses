# Session Summary: Multimodal Alignment Research

**Date**: 2025-10-28
**Task**: Research temporal alignment, audio processing, text embeddings, and fMRI formatting for autoencoder project

## Session Objectives

Research and provide technical recommendations for:
1. Temporal alignment of multimodal data (video @ 25fps, audio @ 44.1kHz, fMRI @ 1Hz, irregular text)
2. Audio mel spectrogram extraction and reconstruction best practices
3. Text embedding models and recovery methods
4. fMRI data formatting (CMU format, shared masks, inter-subject variability)

## Key Findings

### 1. Temporal Alignment

**Main Strategy**: TR binning + HRF convolution

- **Recent research** (2024-2025) uses aggregation into TR bins followed by hemodynamic response function convolution
- **AFIRE Framework** (2024): Aggregate multimodal features at 2 Hz, then bin to TR resolution
- **Dynamic Temporal Alignment** (2025): For decoding, use exponentially weighted multi-frame fusion
- **Critical**: Account for ~6 second hemodynamic lag using HRF convolution (handled automatically)

**Implementation**:
- Use `nilearn.glm.first_level.compute_regressor()` for HRF convolution
- Use `nilearn.glm.first_level.glover_hrf()` to generate HRF
- Oversample HRF at 50x for accurate convolution
- Aggregate high-frequency features (video, audio) into TR-sized bins before or after HRF

### 2. Audio Processing

**Extraction Parameters**:
- Sample rate: 22050 Hz (standard for vocoders)
- n_mels: 80 (HiFi-GAN standard) or 128 (higher quality)
- n_fft: 1024
- hop_length: 256
- fmin: 0.0, fmax: 8000.0

**Reconstruction**:
- **DO NOT use Griffin-Lim** - produces poor quality with artifacts
- **USE HiFi-GAN vocoder** - state-of-the-art, fast inference, high quality
  - Available via `torchaudio.prototype.pipelines.HIFIGAN_VOCODER_V3_LJSPEECH`
  - Pretrained models work well without fine-tuning
  - 167.9x faster than real-time on single GPU
- **Alternative**: WaveGlow (also good quality but slower)

**Key Libraries**:
- `librosa`: Mel spectrogram extraction
- `torchaudio`: HiFi-GAN vocoder models

### 3. Text Embeddings

**Best Models** (MTEB 2024-2025):
1. `BAAI/bge-large-en-v1.5` - State-of-the-art, 1024-dim
2. `sentence-transformers/all-mpnet-base-v2` - Balanced, 768-dim
3. `NV-Embed-v2` - Multimodal-aware (based on Mistral 7B)

**Embedding Recovery**:
- **Practical approach**: Nearest-neighbor retrieval from original annotation database
  - Use `sklearn.neighbors.NearestNeighbors` with cosine metric
  - Build index of all original text annotations
  - Query with predicted embeddings to retrieve closest match

- **Advanced approach**: vec2text library
  - GitHub: https://github.com/vec2text/vec2text
  - Iterative refinement for semantic reconstruction
  - Recovers semantic meaning (not exact text)

**Important**: Embedding-to-text is semantically lossy. Embeddings preserve meaning but not exact wording.

### 4. fMRI Data Formatting

**CMU Format** (BrainIAK):
- **Definition**: (n_voxels, n_timepoints) - rows are voxels
- **nilearn format**: (n_timepoints, n_voxels) - rows are time
- **Conversion**: Simply transpose `.T`

**Shared Brain Mask**:
- Use `nilearn.masking.compute_multi_epi_mask()`
- Parameters:
  - threshold: 0.5-0.8 (fraction of subjects with signal at each voxel)
  - opening: 2 (morphological operation to remove noise)
  - connected: True (keep largest connected component)

**Inter-Subject Variability**:
Three standard approaches:
1. **Shared Response Model (SRM)** - BrainIAK
   - Maps individual brains to shared low-dimensional space
   - Use `brainiak.funcalign.srm.SRM`
   - 50-100 features typical

2. **Hyperalignment** - More flexible than SRM
   - Can align subsets of voxels
   - Use `brainiak.funcalign.rsrm.RSRM`

3. **Template-based** - MNI space alignment
   - Use fMRIPrep outputs (already in MNI space)
   - Most common for group analysis

## Generated Documents

1. **multimodal_alignment_recommendations.md** (38KB)
   - Comprehensive technical guide with code examples
   - Detailed explanations of each approach
   - Complete pipeline implementations
   - Reference links and citations

2. **quick_reference.md** (7KB)
   - Concise code snippets and parameters
   - Quick lookup table for critical values
   - Installation instructions
   - Pipeline overview

## Critical Parameters Summary

| Modality | Key Parameters |
|----------|---------------|
| Video | 25 fps → TR binning → HRF convolve |
| Audio | 22050 Hz, n_mels=80, n_fft=1024, hop=256 |
| Text | BAAI/bge-large-en-v1.5, 1024-dim |
| fMRI | TR=1s, shared mask threshold=0.5-0.8 |
| HRF | Glover model, oversampling=50 |
| SRM | 50-100 features |

## Recommended Tech Stack

**Core Libraries**:
- `nibabel`, `nilearn`, `brainiak` - fMRI processing
- `librosa`, `torchaudio` - Audio processing
- `sentence-transformers`, `vec2text` - Text embeddings
- `numpy`, `scipy`, `scikit-learn` - General scientific computing

## Key Insights

1. **Temporal alignment is non-trivial**: Must account for hemodynamic lag (~6s) using HRF convolution
2. **Audio reconstruction requires neural vocoders**: Griffin-Lim is insufficient
3. **Text embedding recovery is limited**: Can retrieve semantically similar text, not exact reconstruction
4. **fMRI format matters**: BrainIAK expects transposed format from nilearn
5. **Inter-subject alignment is crucial**: Use SRM for functional alignment across subjects

## Next Steps for Project

Based on this research, recommended implementation order:

1. Set up data preprocessing pipeline:
   - fMRI: Apply shared mask, convert to CMU format
   - Video: Extract frames at 25 fps
   - Audio: Extract mel spectrograms with recommended parameters
   - Text: Embed annotations with BAAI/bge-large-en-v1.5

2. Implement temporal alignment:
   - Aggregate all modalities to TR resolution (1 Hz)
   - Apply HRF convolution to stimulus features
   - Verify alignment across modalities

3. Set up reconstruction pipelines:
   - Video: Frame generation from latent features
   - Audio: HiFi-GAN vocoder for mel-to-audio
   - Text: Nearest-neighbor retrieval for annotation recovery
   - fMRI: SRM for across-subject generalization

4. Build autoencoder architecture:
   - Input: Aligned multimodal features at TR resolution
   - Middle layer: Match fMRI voxel count
   - Output: Reconstruct all modalities
   - Loss: Combination of reconstruction + fMRI alignment

## Web Search Results Summary

Conducted 10+ web searches covering:
- Recent papers (2024-2025) on fMRI-video encoding models
- Audio vocoder comparisons and best practices
- MTEB leaderboard and text embedding models
- BrainIAK documentation and tutorials
- Nilearn preprocessing workflows
- HiFi-GAN and neural vocoder implementations

Key papers referenced:
- AFIRE Framework (2024) - Multimodal fMRI encoding
- Dynamic Temporal Alignment (2025) - fMRI-to-video decoding
- Vec2Text (2024) - Text embedding inversion
- HiFi-GAN paper and implementation

## Files Created

- `/Users/jmanning/giblet-responses/notes/multimodal_alignment_recommendations.md`
- `/Users/jmanning/giblet-responses/notes/quick_reference.md`
- `/Users/jmanning/giblet-responses/notes/session_2025-10-28_multimodal_research.md` (this file)

## Session Statistics

- Duration: Single session
- Web searches: 10
- Documents generated: 3
- Total content: ~45KB of technical documentation
- Code examples: 30+ working snippets
- References cited: 15+ papers and libraries

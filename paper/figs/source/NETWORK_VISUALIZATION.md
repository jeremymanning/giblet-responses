# Multimodal Autoencoder Network Visualization

## Architecture Overview

This document describes the PlotNeuralNet visualization for the multimodal autoencoder architecture specified in issue #2.

### Architecture Layers

The network consists of 11 layers with parallel branches:

#### Encoder (Layers 1-6)

- **Layer 1: Input (3 parallel branches)**
  - Video Input: 90×160×3 = 43,200 dimensions
  - Audio Input: 2,048 mel frequency bins
  - Text Input: 1,024 embeddings

- **Layer 2A/B/C: Modality-Specific Encoders (parallel)**
  - 2A: Video Conv2D → 1,024 features
  - 2B: Audio Conv1D → 256 features
  - 2C: Text Linear → 256 features

- **Layer 3: Pooled Features (concatenation)**
  - Concatenates all modality features → 1,536 dimensions

- **Layer 4: Feature Convolution + ReLU**
  - Processes pooled features → 1,536 dimensions

- **Layer 5: Linear to Voxels**
  - Maps to brain voxel space → 85,810 dimensions

- **Layer 6: BOTTLENECK (MIDDLE/SMALLEST LAYER)**
  - Compressed representation → 8,000 dimensions
  - This is the central compression point

#### Decoder (Layers 7-11 - Symmetric to Encoder)

- **Layer 7: Expand from Bottleneck**
  - 8,000 → 2,048 dimensions

- **Layer 8: Feature Deconvolution + ReLU**
  - 2,048 → 4,096 dimensions

- **Layer 9: Unpool Features**
  - 4,096 → 8,192 dimensions

- **Layer 10A/B/C: Modality-Specific Decoders (parallel)**
  - 10A: Video Decoder → 1,024 features
  - 10B: Audio Decoder → 256 features
  - 10C: Text Decoder → 256 features

- **Layer 11: Outputs (3 parallel branches)**
  - Video Output: 43,200 dimensions (reconstructed video)
  - Audio Output: 2,048 dimensions (reconstructed audio)
  - Text Output: 1,024 dimensions (reconstructed text)

### Visualization Features

The diagram shows:

1. **Horizontal left-to-right flow**
2. **Parallel branches** at layers 2 and 10 clearly visible
3. **Bottleneck emphasis** in the middle (Layer 6) as the smallest layer
4. **Symmetric architecture** between encoder and decoder
5. **Color coding**:
   - Yellow/Red: Convolution layers
   - Blue: Fully connected layers
   - Red: Pooling layers
   - Green/Blue: Bottleneck (middle layer)
   - Orange: Text processing

### Key Architecture Principles

1. **Multimodal Input**: Three independent input streams
2. **Progressive Compression**: Layers 1-6 progressively compress information
3. **Bottleneck**: Layer 6 is the compressed "brain activity" representation
4. **Progressive Expansion**: Layers 7-11 progressively expand back to original space
5. **Parallel Processing**: Separate pathways for each modality before pooling

##  Generated Files

- `network.tex`: LaTeX source for the diagram
- `network.pdf`: Compiled diagram (if LaTeX compilation successful)
- `layers/`: PlotNeuralNet layer definitions (required for compilation)

## Manual Compilation Instructions

If automatic PDF compilation fails, you can compile manually:

```bash
cd paper/figs/source
pdflatex network.tex
```

Requirements:
- LaTeX installation (pdflatex)
- PlotNeuralNet layers directory (automatically copied)

### Installing LaTeX

- **macOS**: `brew install --cask mactex-no-gui`
- **Ubuntu**: `sudo apt-get install texlive-latex-extra`

## Integration with Paper

The generated `network.pdf` can be included in LaTeX documents:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figs/source/network.pdf}
    \caption{Multimodal autoencoder architecture with 11 layers.
             The encoder (Layers 1-6) compresses multimodal input to a
             bottleneck representation, while the decoder (Layers 7-11)
             reconstructs the original modalities.}
    \label{fig:network_architecture}
\end{figure}
```

## References

- Issue #2: Architecture specification
- Issue #18: PlotNeuralNet visualization
- PlotNeuralNet: https://github.com/HarisIqbal88/PlotNeuralNet

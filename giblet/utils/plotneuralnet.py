"""
PlotNeuralNet visualization utilities for the multimodal autoencoder.

This module provides functions to generate publication-quality network diagrams
using PlotNeuralNet's LaTeX/TikZ rendering system.

The architecture follows issue #2 specification:
- Layer 1: Input (video + audio + text)
- Layer 2A/B/C: Parallel modality-specific encoders
- Layer 3: Pooled multimodal features
- Layer 4: Feature convolution + ReLU
- Layer 5: Linear to 85,810 voxels
- Layer 6: Bottleneck (8,000 dims - SMALLEST layer)
- Layer 7-11: Symmetric decoder (mirrors layers 5-1)

Requires:
- PlotNeuralNet repository at /tmp/PlotNeuralNet
- LaTeX installation with pdflatex (optional for PDF compilation)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


# PlotNeuralNet library path
PLOTNEURALNET_PATH = "/tmp/PlotNeuralNet"


def generate_multimodal_architecture_latex(
    output_path: str,
    video_height: int = 90,
    video_width: int = 160,
    audio_mels: int = 2048,
    text_dim: int = 1024,
    n_voxels: int = 85810,
    bottleneck_dim: int = 8000,
    video_features: int = 1024,
    audio_features: int = 256,
    text_features: int = 256,
) -> str:
    """
    Generate LaTeX code for multimodal autoencoder architecture visualization.

    Creates a horizontal left-to-right flow diagram showing:
    1. Encoder layers (1-6) with parallel branches at layer 2
    2. Bottleneck layer (smallest, middle layer)
    3. Decoder layers (7-11) with parallel branches at layer 10

    Parameters
    ----------
    output_path : str
        Path to save the .tex file
    video_height : int, default=90
        Video frame height
    video_width : int, default=160
        Video frame width
    audio_mels : int, default=2048
        Number of mel frequency bins
    text_dim : int, default=1024
        Dimensionality of text embeddings
    n_voxels : int, default=85810
        Number of brain voxels
    bottleneck_dim : int, default=8000
        Dimensionality of bottleneck layer
    video_features : int, default=1024
        Video encoder output features
    audio_features : int, default=256
        Audio encoder output features
    text_features : int, default=256
        Text encoder output features

    Returns
    -------
    tex_path : str
        Path to generated .tex file

    Notes
    -----
    The visualization uses PlotNeuralNet's 3D box rendering to show:
    - Box height/depth: represents feature dimensionality
    - Box width: represents layer depth/complexity
    - Parallel branches: shown as separate paths that merge
    """
    # Calculate derived dimensions
    video_input_dim = video_height * video_width * 3  # 43,200
    pooled_dim = video_features + audio_features + text_features  # 1,536

    # Color definitions
    colors = """
\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}
\\def\\ConvReluColor{rgb:yellow,5;red,5;white,5}
\\def\\PoolColor{rgb:red,1;black,0.3}
\\def\\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\\def\\FcColor{rgb:blue,5;red,2.5;white,5}
\\def\\FcReluColor{rgb:blue,5;red,5;white,4}
\\def\\SoftmaxColor{rgb:magenta,5;black,7}
\\def\\BottleneckColor{rgb:green,5;blue,3;white,2}
\\def\\TextColor{rgb:orange,5;white,3}
"""

    # Build LaTeX document
    # Use relative path to layers directory (will be copied to same directory as .tex file)
    latex_code = (
        r"""\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}

"""
        + colors
        + r"""

\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

%%%% ENCODER %%%%

%%% Layer 1: Input (video, audio, text) %%%
% Video input
\pic[shift={(0,2,0)}] at (0,0,0) {
    Box={
        name=input_video,
        caption=Video Input,
        xlabel={{" "}},
        zlabel="""
        + str(video_input_dim)
        + r""",
        fill=\ConvColor,
        height=35,
        width=2,
        depth=35
    }
};

% Audio input
\pic[shift={(0,0,0)}] at (0,0,0) {
    Box={
        name=input_audio,
        caption=Audio Input,
        xlabel={{" "}},
        zlabel="""
        + str(audio_mels)
        + r""",
        fill=\ConvColor,
        height=25,
        width=2,
        depth=25
    }
};

% Text input
\pic[shift={(0,-2,0)}] at (0,0,0) {
    Box={
        name=input_text,
        caption=Text Input,
        xlabel={{" "}},
        zlabel="""
        + str(text_dim)
        + r""",
        fill=\TextColor,
        height=20,
        width=2,
        depth=20
    }
};

%%% Layer 2A: Video Encoder (Conv2D) %%%
\pic[shift={(2.5,2,0)}] at (0,0,0) {
    Box={
        name=video_conv,
        caption=Layer 2A,
        xlabel={{" "}},
        zlabel="""
        + str(video_features)
        + r""",
        fill=\ConvReluColor,
        height=30,
        width=3,
        depth=30
    }
};
\draw [connection] (input_video-east) -- node {\midarrow} (video_conv-west);

%%% Layer 2B: Audio Encoder (Conv1D) %%%
\pic[shift={(2.5,0,0)}] at (0,0,0) {
    Box={
        name=audio_conv,
        caption=Layer 2B,
        xlabel={{" "}},
        zlabel="""
        + str(audio_features)
        + r""",
        fill=\ConvReluColor,
        height=18,
        width=3,
        depth=18
    }
};
\draw [connection] (input_audio-east) -- node {\midarrow} (audio_conv-west);

%%% Layer 2C: Text Encoder (Linear) %%%
\pic[shift={(2.5,-2,0)}] at (0,0,0) {
    Box={
        name=text_linear,
        caption=Layer 2C,
        xlabel={{" "}},
        zlabel="""
        + str(text_features)
        + r""",
        fill=\FcReluColor,
        height=16,
        width=3,
        depth=16
    }
};
\draw [connection] (input_text-east) -- node {\midarrow} (text_linear-west);

%%% Layer 3: Pooled Features (Concatenation) %%%
\pic[shift={(5.5,0,0)}] at (0,0,0) {
    Box={
        name=pool_concat,
        caption=Layer 3,
        xlabel={{" "}},
        zlabel="""
        + str(pooled_dim)
        + r""",
        fill=\PoolColor,
        height=28,
        width=2,
        depth=28,
        opacity=0.6
    }
};
\draw [connection] (video_conv-east) -- node {\midarrow} (pool_concat-west);
\draw [connection] (audio_conv-east) -- node {\midarrow} (pool_concat-west);
\draw [connection] (text_linear-east) -- node {\midarrow} (pool_concat-west);

%%% Layer 4: Feature Convolution + ReLU %%%
\pic[shift={(8,0,0)}] at (0,0,0) {
    Box={
        name=feature_conv,
        caption=Layer 4,
        xlabel={{" "}},
        zlabel="""
        + str(pooled_dim)
        + r""",
        fill=\ConvReluColor,
        height=28,
        width=3,
        depth=28
    }
};
\draw [connection] (pool_concat-east) -- node {\midarrow} (feature_conv-west);

%%% Layer 5: Linear to Voxels %%%
\pic[shift={(11,0,0)}] at (0,0,0) {
    Box={
        name=to_voxels,
        caption=Layer 5,
        xlabel={{" "}},
        zlabel="""
        + str(n_voxels)
        + r""",
        fill=\FcColor,
        height=38,
        width=2,
        depth=38
    }
};
\draw [connection] (feature_conv-east) -- node {\midarrow} (to_voxels-west);

%%% Layer 6: Bottleneck (SMALLEST/MIDDLE LAYER) %%%
\pic[shift={(13.5,0,0)}] at (0,0,0) {
    Box={
        name=bottleneck,
        caption=Layer 6,
        xlabel={{" "}},
        zlabel="""
        + str(bottleneck_dim)
        + r""",
        fill=\BottleneckColor,
        height=22,
        width=4,
        depth=22,
        opacity=0.8
    }
};
\draw [connection] (to_voxels-east) -- node {\midarrow} (bottleneck-west);

%%%% DECODER (SYMMETRIC) %%%%

%%% Layer 7: Expand from Bottleneck %%%
\pic[shift={(17,0,0)}] at (0,0,0) {
    Box={
        name=from_bottleneck,
        caption=Layer 7,
        xlabel={{" "}},
        zlabel=2048,
        fill=\FcColor,
        height=26,
        width=2,
        depth=26
    }
};
\draw [connection] (bottleneck-east) -- node {\midarrow} (from_bottleneck-west);

%%% Layer 8: Feature Deconv + ReLU %%%
\pic[shift={(19.5,0,0)}] at (0,0,0) {
    Box={
        name=feature_deconv,
        caption=Layer 8,
        xlabel={{" "}},
        zlabel=4096,
        fill=\UnpoolColor,
        height=28,
        width=3,
        depth=28,
        opacity=0.6
    }
};
\draw [connection] (from_bottleneck-east) -- node {\midarrow} (feature_deconv-west);

%%% Layer 9: Unpool Features %%%
\pic[shift={(22.5,0,0)}] at (0,0,0) {
    Box={
        name=unpool,
        caption=Layer 9,
        xlabel={{" "}},
        zlabel=8192,
        fill=\UnpoolColor,
        height=30,
        width=2,
        depth=30,
        opacity=0.6
    }
};
\draw [connection] (feature_deconv-east) -- node {\midarrow} (unpool-west);

%%% Layer 10A: Video Decoder %%%
\pic[shift={(25.5,2,0)}] at (0,0,0) {
    Box={
        name=video_decoder,
        caption=Layer 10A,
        xlabel={{" "}},
        zlabel="""
        + str(video_features)
        + r""",
        fill=\UnpoolColor,
        height=30,
        width=3,
        depth=30,
        opacity=0.6
    }
};
\draw [connection] (unpool-east) -- node {\midarrow} (video_decoder-west);

%%% Layer 10B: Audio Decoder %%%
\pic[shift={(25.5,0,0)}] at (0,0,0) {
    Box={
        name=audio_decoder,
        caption=Layer 10B,
        xlabel={{" "}},
        zlabel="""
        + str(audio_features)
        + r""",
        fill=\UnpoolColor,
        height=18,
        width=3,
        depth=18,
        opacity=0.6
    }
};
\draw [connection] (unpool-east) -- node {\midarrow} (audio_decoder-west);

%%% Layer 10C: Text Decoder %%%
\pic[shift={(25.5,-2,0)}] at (0,0,0) {
    Box={
        name=text_decoder,
        caption=Layer 10C,
        xlabel={{" "}},
        zlabel="""
        + str(text_features)
        + r""",
        fill=\FcReluColor,
        height=16,
        width=3,
        depth=16
    }
};
\draw [connection] (unpool-east) -- node {\midarrow} (text_decoder-west);

%%% Layer 11: Outputs %%%
% Video output
\pic[shift={(29,2,0)}] at (0,0,0) {
    Box={
        name=output_video,
        caption=Video Output,
        xlabel={{" "}},
        zlabel="""
        + str(video_input_dim)
        + r""",
        fill=\ConvColor,
        height=35,
        width=2,
        depth=35
    }
};
\draw [connection] (video_decoder-east) -- node {\midarrow} (output_video-west);

% Audio output
\pic[shift={(29,0,0)}] at (0,0,0) {
    Box={
        name=output_audio,
        caption=Audio Output,
        xlabel={{" "}},
        zlabel="""
        + str(audio_mels)
        + r""",
        fill=\ConvColor,
        height=25,
        width=2,
        depth=25
    }
};
\draw [connection] (audio_decoder-east) -- node {\midarrow} (output_audio-west);

% Text output
\pic[shift={(29,-2,0)}] at (0,0,0) {
    Box={
        name=output_text,
        caption=Text Output,
        xlabel={{" "}},
        zlabel="""
        + str(text_dim)
        + r""",
        fill=\TextColor,
        height=20,
        width=2,
        depth=20
    }
};
\draw [connection] (text_decoder-east) -- node {\midarrow} (output_text-west);

\end{tikzpicture}
\end{document}
"""
    )

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(latex_code)

    return str(output_path)


def compile_latex_to_pdf(
    tex_path: str, output_dir: Optional[str] = None, cleanup: bool = True
) -> Optional[str]:
    """
    Compile LaTeX file to PDF using pdflatex.

    Parameters
    ----------
    tex_path : str
        Path to .tex file
    output_dir : str, optional
        Directory for output PDF (default: same as .tex file)
    cleanup : bool, default=True
        Remove intermediate files (.aux, .log, etc.)

    Returns
    -------
    pdf_path : str or None
        Path to generated PDF, or None if compilation failed

    Notes
    -----
    Requires pdflatex to be installed and available in PATH.
    On macOS: brew install --cask mactex-no-gui
    On Ubuntu: sudo apt-get install texlive-latex-extra
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_path}")

    # Check if pdflatex is available
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: pdflatex not found. PDF compilation skipped.")
        print("Install LaTeX:")
        print("  macOS: brew install --cask mactex-no-gui")
        print("  Ubuntu: sudo apt-get install texlive-latex-extra")
        return None

    # Set output directory
    if output_dir is None:
        output_dir = tex_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Copy PlotNeuralNet layers to output directory if not already there
    plotneuralnet_layers = Path(PLOTNEURALNET_PATH) / "layers"
    target_layers = output_dir / "layers"
    if plotneuralnet_layers.exists() and not target_layers.exists():
        import shutil

        shutil.copytree(plotneuralnet_layers, target_layers)

    # Compile LaTeX (run twice for proper references)
    try:
        for _ in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    str(output_dir),
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
                cwd=str(tex_path.parent),
            )

            if result.returncode != 0:
                print("LaTeX compilation error:")
                print(result.stdout)
                print(result.stderr)
                return None

        pdf_path = output_dir / (tex_path.stem + ".pdf")

        # Cleanup intermediate files
        if cleanup:
            for ext in [".aux", ".log", ".out"]:
                aux_file = output_dir / (tex_path.stem + ext)
                if aux_file.exists():
                    aux_file.unlink()

        if pdf_path.exists():
            return str(pdf_path)
        else:
            return None

    except Exception as e:
        print(f"Error during compilation: {e}")
        return None


def generate_network_diagram(
    output_pdf_path: str,
    architecture_params: Optional[Dict[str, Any]] = None,
    keep_tex: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Generate complete network diagram (LaTeX + PDF).

    This is the main convenience function that generates both .tex and .pdf files
    for the multimodal autoencoder architecture.

    Parameters
    ----------
    output_pdf_path : str
        Path for output PDF (e.g., "paper/figs/source/network.pdf")
    architecture_params : dict, optional
        Architecture parameters (video_height, audio_mels, etc.)
        If None, uses default values matching the model
    keep_tex : bool, default=True
        Keep the intermediate .tex file

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'tex_path': Path to .tex file (if keep_tex=True)
        - 'pdf_path': Path to .pdf file (or None if compilation failed)
        - 'success': True if PDF was generated successfully

    Examples
    --------
    >>> from giblet.utils.plotneuralnet import generate_network_diagram
    >>> result = generate_network_diagram('paper/figs/source/network.pdf')
    >>> if result['success']:
    ...     print(f"PDF created at: {result['pdf_path']}")
    """
    # Set default architecture parameters
    if architecture_params is None:
        architecture_params = {}

    # Generate .tex file
    output_pdf_path = Path(output_pdf_path)
    tex_path = output_pdf_path.with_suffix(".tex")

    print(f"Generating LaTeX code...")
    tex_file = generate_multimodal_architecture_latex(
        output_path=str(tex_path), **architecture_params
    )
    print(f"LaTeX saved to: {tex_file}")

    # Compile to PDF
    print(f"Compiling PDF...")
    pdf_path = compile_latex_to_pdf(
        tex_path=tex_file, output_dir=str(output_pdf_path.parent), cleanup=True
    )

    result = {
        "tex_path": str(tex_path) if keep_tex else None,
        "pdf_path": pdf_path,
        "success": pdf_path is not None,
    }

    if pdf_path:
        print(f"PDF created at: {pdf_path}")
    else:
        print("PDF compilation failed or pdflatex not available.")
        print(f"LaTeX source available at: {tex_file}")

    # Clean up .tex if requested
    if not keep_tex and Path(tex_file).exists():
        Path(tex_file).unlink()
        result["tex_path"] = None

    return result


# Convenience function for issue #18
def create_paper_figure(paper_dir: str = "/Users/jmanning/giblet-responses/paper"):
    """
    Create network diagram for paper (refs #18).

    Parameters
    ----------
    paper_dir : str, default="/Users/jmanning/giblet-responses/paper"
        Path to paper directory

    Returns
    -------
    result : dict
        Dictionary with 'tex_path', 'pdf_path', and 'success' keys
    """
    output_path = Path(paper_dir) / "figs" / "source" / "network.pdf"
    return generate_network_diagram(str(output_path))

"""
Alignment module for multimodal temporal synchronization and HRF convolution.

Provides tools for:
- Temporal synchronization across modalities
- HRF (Hemodynamic Response Function) convolution
- Alignment of video, audio, text, and fMRI data to common TR grid
"""

from .hrf import (
    apply_hrf,
    convolve_with_padding,
    get_canonical_hrf,
    get_hrf_peak_latency,
)
from .sync import align_all_modalities, get_alignment_info

__all__ = [
    # HRF functions
    "get_canonical_hrf",
    "apply_hrf",
    "convolve_with_padding",
    "get_hrf_peak_latency",
    # Synchronization functions
    "align_all_modalities",
    "get_alignment_info",
]

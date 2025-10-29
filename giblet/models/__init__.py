"""
Models module for giblet autoencoder.

Contains encoder and decoder architectures for multimodal fMRI reconstruction.
"""

from .encoder import (
    VideoEncoder,
    AudioEncoder,
    TextEncoder,
    SherlockEncoder,
    create_encoder
)

__all__ = [
    'VideoEncoder',
    'AudioEncoder',
    'TextEncoder',
    'SherlockEncoder',
    'create_encoder'
]

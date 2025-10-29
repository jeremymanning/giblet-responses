"""
Models module for giblet autoencoder.

Contains encoder, decoder, and full autoencoder architectures for multimodal fMRI reconstruction.
"""

from .encoder import (
    VideoEncoder,
    AudioEncoder,
    TextEncoder,
    SherlockEncoder,
    create_encoder
)
from .decoder import MultimodalDecoder
from .autoencoder import (
    SherlockAutoencoder,
    create_autoencoder,
    prepare_for_distributed
)

__all__ = [
    'VideoEncoder',
    'AudioEncoder',
    'TextEncoder',
    'SherlockEncoder',
    'create_encoder',
    'MultimodalDecoder',
    'SherlockAutoencoder',
    'create_autoencoder',
    'prepare_for_distributed'
]

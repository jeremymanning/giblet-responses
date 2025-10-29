"""
Models module for giblet autoencoder.

Contains encoder, decoder, and full autoencoder architectures for multimodal fMRI reconstruction.
"""

from .encoder import (
    VideoEncoder,
    AudioEncoder,
    TextEncoder,
    MultimodalEncoder,
    create_encoder
)
from .decoder import MultimodalDecoder
from .autoencoder import (
    MultimodalAutoencoder,
    create_autoencoder,
    prepare_for_distributed
)

__all__ = [
    'VideoEncoder',
    'AudioEncoder',
    'TextEncoder',
    'MultimodalEncoder',
    'create_encoder',
    'MultimodalDecoder',
    'MultimodalAutoencoder',
    'create_autoencoder',
    'prepare_for_distributed'
]

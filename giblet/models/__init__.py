"""
Models module for giblet autoencoder.

Contains encoder, decoder, and full autoencoder architectures for multimodal fMRI reconstruction.
"""

from .autoencoder import (
    MultimodalAutoencoder,
    create_autoencoder,
    prepare_for_distributed,
)
from .decoder import MultimodalDecoder
from .encoder import (
    AudioEncoder,
    MultimodalEncoder,
    TextEncoder,
    VideoEncoder,
    create_encoder,
)

# Alias for backwards compatibility
SherlockAutoencoder = MultimodalAutoencoder

__all__ = [
    "VideoEncoder",
    "AudioEncoder",
    "TextEncoder",
    "MultimodalEncoder",
    "create_encoder",
    "MultimodalDecoder",
    "MultimodalAutoencoder",
    "SherlockAutoencoder",  # Backwards compatibility alias
    "create_autoencoder",
    "prepare_for_distributed",
]

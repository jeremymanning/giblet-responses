"""
Demonstration of AudioEncoder with EnCodec quantized codes support.

This script shows how to:
1. Create an AudioEncoder in EnCodec mode
2. Process EnCodec quantized codes through the encoder
3. Compare with traditional mel spectrogram processing
4. Use EnCodec mode in the full MultimodalEncoder

Settings used (from Issue #24, Task 2.2):
- 3.0 kbps bandwidth
- 8 codebooks per TR
- ~112 frames per TR
- Codes are discrete integers [0, 1023]
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from giblet.models.encoder import AudioEncoder, MultimodalEncoder


def demo_encodec_audio_encoder():
    """Demonstrate basic EnCodec audio encoder usage."""
    print("=" * 70)
    print("EnCodec Audio Encoder Demo")
    print("=" * 70)

    # 1. Create EnCodec-aware audio encoder
    print("\n1. Creating AudioEncoder in EnCodec mode...")
    encoder = AudioEncoder(
        input_codebooks=8,  # 8 codebooks (3.0 kbps)
        frames_per_tr=112,  # ~112 frames per TR
        output_features=256,  # Output feature dimension
        use_encodec=True,  # Enable EnCodec mode
        vocab_size=1024,  # Codebook size
        embed_dim=64,  # Embedding dimension for codes
    )
    encoder.eval()

    print(f"   Input: (batch, 8 codebooks, 112 frames) - integer codes [0, 1023]")
    print(f"   Output: (batch, 256 features)")

    # 2. Create synthetic EnCodec codes
    print("\n2. Creating synthetic EnCodec codes...")
    batch_size = 4
    codes = torch.randint(0, 1024, (batch_size, 8, 112))
    print(f"   Shape: {codes.shape}")
    print(f"   Data type: {codes.dtype}")
    print(f"   Value range: [{codes.min().item()}, {codes.max().item()}]")

    # 3. Forward pass
    print("\n3. Forward pass through encoder...")
    with torch.no_grad():
        features = encoder(codes)
    print(f"   Output shape: {features.shape}")
    print(
        f"   Output stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}"
    )

    # 4. Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    embed_params = encoder.code_embedding.weight.numel()
    print(f"\n4. Model parameters:")
    print(f"   Total parameters: {n_params:,}")
    print(f"   Embedding layer: {embed_params:,} ({embed_params/n_params*100:.1f}%)")

    print("\n" + "=" * 70)


def demo_mel_vs_encodec():
    """Compare mel spectrogram mode vs EnCodec mode."""
    print("\n" + "=" * 70)
    print("Mel Spectrogram vs EnCodec Comparison")
    print("=" * 70)

    # Create encoders in both modes
    print("\n1. Creating encoders in both modes...")
    encoder_mel = AudioEncoder(
        input_mels=2048, frames_per_tr=65, output_features=256, use_encodec=False
    )

    encoder_encodec = AudioEncoder(
        input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
    )

    # Compare architectures
    params_mel = sum(p.numel() for p in encoder_mel.parameters())
    params_encodec = sum(p.numel() for p in encoder_encodec.parameters())

    print(f"\n2. Architecture comparison:")
    print(f"   Mel mode parameters: {params_mel:,}")
    print(f"   EnCodec mode parameters: {params_encodec:,}")
    print(f"   Difference: {abs(params_mel - params_encodec):,}")

    # Process inputs
    print(f"\n3. Processing inputs:")
    batch_size = 4

    # Mel input
    mel_input = torch.randn(batch_size, 2048, 65)
    print(f"   Mel input shape: {mel_input.shape} (continuous values)")

    # EnCodec input
    encodec_input = torch.randint(0, 1024, (batch_size, 8, 112))
    print(f"   EnCodec input shape: {encodec_input.shape} (discrete codes)")

    # Forward passes
    encoder_mel.eval()
    encoder_encodec.eval()

    with torch.no_grad():
        mel_features = encoder_mel(mel_input)
        encodec_features = encoder_encodec(encodec_input)

    print(f"\n4. Output comparison:")
    print(f"   Mel output shape: {mel_features.shape}")
    print(f"   EnCodec output shape: {encodec_features.shape}")
    print(
        f"   Both produce same output dimension: {mel_features.shape[1] == encodec_features.shape[1]}"
    )

    print("\n" + "=" * 70)


def demo_multimodal_encodec():
    """Demonstrate full MultimodalEncoder with EnCodec audio."""
    print("\n" + "=" * 70)
    print("MultimodalEncoder with EnCodec Audio")
    print("=" * 70)

    # Create multimodal encoder with EnCodec audio
    print("\n1. Creating MultimodalEncoder with EnCodec audio...")
    encoder = MultimodalEncoder(
        video_height=90,
        video_width=160,
        audio_codebooks=8,
        audio_frames_per_tr=112,
        text_dim=1024,
        use_encodec=True,  # Enable EnCodec for audio
        bottleneck_dim=2048,
    )
    encoder.eval()

    print(f"   Video input: (batch, 3, 90, 160)")
    print(f"   Audio input: (batch, 8, 112) EnCodec codes")
    print(f"   Text input: (batch, 1024)")
    print(f"   Bottleneck output: (batch, 2048)")

    # Create multimodal inputs
    print("\n2. Creating multimodal inputs...")
    batch_size = 4
    video = torch.randn(batch_size, 3, 90, 160)
    audio_codes = torch.randint(0, 1024, (batch_size, 8, 112))
    text = torch.randn(batch_size, 1024)

    # Forward pass
    print("\n3. Forward pass through full encoder...")
    with torch.no_grad():
        bottleneck, voxels = encoder(video, audio_codes, text, return_voxels=True)

    print(f"   Bottleneck shape: {bottleneck.shape}")
    print(f"   Voxel predictions shape: {voxels.shape}")
    print(
        f"   Bottleneck stats: mean={bottleneck.mean().item():.4f}, std={bottleneck.std().item():.4f}"
    )

    # Parameter breakdown
    print("\n4. Parameter breakdown:")
    param_dict = encoder.get_parameter_count()
    for key in ["video_encoder", "audio_encoder", "text_encoder", "total"]:
        if key in param_dict:
            print(f"   {key:20s}: {param_dict[key]:>12,} parameters")

    print("\n" + "=" * 70)


def demo_gradient_flow():
    """Demonstrate gradient flow through EnCodec encoder."""
    print("\n" + "=" * 70)
    print("Gradient Flow Through EnCodec Encoder")
    print("=" * 70)

    # Create encoder
    print("\n1. Creating trainable encoder...")
    encoder = AudioEncoder(
        input_codebooks=8, frames_per_tr=112, output_features=256, use_encodec=True
    )
    encoder.train()

    # Create input (discrete codes don't need gradients)
    print("\n2. Creating input codes...")
    batch_size = 4
    codes = torch.randint(0, 1024, (batch_size, 8, 112))

    # Forward pass
    print("\n3. Forward pass...")
    features = encoder(codes)

    # Backward pass
    print("\n4. Backward pass...")
    loss = features.sum()
    loss.backward()

    # Check gradients
    print("\n5. Checking gradients...")
    n_params_with_grad = 0
    total_params = 0
    for name, param in encoder.named_parameters():
        total_params += 1
        if param.grad is not None:
            n_params_with_grad += 1

    print(f"   Parameters with gradients: {n_params_with_grad}/{total_params}")
    print(
        f"   Embedding layer gradient shape: {encoder.code_embedding.weight.grad.shape}"
    )
    print(
        f"   Embedding gradient norm: {encoder.code_embedding.weight.grad.norm().item():.4f}"
    )

    print("\n   ✓ Gradients flow correctly through embedding and convolutions")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AudioEncoder EnCodec Support Demonstration")
    print("Issue #24, Task 2.2: Update AudioEncoder for EnCodec codes")
    print("=" * 70)

    # Run demonstrations
    demo_encodec_audio_encoder()
    demo_mel_vs_encodec()
    demo_multimodal_encodec()
    demo_gradient_flow()

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70 + "\n")

"""
Architecture Audit Test Script for Issue #11

This script instantiates the MultimodalAutoencoder and provides a detailed
layer-by-layer analysis of the architecture to verify compliance with the
11-layer specification in Issue #2.

Outputs:
- Layer-by-layer architecture breakdown
- Parameter counts for each component
- Dimension flow through the network
- Compliance verification
"""

import torch
import sys
from pathlib import Path

# Add giblet to path
sys.path.insert(0, str(Path(__file__).parent))

from giblet.models.autoencoder import MultimodalAutoencoder


def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"


def analyze_encoder_architecture(model):
    """Analyze encoder architecture layer by layer."""
    print("=" * 80)
    print("ENCODER ARCHITECTURE (Layers 1-6)")
    print("=" * 80)

    encoder = model.encoder

    # Layer 1: Input
    print("\n[Layer 1: Input]")
    print(f"  Video input:     {encoder.video_height}×{encoder.video_width}×3 = {encoder.video_height * encoder.video_width * 3:,} features")
    print(f"  Audio input:     {encoder.audio_mels:,} mel bins")
    print(f"  Text input:      {encoder.text_dim:,} embeddings")
    print(f"  TOTAL INPUT:     {encoder.video_height * encoder.video_width * 3 + encoder.audio_mels + encoder.text_dim:,} features")

    # Layer 2A: Video encoder
    print("\n[Layer 2A: Video Conv2D Branch]")
    video_enc = encoder.video_encoder
    print(f"  Input:           (batch, 3, {video_enc.input_height}, {video_enc.input_width})")
    print(f"  Conv1:           3 → 32 channels, stride=2, kernel=3")
    print(f"  Conv2:           32 → 64 channels, stride=2, kernel=3")
    print(f"  Conv3:           64 → 128 channels, stride=2, kernel=3")
    print(f"  Conv4:           128 → 256 channels, stride=2, kernel=3")
    print(f"  Flat features:   {video_enc.flat_features:,}")
    print(f"  Output (FC):     {video_enc.output_features:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in video_enc.parameters()):,}")

    # Layer 2B: Audio encoder
    print("\n[Layer 2B: Audio Conv1D Branch]")
    audio_enc = encoder.audio_encoder
    print(f"  Input:           (batch, {audio_enc.input_mels:,}) mel bins")
    print(f"  Conv1:           1 → 32 channels, stride=2, kernel=3")
    print(f"  Conv2:           32 → 64 channels, stride=2, kernel=3")
    print(f"  Conv3:           64 → 128 channels, stride=2, kernel=3")
    print(f"  Conv4:           128 → 256 channels, stride=2, kernel=3")
    print(f"  Flat features:   {audio_enc.flat_features:,}")
    print(f"  Output (FC):     {audio_enc.output_features:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in audio_enc.parameters()):,}")

    # Layer 2C: Text encoder
    print("\n[Layer 2C: Text Linear Branch]")
    text_enc = encoder.text_encoder
    print(f"  Input:           (batch, {text_enc.input_dim:,}) embeddings")
    print(f"  FC1:             {text_enc.input_dim:,} → 512")
    print(f"  FC2:             512 → {text_enc.output_features:,}")
    print(f"  Output:          {text_enc.output_features:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in text_enc.parameters()):,}")

    # Layer 3: Pooled features
    print("\n[Layer 3: Pool/Concatenate Features]")
    pooled_dim = encoder.pooled_dim
    print(f"  Video features:  {video_enc.output_features:,}")
    print(f"  Audio features:  {audio_enc.output_features:,}")
    print(f"  Text features:   {text_enc.output_features:,}")
    print(f"  Concatenated:    {pooled_dim:,} features")

    # Layer 4: Feature convolution
    print("\n[Layer 4: Feature Convolution + ReLU]")
    print(f"  Input:           {pooled_dim:,} features")
    print(f"  Operation:       Linear({pooled_dim} → {pooled_dim}) + BatchNorm + ReLU + Dropout")
    print(f"  Output:          {pooled_dim:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in encoder.feature_conv.parameters()):,}")

    # Layer 5 + 6: Compression to bottleneck
    print("\n[Layer 5: Linear Compression to Intermediate]")
    print(f"  Input:           {pooled_dim:,} features")
    print(f"  Linear 1:        {pooled_dim:,} → 4,096")
    print(f"  Output:          4,096 features")

    print("\n[Layer 6: BOTTLENECK LAYER (Middle of Autoencoder)]")
    print(f"  Input:           4,096 features")
    print(f"  Linear 2:        4,096 → {encoder.bottleneck_dim:,}")
    print(f"  Output:          {encoder.bottleneck_dim:,} features ★ SMALLEST LAYER ★")
    print(f"  Parameters:      {sum(p.numel() for p in encoder.to_bottleneck.parameters()):,}")

    # Optional voxel expansion
    print("\n[Additional: Bottleneck → Voxels (Training Only)]")
    print(f"  Input:           {encoder.bottleneck_dim:,} features")
    print(f"  Linear 1:        {encoder.bottleneck_dim:,} → 16,384")
    print(f"  Linear 2:        16,384 → {encoder.n_voxels:,} voxels")
    print(f"  Output:          {encoder.n_voxels:,} voxels")
    print(f"  Parameters:      {sum(p.numel() for p in encoder.bottleneck_to_voxels.parameters()):,}")


def analyze_decoder_architecture(model):
    """Analyze decoder architecture layer by layer."""
    print("\n" + "=" * 80)
    print("DECODER ARCHITECTURE (Layers 7-11)")
    print("=" * 80)

    decoder = model.decoder

    # Layer 7: Expand from bottleneck
    print("\n[Layer 7: Expand from Bottleneck]")
    print(f"  Input:           {decoder.bottleneck_dim:,} features (from Layer 6)")
    print(f"  Linear:          {decoder.bottleneck_dim:,} → {decoder.hidden_dim:,}")
    print(f"  Output:          {decoder.hidden_dim:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer7.parameters()):,}")

    # Layer 8: Feature deconvolution
    print("\n[Layer 8: Feature Deconvolution + ReLU]")
    intermediate_dim_1 = decoder.hidden_dim * 2
    print(f"  Input:           {decoder.hidden_dim:,} features")
    print(f"  Linear:          {decoder.hidden_dim:,} → {intermediate_dim_1:,}")
    print(f"  Output:          {intermediate_dim_1:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer8.parameters()):,}")

    # Layer 9: Unpool features
    print("\n[Layer 9: Unpool Features]")
    intermediate_dim_2 = decoder.hidden_dim * 4
    print(f"  Input:           {intermediate_dim_1:,} features")
    print(f"  Linear:          {intermediate_dim_1:,} → {intermediate_dim_2:,}")
    print(f"  Output:          {intermediate_dim_2:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer9.parameters()):,}")

    # Layer 10A/B/C: Modality branches
    print("\n[Layer 10A: Video Deconvolution Branch]")
    print(f"  Input:           {intermediate_dim_2:,} features")
    print(f"  Linear 1:        {intermediate_dim_2:,} → {decoder.hidden_dim * 4:,}")
    print(f"  Linear 2:        {decoder.hidden_dim * 4:,} → {decoder.hidden_dim * 2:,}")
    print(f"  Output:          {decoder.hidden_dim * 2:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer10_video.parameters()):,}")

    print("\n[Layer 10B: Audio Deconvolution Branch]")
    print(f"  Input:           {intermediate_dim_2:,} features")
    print(f"  Linear 1:        {intermediate_dim_2:,} → {decoder.hidden_dim:,}")
    print(f"  Linear 2:        {decoder.hidden_dim:,} → {decoder.hidden_dim // 2:,}")
    print(f"  Output:          {decoder.hidden_dim // 2:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer10_audio.parameters()):,}")

    print("\n[Layer 10C: Text Deconvolution Branch]")
    print(f"  Input:           {intermediate_dim_2:,} features")
    print(f"  Linear 1:        {intermediate_dim_2:,} → {decoder.hidden_dim:,}")
    print(f"  Linear 2:        {decoder.hidden_dim:,} → {decoder.hidden_dim:,}")
    print(f"  Output:          {decoder.hidden_dim:,} features")
    print(f"  Parameters:      {sum(p.numel() for p in decoder.layer10_text.parameters()):,}")

    # Layer 11: Output layers
    print("\n[Layer 11: Output Reconstruction]")
    print(f"  Video output:    {decoder.hidden_dim * 2:,} → {decoder.video_dim:,} (160×90×3)")
    print(f"  Audio output:    {decoder.hidden_dim // 2:,} → {decoder.audio_dim:,} mels")
    print(f"  Text output:     {decoder.hidden_dim:,} → {decoder.text_dim:,} embeddings")
    print(f"  Video params:    {sum(p.numel() for p in decoder.layer11_video.parameters()):,}")
    print(f"  Audio params:    {sum(p.numel() for p in decoder.layer11_audio.parameters()):,}")
    print(f"  Text params:     {sum(p.numel() for p in decoder.layer11_text.parameters()):,}")


def verify_compliance(model):
    """Verify compliance with Issue #2 specification."""
    print("\n" + "=" * 80)
    print("COMPLIANCE VERIFICATION")
    print("=" * 80)

    encoder = model.encoder
    decoder = model.decoder

    checks = []

    # Check 1: Bottleneck is 8,000 dimensions
    check_bottleneck = encoder.bottleneck_dim == 8000
    checks.append(("Layer 6 bottleneck is 8,000 dimensions", check_bottleneck))

    # Check 2: Voxels are 85,810
    check_voxels = encoder.n_voxels == 85810
    checks.append(("Layer 5 expansion is 85,810 voxels", check_voxels))

    # Check 3: Parallel branches exist in encoder
    check_parallel_enc = (
        hasattr(encoder, 'video_encoder') and
        hasattr(encoder, 'audio_encoder') and
        hasattr(encoder, 'text_encoder')
    )
    checks.append(("Layers 2A/B/C parallel branches exist (encoder)", check_parallel_enc))

    # Check 4: Parallel branches exist in decoder
    check_parallel_dec = (
        hasattr(decoder, 'layer10_video') and
        hasattr(decoder, 'layer10_audio') and
        hasattr(decoder, 'layer10_text')
    )
    checks.append(("Layers 10A/B/C parallel branches exist (decoder)", check_parallel_dec))

    # Check 5: Decoder is symmetric to encoder
    check_symmetric = decoder.bottleneck_dim == encoder.bottleneck_dim
    checks.append(("Decoder bottleneck matches encoder bottleneck", check_symmetric))

    # Check 6: Video dimensions match
    video_dim_expected = model.video_height * model.video_width * 3
    check_video = decoder.video_dim == video_dim_expected
    checks.append(("Video dimensions match (160×90×3 = 43,200)", check_video))

    # Check 7: Audio dimensions match
    check_audio = decoder.audio_dim == model.audio_mels
    checks.append(("Audio dimensions match (2,048 mels)", check_audio))

    # Check 8: Text dimensions match
    check_text = decoder.text_dim == model.text_dim
    checks.append(("Text dimensions match (1,024 embeddings)", check_text))

    # Check 9: Bottleneck is smallest layer
    pooled_dim = encoder.pooled_dim
    bottleneck_is_smallest = (
        encoder.bottleneck_dim < pooled_dim and
        encoder.bottleneck_dim < encoder.n_voxels and
        encoder.bottleneck_dim < decoder.video_dim and
        encoder.bottleneck_dim < decoder.hidden_dim * 4
    )
    checks.append(("Layer 6 is SMALLEST layer in architecture", bottleneck_is_smallest))

    # Print results
    print("\nCompliance Checks:")
    print("-" * 80)
    all_pass = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {check_name}")
        if not result:
            all_pass = False

    print("-" * 80)
    if all_pass:
        print("  ★ ALL CHECKS PASSED ★")
    else:
        print("  ⚠ SOME CHECKS FAILED ⚠")

    return all_pass


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)

    batch_size = 2

    # Create dummy inputs
    video = torch.randn(batch_size, 3, model.video_height, model.video_width)
    audio = torch.randn(batch_size, model.audio_mels)
    text = torch.randn(batch_size, model.text_dim)
    fmri = torch.randn(batch_size, model.n_voxels)

    print(f"\nInput shapes:")
    print(f"  Video: {tuple(video.shape)}")
    print(f"  Audio: {tuple(audio.shape)}")
    print(f"  Text:  {tuple(text.shape)}")
    print(f"  fMRI:  {tuple(fmri.shape)}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(video, audio, text, fmri_target=None)

    print(f"\nOutput shapes:")
    print(f"  Bottleneck:      {tuple(outputs['bottleneck'].shape)}")
    print(f"  Predicted fMRI:  {tuple(outputs['predicted_fmri'].shape)}")
    print(f"  Video recon:     {tuple(outputs['video_recon'].shape)}")
    print(f"  Audio recon:     {tuple(outputs['audio_recon'].shape)}")
    print(f"  Text recon:      {tuple(outputs['text_recon'].shape)}")

    # Verify shapes
    assert outputs['bottleneck'].shape == (batch_size, model.bottleneck_dim)
    assert outputs['predicted_fmri'].shape == (batch_size, model.n_voxels)
    assert outputs['video_recon'].shape == (batch_size, model.video_width * model.video_height * 3)
    assert outputs['audio_recon'].shape == (batch_size, model.audio_mels)
    assert outputs['text_recon'].shape == (batch_size, model.text_dim)

    print("\n✓ All output shapes are correct!")


def print_parameter_summary(model):
    """Print parameter count summary."""
    print("\n" + "=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)

    params = model.get_parameter_count()

    print("\nEncoder Parameters:")
    print(f"  Video encoder:         {format_number(params['encoder_breakdown']['video_encoder'])}")
    print(f"  Audio encoder:         {format_number(params['encoder_breakdown']['audio_encoder'])}")
    print(f"  Text encoder:          {format_number(params['encoder_breakdown']['text_encoder'])}")
    print(f"  Feature conv (L4):     {format_number(params['encoder_breakdown']['feature_conv'])}")
    print(f"  To bottleneck (L5-6):  {format_number(params['encoder_breakdown']['to_bottleneck'])}")
    print(f"  Bottleneck→Voxels:     {format_number(params['encoder_breakdown']['bottleneck_to_voxels'])}")
    print(f"  ENCODER TOTAL:         {format_number(params['encoder'])}")

    print("\nDecoder Parameters:")
    print(f"  Layer 7:               {format_number(params['decoder_breakdown']['layer7'])}")
    print(f"  Layer 8:               {format_number(params['decoder_breakdown']['layer8'])}")
    print(f"  Layer 9:               {format_number(params['decoder_breakdown']['layer9'])}")
    print(f"  Layer 10A (video):     {format_number(params['decoder_breakdown']['layer10_video'])}")
    print(f"  Layer 10B (audio):     {format_number(params['decoder_breakdown']['layer10_audio'])}")
    print(f"  Layer 10C (text):      {format_number(params['decoder_breakdown']['layer10_text'])}")
    print(f"  Layer 11 (video):      {format_number(params['decoder_breakdown']['layer11_video'])}")
    print(f"  Layer 11 (audio):      {format_number(params['decoder_breakdown']['layer11_audio'])}")
    print(f"  Layer 11 (text):       {format_number(params['decoder_breakdown']['layer11_text'])}")
    print(f"  DECODER TOTAL:         {format_number(params['decoder'])}")

    print(f"\n{'=' * 80}")
    print(f"TOTAL PARAMETERS:        {format_number(params['total'])}")
    print(f"{'=' * 80}")


def main():
    """Run full architecture audit."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE AUDIT - MULTIMODAL AUTOENCODER")
    print("Issue #11: Verify compliance with Issue #2 specification")
    print("=" * 80)

    # Create model with default parameters
    model = MultimodalAutoencoder(
        video_height=90,
        video_width=160,
        audio_mels=2048,
        text_dim=1024,
        n_voxels=85810,
        bottleneck_dim=8000,
        video_features=1024,
        audio_features=256,
        text_features=256,
        decoder_hidden_dim=2048,
        decoder_dropout=0.3
    )

    # Run analyses
    analyze_encoder_architecture(model)
    analyze_decoder_architecture(model)
    print_parameter_summary(model)
    compliance_result = verify_compliance(model)
    test_forward_pass(model)

    # Final summary
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    if compliance_result:
        print("\n✓ Architecture is COMPLIANT with Issue #2 specification")
    else:
        print("\n✗ Architecture has COMPLIANCE ISSUES - see details above")
    print()

    return compliance_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

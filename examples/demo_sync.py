"""
Demonstration of the temporal synchronization module.

Shows how to:
1. Create sample data matching example dataset dimensions
2. Align all modalities to a common TR grid
3. Apply HRF convolution to stimulus features
4. Verify output dimensions and structure
"""

import numpy as np
from giblet.alignment.sync import align_all_modalities, get_alignment_info


def main():
    """Run synchronization demo."""
    print("=" * 70)
    print("TEMPORAL SYNCHRONIZATION MODULE DEMONSTRATION")
    print("=" * 70)

    # Create sample data with example dataset dimensions (Sherlock)
    print("\n1. Creating sample data with example dimensions...")
    print("-" * 70)

    np.random.seed(42)

    # Video: 950 TRs, downsampled RGB frames (160x90x3 = 43200 features)
    video_features = np.random.randn(950, 43200).astype(np.float32)
    print(f"   Video features:  {video_features.shape}")
    print(f"      - 950 TRs at 25fps (example duration)")
    print(f"      - Downsampled RGB frames (160x90x3)")

    # Audio: 946 TRs, mel spectrogram (128 mels)
    audio_features = np.random.randn(946, 128).astype(np.float32)
    print(f"   Audio features:  {audio_features.shape}")
    print(f"      - 946 TRs from librosa mel spectrogram")
    print(f"      - 128 mel frequency bins")

    # Text: 950 TRs, embeddings (1024-dim)
    text_features = np.random.randn(950, 1024).astype(np.float32)
    print(f"   Text features:   {text_features.shape}")
    print(f"      - 950 TRs from annotations")
    print(f"      - BAAI/bge-large-en-v1.5 embeddings (1024-dim)")

    # fMRI: 920 TRs (reference), voxel timeseries
    fmri_features = np.random.randn(920, 85810).astype(np.float32)
    print(f"   fMRI features:   {fmri_features.shape}")
    print(f"      - 920 TRs (minimum, reference)")
    print(f"      - Brain voxels extracted via mask (~85,810 voxels)")

    # Align all modalities WITHOUT HRF convolution
    print("\n2. Aligning all modalities (without HRF)...")
    print("-" * 70)

    result_no_hrf = align_all_modalities(
        video_features=video_features,
        audio_features=audio_features,
        text_features=text_features,
        fmri_features=fmri_features,
        apply_hrf_conv=False,
        tr=1.5
    )

    print("\n   Alignment result (no HRF):")
    print(f"   - Video:  {result_no_hrf['video'].shape}")
    print(f"   - Audio:  {result_no_hrf['audio'].shape}")
    print(f"   - Text:   {result_no_hrf['text'].shape}")
    print(f"   - fMRI:   {result_no_hrf['fmri'].shape}")
    print(f"   - Target: {result_no_hrf['n_trs']} TRs")

    # Verify all outputs have 920 TRs
    assert result_no_hrf['video'].shape[0] == 920, "Video TRs mismatch"
    assert result_no_hrf['audio'].shape[0] == 920, "Audio TRs mismatch"
    assert result_no_hrf['text'].shape[0] == 920, "Text TRs mismatch"
    assert result_no_hrf['fmri'].shape[0] == 920, "fMRI TRs mismatch"
    print("\n   ✓ All modalities aligned to 920 TRs")

    # Verify feature dimensions are preserved
    assert result_no_hrf['video'].shape[1] == 43200, "Video features changed"
    assert result_no_hrf['audio'].shape[1] == 128, "Audio features changed"
    assert result_no_hrf['text'].shape[1] == 1024, "Text features changed"
    assert result_no_hrf['fmri'].shape[1] == 85810, "fMRI features changed"
    print("   ✓ Feature dimensions preserved")

    # Align all modalities WITH HRF convolution
    print("\n3. Aligning all modalities (with HRF convolution)...")
    print("-" * 70)

    result_with_hrf = align_all_modalities(
        video_features=video_features,
        audio_features=audio_features,
        text_features=text_features,
        fmri_features=fmri_features,
        apply_hrf_conv=True,
        tr=1.5
    )

    print("\n   Alignment result (with HRF):")
    print(f"   - Video:  {result_with_hrf['video'].shape}")
    print(f"   - Audio:  {result_with_hrf['audio'].shape}")
    print(f"   - Text:   {result_with_hrf['text'].shape}")
    print(f"   - fMRI:   {result_with_hrf['fmri'].shape}")
    print(f"   - Target: {result_with_hrf['n_trs']} TRs")

    # Verify all outputs have 920 TRs
    assert result_with_hrf['video'].shape[0] == 920, "Video TRs mismatch (HRF)"
    assert result_with_hrf['audio'].shape[0] == 920, "Audio TRs mismatch (HRF)"
    assert result_with_hrf['text'].shape[0] == 920, "Text TRs mismatch (HRF)"
    assert result_with_hrf['fmri'].shape[0] == 920, "fMRI TRs mismatch (HRF)"
    print("\n   ✓ All modalities aligned to 920 TRs")

    # Verify feature dimensions are preserved
    assert result_with_hrf['video'].shape[1] == 43200, "Video features changed (HRF)"
    assert result_with_hrf['audio'].shape[1] == 128, "Audio features changed (HRF)"
    assert result_with_hrf['text'].shape[1] == 1024, "Text features changed (HRF)"
    assert result_with_hrf['fmri'].shape[1] == 85810, "fMRI features changed (HRF)"
    print("   ✓ Feature dimensions preserved")

    # Verify HRF changed stimulus features
    assert not np.allclose(result_no_hrf['video'], result_with_hrf['video']), \
        "HRF didn't change video"
    assert not np.allclose(result_no_hrf['audio'], result_with_hrf['audio']), \
        "HRF didn't change audio"
    assert not np.allclose(result_no_hrf['text'], result_with_hrf['text']), \
        "HRF didn't change text"
    print("   ✓ HRF convolution modified stimulus features")

    # Verify fMRI is identical (not convolved)
    np.testing.assert_array_equal(result_no_hrf['fmri'], result_with_hrf['fmri'])
    print("   ✓ fMRI unchanged (as expected)")

    # Get alignment info
    print("\n4. Alignment information...")
    print("-" * 70)

    info = get_alignment_info(result_with_hrf)

    print(f"\n   Aligned dimensions:")
    print(f"   - Number of TRs:    {info['n_trs']}")
    print(f"   - Duration:         {info['n_trs'] * 1.5 / 60:.2f} minutes")

    print(f"\n   Feature dimensions:")
    print(f"   - Video features:   {info['video_features']}")
    print(f"   - Audio features:   {info['audio_features']}")
    print(f"   - Text features:    {info['text_features']}")
    print(f"   - fMRI features:    {info['fmri_features']}")

    print(f"\n   Original TR counts:")
    print(f"   - Video:  {info['video_orig_trs']} TRs")
    print(f"   - Audio:  {info['audio_orig_trs']} TRs")
    print(f"   - Text:   {info['text_orig_trs']} TRs")
    print(f"   - fMRI:   {info['fmri_orig_trs']} TRs")

    # Verify all values are finite
    print("\n5. Data quality checks...")
    print("-" * 70)

    for modality in ['video', 'audio', 'text', 'fmri']:
        data = result_with_hrf[modality]
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        assert not has_nan, f"{modality} contains NaN"
        assert not has_inf, f"{modality} contains Inf"
        print(f"   ✓ {modality}: all finite values")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE - ALL CHECKS PASSED!")
    print("=" * 70)

    print("\nKey results:")
    print(f"  - Video, audio, text, and fMRI are all aligned to {result_with_hrf['n_trs']} TRs")
    print(f"  - Duration: {result_with_hrf['n_trs'] * 1.5:.0f} seconds (~{result_with_hrf['n_trs'] * 1.5 / 60:.1f} minutes)")
    print(f"  - HRF convolution applied to stimulus features")
    print(f"  - All output shapes: (920, n_features)")
    print(f"\nUsage:")
    print(f"  from giblet.alignment.sync import align_all_modalities")
    print(f"  result = align_all_modalities(video, audio, text, fmri, apply_hrf_conv=True)")


if __name__ == '__main__':
    main()

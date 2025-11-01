"""
Temporal synchronization module for multimodal alignment.

This module aligns all stimulus modalities (video, audio, text) to a common
temporal grid based on fMRI TRs. It also applies HRF convolution to stimulus
features to predict BOLD responses.

All data is aligned to the minimum number of TRs across modalities.
(Example: 920 TRs at TR=1.5s ≈ 23 minutes total duration).

References:
    The approach follows standard fMRI analysis practices:
    - Align all inputs to common TR grid
    - Convolve stimulus features with HRF to predict BOLD
    - Truncate to minimum duration across modalities
"""

import numpy as np
from typing import Optional, Dict, Union
from .hrf import apply_hrf, convolve_with_padding


def _resample_features(features: np.ndarray, current_trs: int, target_trs: int) -> np.ndarray:
    """
    Resample feature matrix from current_trs to target_trs.

    Uses linear interpolation for continuous features (float dtype).
    Uses nearest-neighbor interpolation for discrete codes (int dtype).
    Handles both 2D (n_trs, n_features) and 3D (n_trs, n_mels, frames_per_tr) arrays.

    Parameters
    ----------
    features : np.ndarray
        Input features with shape (current_trs, n_features) or (current_trs, n_mels, frames_per_tr)
        For EnCodec codes: dtype should be int64
        For mel spectrograms: dtype should be float32/float64
    current_trs : int
        Current number of TRs
    target_trs : int
        Target number of TRs

    Returns
    -------
    resampled : np.ndarray
        Resampled features with shape (target_trs, n_features) or (target_trs, n_mels, frames_per_tr)

    Notes
    -----
    Discrete Code Handling (EnCodec):
    - Integer dtypes (int32, int64) are detected automatically
    - Uses nearest-neighbor interpolation (no averaging/blending)
    - Preserves code validity (codes remain in original codebook range)

    Continuous Feature Handling (Mel, Text, Video):
    - Float dtypes use linear interpolation
    - Allows smooth temporal transitions
    """
    if current_trs == target_trs:
        return features.copy()

    # Detect if features are discrete codes (integer dtype)
    is_discrete = features.dtype in [np.int32, np.int64]

    # Handle 3D audio features (n_trs, n_mels/n_codebooks, frames_per_tr)
    if features.ndim == 3:
        n_trs, n_mels, frames_per_tr = features.shape

        # Create time indices for current and target grids
        current_indices = np.arange(current_trs)
        target_indices = np.linspace(0, current_trs - 1, target_trs)

        if is_discrete:
            # EnCodec codes: Use nearest-neighbor interpolation
            # Round target indices to nearest integer for code lookup
            target_indices_int = np.round(target_indices).astype(int)
            target_indices_int = np.clip(target_indices_int, 0, current_trs - 1)

            # Direct indexing (no interpolation)
            resampled = features[target_indices_int]  # Shape: (target_trs, n_codebooks, frames_per_tr)

        else:
            # Continuous features: Use linear interpolation
            resampled = np.zeros((target_trs, n_mels, frames_per_tr), dtype=features.dtype)

            for mel_idx in range(n_mels):
                for frame_idx in range(frames_per_tr):
                    resampled[:, mel_idx, frame_idx] = np.interp(
                        target_indices,
                        current_indices,
                        features[:, mel_idx, frame_idx]
                    )

        return resampled

    # Handle 2D features (n_trs, n_features)
    # Create time indices for current and target grids
    # Current grid: 0, 1, 2, ..., current_trs-1
    # Target grid: 0, 1, 2, ..., target_trs-1
    # Map target to current via linear scaling
    current_indices = np.arange(current_trs)
    target_indices = np.linspace(0, current_trs - 1, target_trs)

    if is_discrete:
        # Discrete codes (rare for 2D, but handle it)
        target_indices_int = np.round(target_indices).astype(int)
        target_indices_int = np.clip(target_indices_int, 0, current_trs - 1)
        resampled = features[target_indices_int]

    else:
        # Continuous features: Interpolate each feature dimension
        n_features = features.shape[1]
        resampled = np.zeros((target_trs, n_features), dtype=features.dtype)

        for feat_idx in range(n_features):
            resampled[:, feat_idx] = np.interp(
                target_indices,
                current_indices,
                features[:, feat_idx]
            )

    return resampled


def align_all_modalities(
    video_features: np.ndarray,
    audio_features: np.ndarray,
    text_features: np.ndarray,
    fmri_features: np.ndarray,
    apply_hrf_conv: bool = True,
    tr: float = 1.5,
    hrf_padding_duration: float = 10.0
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Align all stimulus modalities to a common TR grid and apply HRF convolution.

    This function:
    1. Determines the target number of TRs (minimum across all modalities)
    2. Resamples all stimulus features to the target TR count
    3. Optionally applies HRF convolution to stimulus features
    4. Returns aligned arrays and metadata

    Parameters
    ----------
    video_features : np.ndarray
        Video features with shape (n_video_trs, n_video_features).
        Typically (950, 43200) for downsampled RGB frames.
    audio_features : np.ndarray
        Audio mel spectrogram features with shape (n_audio_trs, n_audio_features).
        Typically (946, 128) for 128-bin mel spectrograms.
    text_features : np.ndarray
        Text embeddings with shape (n_text_trs, n_text_features).
        Typically (950, 1024) for BAAI/bge-large-en-v1.5 embeddings.
    fmri_features : np.ndarray
        fMRI voxel timeseries with shape (n_fmri_trs, n_fmri_features).
        Typically (920, 85810) for brain voxels extracted via mask.
    apply_hrf_conv : bool, default=True
        Whether to convolve stimulus features with canonical HRF.
        If True, stimulus features are convolved to predict BOLD response.
        If False, features are used as-is.
    tr : float, default=1.5
        Repetition time in seconds (must match fMRI TR).
    hrf_padding_duration : float, default=10.0
        Duration of zero-padding (seconds) for HRF convolution.
        Used to minimize edge effects during convolution.
        Should be at least as long as HRF duration (~20s).

    Returns
    -------
    alignment_dict : dict
        Dictionary with keys:
        - 'video': Aligned video features, shape (target_trs, n_video_features)
        - 'audio': Aligned audio features, shape (target_trs, n_audio_features)
        - 'text': Aligned text features, shape (target_trs, n_text_features)
        - 'fmri': fMRI features (truncated), shape (target_trs, n_fmri_features)
        - 'n_trs': int, the target number of TRs used for alignment
        - 'video_orig_trs': Original number of video TRs
        - 'audio_orig_trs': Original number of audio TRs
        - 'text_orig_trs': Original number of text TRs
        - 'fmri_orig_trs': Original number of fMRI TRs

    Notes
    -----
    Alignment Strategy:
    - All stimulus modalities are resampled to match the minimum TR count
    - This is typically the fMRI TRs (example: 920 TRs), which is the
      minimum number of samples across modalities
    - Resampling uses linear interpolation to maintain feature continuity

    HRF Convolution:
    - Only applied to stimulus features (video, audio, text), NOT fMRI
    - Uses the Glover HRF model from nilearn
    - Convolution is performed with padding to minimize edge effects
    - The convolved features can be used to predict expected BOLD responses

    Examples
    --------
    >>> # Align all modalities
    >>> result = align_all_modalities(
    ...     video_features=video_data,      # (950, 43200)
    ...     audio_features=audio_data,      # (946, 128)
    ...     text_features=text_data,        # (950, 1024)
    ...     fmri_features=fmri_data,        # (920, 85810)
    ...     apply_hrf_conv=True,
    ...     tr=1.5
    ... )
    >>> result['fmri'].shape
    (920, 85810)
    >>> result['video'].shape
    (920, 43200)
    >>> result['n_trs']
    920

    >>> # Access original counts
    >>> result['video_orig_trs']
    950
    >>> result['fmri_orig_trs']
    920
    """
    # Get original TR counts
    n_video = video_features.shape[0]
    n_audio = audio_features.shape[0]
    n_text = text_features.shape[0]
    n_fmri = fmri_features.shape[0]

    # Determine target TRs (minimum across all modalities)
    # This ensures all data aligns to the shortest duration
    target_trs = min(n_video, n_audio, n_text, n_fmri)

    print(f"Aligning modalities to common TR grid:")
    print(f"  Video:  {n_video} TRs -> {target_trs} TRs")
    print(f"  Audio:  {n_audio} TRs -> {target_trs} TRs")
    print(f"  Text:   {n_text} TRs -> {target_trs} TRs")
    print(f"  fMRI:   {n_fmri} TRs (reference, already {target_trs})")
    print(f"  Target: {target_trs} TRs at TR={tr}s (~{target_trs * tr / 60:.1f} minutes)")

    # Resample stimulus modalities to target TRs
    video_aligned = _resample_features(video_features, n_video, target_trs)
    audio_aligned = _resample_features(audio_features, n_audio, target_trs)
    text_aligned = _resample_features(text_features, n_text, target_trs)

    # Truncate fMRI to target TRs
    fmri_aligned = fmri_features[:target_trs].copy()

    # Apply HRF convolution to stimulus features if requested
    if apply_hrf_conv:
        print("Applying HRF convolution to stimulus features...")

        # Convolve each stimulus modality with HRF
        # Use padding to minimize edge effects
        video_aligned = convolve_with_padding(
            video_aligned,
            tr=tr,
            padding_duration=hrf_padding_duration
        )
        audio_aligned = convolve_with_padding(
            audio_aligned,
            tr=tr,
            padding_duration=hrf_padding_duration
        )
        text_aligned = convolve_with_padding(
            text_aligned,
            tr=tr,
            padding_duration=hrf_padding_duration
        )

        print("  HRF convolution complete")

    # Build result dictionary
    result = {
        'video': video_aligned,
        'audio': audio_aligned,
        'text': text_aligned,
        'fmri': fmri_aligned,
        'n_trs': target_trs,
        'video_orig_trs': n_video,
        'audio_orig_trs': n_audio,
        'text_orig_trs': n_text,
        'fmri_orig_trs': n_fmri,
    }

    # Print alignment summary
    print(f"\nAlignment complete:")
    print(f"  All outputs have shape (920, n_features)")
    print(f"  Video:  ({target_trs}, {video_aligned.shape[1]})")
    print(f"  Audio:  ({target_trs}, {audio_aligned.shape[1]})")
    print(f"  Text:   ({target_trs}, {text_aligned.shape[1]})")
    print(f"  fMRI:   ({target_trs}, {fmri_aligned.shape[1]})")

    return result


def get_alignment_info(alignment_result: Dict) -> Dict:
    """
    Get summary information about an alignment result.

    Parameters
    ----------
    alignment_result : dict
        Result dictionary from align_all_modalities()

    Returns
    -------
    info : dict
        Dictionary with alignment statistics
    """
    return {
        'n_trs': alignment_result['n_trs'],
        'video_trs_aligned': alignment_result['video'].shape[0],
        'audio_trs_aligned': alignment_result['audio'].shape[0],
        'text_trs_aligned': alignment_result['text'].shape[0],
        'fmri_trs_aligned': alignment_result['fmri'].shape[0],
        'video_features': alignment_result['video'].shape[1],
        'audio_features': alignment_result['audio'].shape[1],
        'text_features': alignment_result['text'].shape[1],
        'fmri_features': alignment_result['fmri'].shape[1],
        'video_orig_trs': alignment_result['video_orig_trs'],
        'audio_orig_trs': alignment_result['audio_orig_trs'],
        'text_orig_trs': alignment_result['text_orig_trs'],
        'fmri_orig_trs': alignment_result['fmri_orig_trs'],
    }

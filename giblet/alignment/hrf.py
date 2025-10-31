"""
HRF (Hemodynamic Response Function) convolution module.

This module provides utilities for convolving stimulus features with the
canonical HRF to predict BOLD responses. Uses nilearn's Glover HRF model.

References:
    Glover, G. H. (1999). Deconvolution of impulse response in event-related
    BOLD fMRI. NeuroImage, 9(4), 416-429.
"""

import numpy as np
from scipy import signal
from nilearn.glm.first_level import glover_hrf


def get_canonical_hrf(tr=1.5, duration=32.0):
    """
    Get the canonical HRF kernel.

    Uses the Glover HRF model from nilearn, which provides a well-validated
    hemodynamic response function for fMRI studies.

    Parameters
    ----------
    tr : float
        Repetition time in seconds. Default is 1.5 seconds (common for Algonauts).
    duration : float
        Duration of HRF kernel in seconds. Default is 32 seconds (nilearn default).
        This captures the main peak (~5-6s) and return to baseline (~30s).

    Returns
    -------
    hrf : numpy.ndarray
        HRF kernel with shape (n_samples,) where n_samples = ceil(duration / tr).
        The HRF is normalized to have a peak of 1.0.

    Notes
    -----
    The Glover HRF has the following properties:
    - Peak at approximately 5-6 seconds post-stimulus
    - Returns to baseline around 15-30 seconds
    - Often shows a post-stimulus undershoot (dip below baseline)
    - Parameterized by the Glover (1999) model

    Example
    -------
    >>> hrf = get_canonical_hrf(tr=1.5)
    >>> hrf.shape  # 32 seconds / 1.5 seconds = ~21 samples
    (21,)
    >>> peak_idx = np.argmax(hrf)
    >>> peak_idx * 1.5  # Time to peak in seconds
    6.0
    """
    # Generate HRF using nilearn's Glover model at the specified TR
    # Use t_r parameter (newer nilearn API, tr is deprecated)
    # Set oversampling=1 to get HRF sampled exactly at TR (not upsampled)
    hrf = glover_hrf(t_r=tr, oversampling=1, time_length=duration)

    # Normalize to peak of 1.0 for interpretability
    hrf = hrf / np.max(np.abs(hrf))

    return hrf


def apply_hrf(features, tr=1.5, mode='same'):
    """
    Convolve stimulus features with canonical HRF.

    Applies the Glover HRF to features (e.g., stimulus intensity, visual features)
    to predict the resulting BOLD response. This accounts for the temporal blurring
    introduced by neurovascular coupling.

    UPDATED: Now handles 3D audio features (n_trs, n_mels, frames_per_tr).

    Parameters
    ----------
    features : numpy.ndarray
        Input stimulus features with shape:
        - (n_timepoints,) for 1D
        - (n_timepoints, n_features) for 2D
        - (n_timepoints, n_mels, frames_per_tr) for 3D audio
        Should be sampled at the same TR as the HRF.
    tr : float
        Repetition time in seconds. Must match the sampling rate of features.
        Default is 1.5 seconds.
    mode : str
        Convolution mode for handling edge effects:
        - 'same': Output has same size as input. Useful for predictions.
          Edges will show edge effects from truncation.
        - 'full': Output size is (n_timepoints + n_hrf - 1). Complete convolution.

    Returns
    -------
    convolved : numpy.ndarray
        HRF-convolved features with same shape as input (if mode='same') or
        expanded along time dimension (if mode='full').

    Notes
    -----
    Edge Effects:
    - mode='same' truncates the convolution to match input size, which can
      introduce edge artifacts at the beginning and end of the signal.
    - mode='full' includes padding but results in a longer output.

    To minimize edge effects:
    1. Use mode='full' and then trim appropriately
    2. Add padding to the input before convolution
    3. Analyze only the middle portion of convolved signal

    Example
    -------
    >>> features = np.random.randn(100, 5)  # 100 timepoints, 5 features
    >>> convolved = apply_hrf(features, tr=1.5, mode='same')
    >>> convolved.shape
    (100, 5)

    >>> # 3D audio features
    >>> audio_features = np.random.randn(100, 2048, 65)
    >>> convolved_audio = apply_hrf(audio_features, tr=1.5, mode='same')
    >>> convolved_audio.shape
    (100, 2048, 65)
    """
    # Get canonical HRF kernel
    hrf = get_canonical_hrf(tr=tr)

    # Handle 3D audio features (n_trs, n_mels, frames_per_tr)
    if features.ndim == 3:
        n_timepoints, n_mels, frames_per_tr = features.shape

        # Determine output shape
        if mode == 'same':
            output_shape = (n_timepoints, n_mels, frames_per_tr)
        else:  # mode == 'full'
            output_shape = (n_timepoints + len(hrf) - 1, n_mels, frames_per_tr)

        convolved = np.zeros(output_shape, dtype=features.dtype)

        # Convolve each mel × frame combination
        for mel_idx in range(n_mels):
            for frame_idx in range(frames_per_tr):
                convolved[:, mel_idx, frame_idx] = signal.convolve(
                    features[:, mel_idx, frame_idx], hrf, mode=mode
                )

        return convolved

    # Handle 1D vs 2D input (original code)
    if features.ndim == 1:
        features = features[:, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False

    n_timepoints, n_features = features.shape

    # Convolve each feature channel with HRF
    convolved = np.zeros_like(features) if mode == 'same' else \
                np.zeros((n_timepoints + len(hrf) - 1, n_features))

    for i in range(n_features):
        convolved[:, i] = signal.convolve(features[:, i], hrf, mode=mode)

    # Return to original shape if input was 1D
    if squeeze_output:
        convolved = convolved.squeeze(axis=1)

    return convolved


def convolve_with_padding(features, tr=1.5, padding_duration=10.0):
    """
    Convolve features with HRF using pre/post-stimulus padding.

    This function minimizes edge effects by adding padding before and after
    the signal, performing convolution, then trimming back to original size.
    Useful when edge artifacts would bias downstream analyses.

    UPDATED: Now handles 3D audio features (n_trs, n_mels, frames_per_tr).

    Parameters
    ----------
    features : numpy.ndarray
        Input stimulus features with shape:
        - (n_timepoints,) for 1D
        - (n_timepoints, n_features) for 2D
        - (n_timepoints, n_mels, frames_per_tr) for 3D audio
    tr : float
        Repetition time in seconds. Default is 1.5 seconds.
    padding_duration : float
        Duration of zero-padding to add before and after signal, in seconds.
        Default is 10 seconds (~6-7 samples at TR=1.5). Should be at least
        as long as the HRF duration (~20 seconds) for best results.

    Returns
    -------
    convolved : numpy.ndarray
        HRF-convolved features with same shape as input.

    Notes
    -----
    The padding approach:
    1. Adds zero-padding before and after the signal
    2. Performs full convolution on the padded signal
    3. Trims the result back to original timepoints

    This eliminates most edge effects but may introduce slight artifacts
    at the very beginning and end depending on padding_duration.

    Example
    -------
    >>> features = np.random.randn(100)
    >>> convolved = convolve_with_padding(features, tr=1.5, padding_duration=10.0)
    >>> convolved.shape
    (100,)

    >>> # 3D audio features
    >>> audio_features = np.random.randn(100, 2048, 65)
    >>> convolved_audio = convolve_with_padding(audio_features, tr=1.5)
    >>> convolved_audio.shape
    (100, 2048, 65)
    """
    # Determine padding size in samples
    padding_samples = int(np.round(padding_duration / tr))

    # Handle 1D, 2D, and 3D input
    if features.ndim == 1:
        features_padded = np.pad(features, (padding_samples, padding_samples),
                                mode='constant', constant_values=0)
        squeeze_output = True
    elif features.ndim == 2:
        features_padded = np.pad(features,
                               ((padding_samples, padding_samples), (0, 0)),
                               mode='constant', constant_values=0)
        squeeze_output = False
    elif features.ndim == 3:
        # 3D audio: (n_trs, n_mels, frames_per_tr)
        features_padded = np.pad(features,
                               ((padding_samples, padding_samples), (0, 0), (0, 0)),
                               mode='constant', constant_values=0)
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported feature dimensionality: {features.ndim}")

    # Convolve with padding
    convolved_padded = apply_hrf(features_padded, tr=tr, mode='full')

    # Trim to original size
    # The convolution output size is: (n_padded + n_hrf - 1)
    # We want the central portion that corresponds to the original signal
    n_hrf = len(get_canonical_hrf(tr=tr))
    start_idx = padding_samples + n_hrf // 2
    end_idx = start_idx + features.shape[0]

    convolved = convolved_padded[start_idx:end_idx]

    if squeeze_output:
        convolved = convolved.squeeze()

    return convolved


def get_hrf_peak_latency(tr=1.5):
    """
    Get the latency (time to peak) of the canonical HRF.

    Parameters
    ----------
    tr : float
        Repetition time in seconds. Default is 1.5 seconds.

    Returns
    -------
    latency : float
        Time (in seconds) from stimulus onset to HRF peak.

    Example
    -------
    >>> latency = get_hrf_peak_latency(tr=1.5)
    >>> 4.0 <= latency <= 6.0  # Should be in typical range
    True
    """
    hrf = get_canonical_hrf(tr=tr)
    peak_idx = np.argmax(hrf)
    latency = peak_idx * tr
    return latency

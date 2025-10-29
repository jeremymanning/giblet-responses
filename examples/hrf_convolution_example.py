"""
HRF Convolution Example
======================

This example demonstrates the basic usage of the HRF module for predicting
BOLD responses from stimulus features using the canonical Glover HRF model.
"""

import numpy as np
import matplotlib.pyplot as plt
from giblet.alignment.hrf import apply_hrf, get_canonical_hrf, get_hrf_peak_latency


def example_1_basic_hrf():
    """Example 1: Get and visualize the canonical HRF."""
    print("=" * 60)
    print("Example 1: Canonical HRF")
    print("=" * 60)

    # Get the HRF kernel at TR=1.5s
    hrf = get_canonical_hrf(tr=1.5)
    peak_latency = get_hrf_peak_latency(tr=1.5)

    print(f"HRF shape: {hrf.shape}")
    print(f"HRF peak latency: {peak_latency:.1f} seconds")
    print(f"HRF peak value (normalized): {np.max(hrf):.3f}")
    print(f"HRF values: {hrf[:5]}...")  # First 5 values
    print()


def example_2_impulse_response():
    """Example 2: Convolve a stimulus impulse with HRF."""
    print("=" * 60)
    print("Example 2: Impulse Response")
    print("=" * 60)

    # Create stimulus with single impulse at t=30s (index 20 at TR=1.5)
    n_timepoints = 100
    stimulus = np.zeros(n_timepoints)
    stimulus[20] = 1.0

    # Convolve with HRF using full mode to see complete response
    response = apply_hrf(stimulus, tr=1.5, mode='full')

    # Find peak of convolved response
    peak_idx = np.argmax(response)
    peak_latency = peak_idx * 1.5  # Convert to seconds (but accounting for padding)

    print(f"Stimulus impulse at index: 20")
    print(f"Response peak at index: {peak_idx}")
    print(f"Shift: {peak_idx - 20} samples = {(peak_idx - 20) * 1.5:.1f} seconds")
    print(f"Response peak value: {np.max(response):.3f}")
    print()


def example_3_multifeature():
    """Example 3: Convolve multiple feature channels."""
    print("=" * 60)
    print("Example 3: Multi-Feature Convolution")
    print("=" * 60)

    # Create 4 independent stimulus features
    n_timepoints = 200
    features = np.zeros((n_timepoints, 4))

    # Feature 1: Sine wave (e.g., visual motion)
    features[:, 0] = np.sin(2 * np.pi * np.arange(n_timepoints) / 50)

    # Feature 2: Sparse pulses (e.g., face presence)
    features[[30, 80, 130], 1] = 1.0

    # Feature 3: Ramp (e.g., attention level)
    features[:, 2] = np.linspace(0, 1, n_timepoints)

    # Feature 4: Noise (e.g., random background)
    features[:, 3] = np.random.randn(n_timepoints) * 0.3

    # Convolve all features with HRF
    convolved = apply_hrf(features, tr=1.5, mode='same')

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {convolved.shape}")
    print(f"Feature 1 RMS (original): {np.sqrt(np.mean(features[:, 0]**2)):.3f}")
    print(f"Feature 1 RMS (convolved): {np.sqrt(np.mean(convolved[:, 0]**2)):.3f}")
    print()


def example_4_realistic_scenario():
    """Example 4: Realistic scenario with sparse stimulus events."""
    print("=" * 60)
    print("Example 4: Realistic Sparse Stimulus Scenario")
    print("=" * 60)

    # Simulate 10-minute fMRI scan (TR=1.5s, 400 timepoints)
    n_timepoints = 400
    tr = 1.5

    # Create stimulus: sparse visual feature activations
    stimulus = np.zeros(n_timepoints)

    # Add 15 random stimulus events
    event_times = np.random.choice(n_timepoints, size=15, replace=False)
    stimulus[event_times] = 1.0

    # Convolve with HRF to get predicted BOLD
    predicted_bold = apply_hrf(stimulus, tr=tr, mode='same')

    print(f"Scan duration: {n_timepoints * tr / 60:.1f} minutes")
    print(f"Number of stimulus events: {np.sum(stimulus > 0)}")
    print(f"Stimulus sparsity: {np.sum(stimulus > 0) / len(stimulus):.1%}")
    print(f"Predicted BOLD peak: {np.max(predicted_bold):.3f}")
    print(f"Predicted BOLD min: {np.min(predicted_bold):.3f}")
    print()


def example_5_edge_handling():
    """Example 5: Demonstrating edge effect handling."""
    print("=" * 60)
    print("Example 5: Edge Effect Handling")
    print("=" * 60)

    # Create test stimulus
    stimulus = np.zeros(100)
    stimulus[10] = 1.0

    # Compare different convolution modes
    response_same = apply_hrf(stimulus, tr=1.5, mode='same')
    response_full = apply_hrf(stimulus, tr=1.5, mode='full')

    # Compare padding approach
    from giblet.alignment.hrf import convolve_with_padding
    response_padded = convolve_with_padding(stimulus, tr=1.5, padding_duration=10.0)

    print(f"mode='same' shape: {response_same.shape}")
    print(f"mode='full' shape: {response_full.shape}")
    print(f"padded shape: {response_padded.shape}")
    print()
    print("Edge effects comparison:")
    print(f"  'same' mode - first 5 values: {response_same[:5]}")
    print(f"  'full' mode - first 5 values: {response_full[:5]}")
    print(f"  padded - first 5 values: {response_padded[:5]}")
    print()
    print("Use mode='full' for impulse response analysis (cleaner)")
    print("Use mode='same' for regression models (matches input shape)")
    print("Use padding for minimum edge artifacts")
    print()


def example_6_visualization():
    """Example 6: Create visualizations."""
    print("=" * 60)
    print("Example 6: Creating Visualizations")
    print("=" * 60)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tr = 1.5

    # Plot 1: HRF kernel
    hrf = get_canonical_hrf(tr=tr)
    time = np.arange(len(hrf)) * tr
    axes[0, 0].plot(time, hrf, 'b-', linewidth=2)
    axes[0, 0].axvline(get_hrf_peak_latency(tr=tr), color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('HRF amplitude')
    axes[0, 0].set_title('Canonical HRF (Glover Model)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Impulse response
    stimulus = np.zeros(80)
    stimulus[15] = 1.0
    response = apply_hrf(stimulus, tr=tr, mode='same')
    time_stimulus = np.arange(len(stimulus)) * tr
    axes[0, 1].bar(time_stimulus, stimulus, width=tr*0.8, alpha=0.5, label='Stimulus')
    axes[0, 1].plot(time_stimulus, response, 'r-', linewidth=2, label='BOLD response')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Impulse Response (Temporal Shift Visible)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Multi-feature convolution
    features = np.zeros((100, 2))
    features[20, 0] = 1.0
    features[60, 1] = 1.0
    convolved = apply_hrf(features, tr=tr, mode='same')
    time_full = np.arange(len(features)) * tr

    axes[1, 0].plot(time_full, convolved[:, 0], label='Feature 1', linewidth=2)
    axes[1, 0].plot(time_full, convolved[:, 1], label='Feature 2', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('BOLD response')
    axes[1, 0].set_title('Multi-Feature Convolution (Independent)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Sparse stimulus
    stimulus_sparse = np.zeros(120)
    stimulus_sparse[[15, 40, 70, 95]] = 1.0
    response_sparse = apply_hrf(stimulus_sparse, tr=tr, mode='same')
    time_sparse = np.arange(len(stimulus_sparse)) * tr

    axes[1, 1].bar(time_sparse, stimulus_sparse, width=tr*0.8, alpha=0.3, color='blue', label='Events')
    axes[1, 1].plot(time_sparse, response_sparse, 'r-', linewidth=2, label='Predicted BOLD')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Sparse Stimulus Events → BOLD Prediction')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("Visualization created (4 subplots)")
    print("- Top-left: Canonical HRF kernel")
    print("- Top-right: Impulse response (showing temporal shift)")
    print("- Bottom-left: Multi-feature independent convolution")
    print("- Bottom-right: Sparse stimulus events to BOLD prediction")
    print()


if __name__ == '__main__':
    # Run all examples
    example_1_basic_hrf()
    example_2_impulse_response()
    example_3_multifeature()
    example_4_realistic_scenario()
    example_5_edge_handling()
    example_6_visualization()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nFor more information:")
    print("  - See giblet/alignment/hrf.py for implementation")
    print("  - See tests/integration/test_hrf.py for comprehensive tests")
    print("  - Run pytest tests/integration/test_hrf.py -v to see all tests")

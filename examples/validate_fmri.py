"""
fMRI modality validation script.

Performs comprehensive round-trip validation:
1. Load real fMRI data from .nii.gz files (s1, s2, s3)
2. Extract features using shared mask
3. Reconstruct NIfTI files from features
4. Verify voxel values match (np.allclose)
5. Plot sample voxel timeseries
6. Create visual brain mask overlays
7. Test on multiple subjects

All tests use REAL data - NO MOCKS.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.data.fmri import FMRIProcessor


def calculate_voxel_correlation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate correlation between original and reconstructed voxel timeseries.

    Parameters
    ----------
    original : np.ndarray
        Shape (n_trs, n_voxels) original timeseries
    reconstructed : np.ndarray
        Shape (n_trs, n_voxels) reconstructed timeseries

    Returns
    -------
    mean_correlation : float
        Mean correlation across all voxels
    """
    n_trs, n_voxels = original.shape

    # Calculate correlation for each voxel
    correlations = []
    for voxel_idx in range(n_voxels):
        if np.std(original[:, voxel_idx]) > 0 and np.std(reconstructed[:, voxel_idx]) > 0:
            corr, _ = pearsonr(original[:, voxel_idx], reconstructed[:, voxel_idx])
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def plot_voxel_timeseries(original: np.ndarray, reconstructed: np.ndarray,
                         voxel_indices: list, subject_id: str, output_path: Path):
    """
    Plot timeseries comparison for sample voxels.

    Parameters
    ----------
    original : np.ndarray
        Shape (n_trs, n_voxels) original timeseries
    reconstructed : np.ndarray
        Shape (n_trs, n_voxels) reconstructed timeseries
    voxel_indices : list
        List of voxel indices to plot
    subject_id : str
        Subject ID for labeling
    output_path : Path
        Path to save plot
    """
    n_voxels = len(voxel_indices)
    fig, axes = plt.subplots(n_voxels, 1, figsize=(15, 3*n_voxels))

    if n_voxels == 1:
        axes = [axes]

    for i, voxel_idx in enumerate(voxel_indices):
        time = np.arange(len(original)) * 1.5  # TR = 1.5s

        # Original
        axes[i].plot(time, original[:, voxel_idx], label='Original',
                    alpha=0.7, linewidth=1)

        # Reconstructed
        axes[i].plot(time, reconstructed[:, voxel_idx], label='Reconstructed',
                    alpha=0.7, linewidth=1, linestyle='--')

        # Calculate correlation for this voxel
        if np.std(original[:, voxel_idx]) > 0:
            corr, _ = pearsonr(original[:, voxel_idx], reconstructed[:, voxel_idx])
        else:
            corr = 0.0

        axes[i].set_title(f'Voxel {voxel_idx} - Correlation: {corr:.4f}', fontsize=12)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('BOLD Signal')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Subject {subject_id} - Sample Voxel Timeseries', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved timeseries plot to {output_path.name}")


def plot_brain_slices(nii_img: nib.Nifti1Image, mask_img: nib.Nifti1Image,
                     subject_id: str, output_path: Path, time_idx: int = 0):
    """
    Plot brain slices with mask overlay.

    Parameters
    ----------
    nii_img : nib.Nifti1Image
        Brain fMRI image
    mask_img : nib.Nifti1Image
        Brain mask image
    subject_id : str
        Subject ID for labeling
    output_path : Path
        Path to save plot
    time_idx : int
        Time index to visualize
    """
    # Get data
    brain_data = nii_img.get_fdata()[:, :, :, time_idx]
    mask_data = mask_img.get_fdata().astype(bool)

    # Select representative slices
    x_mid = brain_data.shape[0] // 2
    y_mid = brain_data.shape[1] // 2
    z_mid = brain_data.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Sagittal slice (x)
    axes[0, 0].imshow(brain_data[x_mid, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Sagittal (x={x_mid})', fontsize=12)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(brain_data[x_mid, :, :].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 0].imshow(mask_data[x_mid, :, :].T, cmap='Reds', origin='lower', alpha=0.5)
    axes[1, 0].set_title(f'Sagittal with Mask', fontsize=12)
    axes[1, 0].axis('off')

    # Coronal slice (y)
    axes[0, 1].imshow(brain_data[:, y_mid, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Coronal (y={y_mid})', fontsize=12)
    axes[0, 1].axis('off')

    axes[1, 1].imshow(brain_data[:, y_mid, :].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 1].imshow(mask_data[:, y_mid, :].T, cmap='Reds', origin='lower', alpha=0.5)
    axes[1, 1].set_title(f'Coronal with Mask', fontsize=12)
    axes[1, 1].axis('off')

    # Axial slice (z)
    axes[0, 2].imshow(brain_data[:, :, z_mid].T, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'Axial (z={z_mid})', fontsize=12)
    axes[0, 2].axis('off')

    axes[1, 2].imshow(brain_data[:, :, z_mid].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 2].imshow(mask_data[:, :, z_mid].T, cmap='Reds', origin='lower', alpha=0.5)
    axes[1, 2].set_title(f'Axial with Mask', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle(f'Subject {subject_id} - Brain Slices (TR {time_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved brain slices to {output_path.name}")


def validate_subject(processor: FMRIProcessor, nii_path: Path,
                    subject_id: str, output_dir: Path) -> dict:
    """
    Validate fMRI processing for a single subject.

    Parameters
    ----------
    processor : FMRIProcessor
        FMRIProcessor instance with shared mask
    nii_path : Path
        Path to subject's NIfTI file
    subject_id : str
        Subject ID
    output_dir : Path
        Directory for output files

    Returns
    -------
    metrics : dict
        Validation metrics
    """
    print("\n" + "="*80)
    print(f"VALIDATING SUBJECT {subject_id}")
    print("="*80)

    # Extract features
    print(f"\n1. Extracting features from NIfTI file...")
    features, coordinates, metadata = processor.nii_to_features(nii_path)

    print(f"   Features shape: {features.shape}")
    print(f"   Coordinates shape: {coordinates.shape}")
    print(f"   Non-zero voxels: {np.sum(np.any(features != 0, axis=0)):,}")
    print(f"   Features range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"   Features mean: {features.mean():.2f}, std: {features.std():.2f}")

    # Store original features
    original_features = features.copy()

    # Reconstruct NIfTI
    print(f"\n2. Reconstructing NIfTI from features...")
    reconstructed_path = output_dir / f'fmri_reconstructed_{subject_id}.nii.gz'
    reconstructed_img = processor.features_to_nii(
        features,
        coordinates,
        reconstructed_path
    )

    # Re-extract features from reconstructed NIfTI
    print(f"\n3. Re-extracting features from reconstructed NIfTI...")
    reconstructed_features, _, _ = processor.nii_to_features(reconstructed_path)

    print(f"   Reconstructed features shape: {reconstructed_features.shape}")

    # Verify shapes match
    assert original_features.shape == reconstructed_features.shape, \
        f"Shape mismatch: {original_features.shape} vs {reconstructed_features.shape}"

    # Calculate metrics
    print(f"\n4. Calculating validation metrics...")

    # Exact match check
    exact_match = np.allclose(original_features, reconstructed_features, atol=1e-6)
    print(f"   Exact match (tolerance=1e-6): {exact_match}")

    # Calculate differences
    abs_diff = np.abs(original_features - reconstructed_features)
    rel_diff = abs_diff / (np.abs(original_features) + 1e-10)

    print(f"   Absolute difference:")
    print(f"     Mean: {abs_diff.mean():.6e}")
    print(f"     Max: {abs_diff.max():.6e}")
    print(f"     Std: {abs_diff.std():.6e}")

    print(f"   Relative difference:")
    print(f"     Mean: {rel_diff.mean():.6e}")
    print(f"     Max: {rel_diff.max():.6e}")

    # Calculate correlation
    mean_corr = calculate_voxel_correlation(original_features, reconstructed_features)
    print(f"   Mean voxel correlation: {mean_corr:.6f}")

    # MSE
    mse = np.mean((original_features - reconstructed_features) ** 2)
    print(f"   MSE: {mse:.6e}")

    # Plot sample voxel timeseries
    print(f"\n5. Plotting sample voxel timeseries...")

    # Select voxels with high variance (more interesting)
    voxel_vars = np.var(original_features, axis=0)
    high_var_indices = np.argsort(voxel_vars)[-5:]  # Top 5 highest variance

    timeseries_path = output_dir / f'fmri_timeseries_{subject_id}.png'
    plot_voxel_timeseries(
        original_features,
        reconstructed_features,
        high_var_indices.tolist(),
        subject_id,
        timeseries_path
    )

    # Plot brain slices with mask
    print(f"\n6. Plotting brain slices with mask...")
    slices_path = output_dir / f'fmri_brain_slices_{subject_id}.png'

    # Load original image for visualization
    original_img = nib.load(str(nii_path))

    plot_brain_slices(
        original_img,
        processor._shared_mask_img,
        subject_id,
        slices_path,
        time_idx=100  # Middle of timeseries
    )

    # Quality assessment
    print(f"\n7. Quality assessment:")
    if exact_match:
        print(f"   ✓ PERFECT reconstruction (exact match)")
    elif abs_diff.max() < 1e-3:
        print(f"   ✓ EXCELLENT reconstruction (max diff < 0.001)")
    elif abs_diff.max() < 1e-1:
        print(f"   ✓ GOOD reconstruction (max diff < 0.1)")
    else:
        print(f"   ⚠ WARNING: Larger differences detected")

    if mean_corr > 0.99:
        print(f"   ✓ EXCELLENT correlation (> 0.99)")
    elif mean_corr > 0.95:
        print(f"   ✓ GOOD correlation (> 0.95)")
    else:
        print(f"   ⚠ WARNING: Lower correlation")

    return {
        'subject_id': subject_id,
        'n_voxels': features.shape[1],
        'n_trs': features.shape[0],
        'exact_match': exact_match,
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'mean_voxel_corr': mean_corr,
        'mse': mse
    }


def validate_fmri_modality():
    """
    Main validation function for fMRI modality.
    """
    print("\n" + "="*80)
    print("fMRI MODALITY VALIDATION")
    print("="*80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'sherlock_nii'
    output_dir = project_root / 'validation_outputs'
    output_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print(f"\n❌ ERROR: fMRI data directory not found: {data_dir}")
        print("Please run download_data_from_dropbox.sh first")
        return

    print(f"\nfMRI data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize processor
    processor = FMRIProcessor(
        tr=1.5,
        max_trs=920,  # Full Sherlock movie
        mask_threshold=0.5
    )

    # Find available subjects
    nii_files = sorted(data_dir.glob('sherlock_movie_s*.nii.gz'),
                      key=lambda f: int(f.name.split('_')[-1].split('.')[0][1:]))

    print(f"\nFound {len(nii_files)} subject files")

    # Create shared mask
    print("\n" + "="*80)
    print("CREATING SHARED MASK")
    print("="*80)

    # Use first 3 subjects for mask creation (for speed in validation)
    # In production, use all subjects
    mask_files = nii_files[:3]
    print(f"\nUsing {len(mask_files)} subjects for mask creation:")
    for f in mask_files:
        print(f"  {f.name}")

    mask_array, mask_img = processor.create_shared_mask(mask_files)

    # Save mask
    mask_path = output_dir / 'shared_brain_mask.nii.gz'
    processor.save_mask(mask_path)

    # Get mask info
    mask_info = processor.get_mask_info()
    print(f"\nMask statistics:")
    print(f"  Total voxels: {mask_info['total_voxels']:,}")
    print(f"  Brain voxels: {mask_info['n_voxels']:,}")
    print(f"  Brain proportion: {mask_info['proportion_brain']:.2%}")
    print(f"  Mask shape: {mask_info['mask_shape']}")

    # Validate each subject
    print("\n" + "="*80)
    print("SUBJECT VALIDATION")
    print("="*80)

    # Test on subjects 1, 2, 3
    test_subjects = [
        (nii_files[0], 's1'),
        (nii_files[1], 's2'),
        (nii_files[2], 's3')
    ]

    all_metrics = []

    for nii_path, subject_id in test_subjects:
        metrics = validate_subject(processor, nii_path, subject_id, output_dir)
        all_metrics.append(metrics)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print("\nMetrics by subject:")
    print(f"{'Subject':<10} {'Voxels':>10} {'TRs':>6} {'Exact Match':>12} "
          f"{'Mean Diff':>12} {'Max Diff':>12} {'Correlation':>12}")
    print("-" * 90)

    for m in all_metrics:
        print(f"{m['subject_id']:<10} "
              f"{m['n_voxels']:>10,} "
              f"{m['n_trs']:>6} "
              f"{'Yes' if m['exact_match'] else 'No':>12} "
              f"{m['mean_abs_diff']:>12.6e} "
              f"{m['max_abs_diff']:>12.6e} "
              f"{m['mean_voxel_corr']:>12.6f}")

    # Overall statistics
    n_exact = sum(1 for m in all_metrics if m['exact_match'])
    avg_corr = np.mean([m['mean_voxel_corr'] for m in all_metrics])
    avg_max_diff = np.mean([m['max_abs_diff'] for m in all_metrics])

    print(f"\nOverall statistics:")
    print(f"  Exact matches: {n_exact}/{len(all_metrics)}")
    print(f"  Average correlation: {avg_corr:.6f}")
    print(f"  Average max difference: {avg_max_diff:.6e}")

    print(f"\nGenerated outputs:")
    output_files = sorted(output_dir.glob('fmri_*'))
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:60s} ({size_mb:.2f} MB)")

    # Technical notes
    print("\n" + "="*80)
    print("TECHNICAL NOTES")
    print("="*80)
    print("""
Round-trip validation methodology:
1. Load original .nii.gz file (4D array: x, y, z, time)
2. Apply shared brain mask to extract only brain voxels
3. Flatten to (n_trs, n_voxels) matrix
4. Reconstruct full 4D volume from matrix using coordinates
5. Save as new .nii.gz file
6. Re-load and re-extract to verify round-trip

Expected results:
- Exact match (within floating-point precision)
- Correlation ≈ 1.0 for all voxels
- Differences should be < 1e-6 (floating-point precision)

Any differences larger than 1e-3 indicate potential issues in:
- Coordinate mapping
- Mask application
- NIfTI file I/O
- Affine transformation handling
    """)

    # Final verdict
    print("\n" + "="*80)
    if n_exact == len(all_metrics):
        print("✅ fMRI VALIDATION PASSED - PERFECT RECONSTRUCTION")
    elif avg_corr > 0.99 and avg_max_diff < 1e-3:
        print("✅ fMRI VALIDATION PASSED - EXCELLENT RECONSTRUCTION")
    elif avg_corr > 0.95 and avg_max_diff < 1e-1:
        print("✅ fMRI VALIDATION PASSED - GOOD RECONSTRUCTION")
    else:
        print("⚠ fMRI VALIDATION WARNING - QUALITY BELOW EXPECTED")
    print("="*80)

    return all_metrics


if __name__ == '__main__':
    try:
        metrics = validate_fmri_modality()
        avg_corr = np.mean([m['mean_voxel_corr'] for m in metrics])
        n_exact = sum(1 for m in metrics if m['exact_match'])
        print(f"\n✓ fMRI validation complete.")
        print(f"  Exact matches: {n_exact}/{len(metrics)}")
        print(f"  Average correlation: {avg_corr:.6f}")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Comprehensive tests for fMRI processing module.

Tests use real fMRI data from all 17 Sherlock subjects.
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

from giblet.data.fmri import FMRIProcessor


# Test configuration
DATA_DIR = Path("/Users/jmanning/giblet-responses/data/sherlock_nii")
EXPECTED_N_VOXELS = 83300  # Expected number of brain voxels
EXPECTED_SHAPE = (61, 73, 61)  # Expected spatial dimensions
TR = 1.5  # Repetition time in seconds
N_SUBJECTS = 17
MAX_TRS = 920  # Truncate to stimulus duration


class TestFMRIProcessor:
    """Test FMRIProcessor with real Sherlock data."""

    @pytest.fixture
    def processor(self):
        """Create FMRIProcessor instance."""
        return FMRIProcessor(tr=TR, max_trs=MAX_TRS, mask_threshold=0.5)

    @pytest.fixture
    def nii_files(self):
        """Get list of all NIfTI files."""
        files = sorted(DATA_DIR.glob("sherlock_movie_s*.nii.gz"))
        assert len(files) == N_SUBJECTS, f"Expected {N_SUBJECTS} files, found {len(files)}"
        return files

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for outputs."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    def test_data_files_exist(self, nii_files):
        """Test that all 17 subject files exist."""
        print(f"\n=== Testing Data Files ===")
        print(f"Data directory: {DATA_DIR}")
        print(f"Found {len(nii_files)} files:")
        for f in nii_files:
            print(f"  - {f.name}")
            assert f.exists(), f"File not found: {f}"

        # Check that files are numbered s1 through s17
        subject_ids = [f.stem.replace('.nii', '').split('_')[-1] for f in nii_files]
        expected_ids = [f"s{i}" for i in range(1, N_SUBJECTS + 1)]
        assert subject_ids == expected_ids, f"Subject IDs mismatch: {subject_ids}"

    def test_load_single_nii(self, nii_files):
        """Test loading a single NIfTI file."""
        print(f"\n=== Testing Single NIfTI Load ===")
        test_file = nii_files[0]
        print(f"Loading: {test_file.name}")

        img = nib.load(str(test_file))
        data = img.get_fdata()

        print(f"Shape: {data.shape}")
        print(f"Dimensions: {data.ndim}")
        print(f"Data type: {data.dtype}")

        assert data.ndim == 4, f"Expected 4D data, got {data.ndim}D"
        assert data.shape[:3] == EXPECTED_SHAPE, f"Spatial shape mismatch: {data.shape[:3]}"
        print(f"Timepoints: {data.shape[3]}")

    def test_create_shared_mask(self, processor, nii_files):
        """Test shared mask creation across all subjects."""
        print(f"\n=== Testing Shared Mask Creation ===")

        mask_array, mask_img = processor.create_shared_mask(nii_files)

        # Check mask shape
        assert mask_array.shape == EXPECTED_SHAPE, f"Mask shape mismatch: {mask_array.shape}"
        assert mask_array.dtype == bool, f"Mask should be boolean, got {mask_array.dtype}"

        # Check number of brain voxels
        n_voxels = np.sum(mask_array)
        print(f"Brain voxels: {n_voxels:,}")
        print(f"Total voxels: {np.prod(mask_array.shape):,}")
        print(f"Proportion brain: {n_voxels / np.prod(mask_array.shape):.2%}")

        # Allow some tolerance in expected voxel count
        assert abs(n_voxels - EXPECTED_N_VOXELS) < 5000, \
            f"Expected ~{EXPECTED_N_VOXELS:,} voxels, got {n_voxels:,}"

        # Check that mask is a NIfTI image
        assert isinstance(mask_img, nib.Nifti1Image), "Mask should be NIfTI image"

        # Check that coordinates were computed
        assert processor._coordinates is not None, "Coordinates not computed"
        assert processor._coordinates.shape == (n_voxels, 3), \
            f"Coordinates shape mismatch: {processor._coordinates.shape}"

        print(f"Coordinates shape: {processor._coordinates.shape}")

    def test_mask_info(self, processor, nii_files):
        """Test get_mask_info method."""
        print(f"\n=== Testing Mask Info ===")

        processor.create_shared_mask(nii_files)
        info = processor.get_mask_info()

        print(f"Mask info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        assert 'n_voxels' in info
        assert 'total_voxels' in info
        assert 'proportion_brain' in info
        assert 'proportion_zero' in info
        assert 'mask_shape' in info

        # Check that proportion_brain + proportion_zero = 1
        assert abs(info['proportion_brain'] + info['proportion_zero'] - 1.0) < 1e-6

    def test_nii_to_features_single_subject(self, processor, nii_files):
        """Test extracting features from a single subject."""
        print(f"\n=== Testing NIfTI to Features (Single Subject) ===")

        # Create mask first
        processor.create_shared_mask(nii_files)

        # Extract features from first subject
        test_file = nii_files[0]
        print(f"Processing: {test_file.name}")

        features, coordinates, metadata = processor.nii_to_features(test_file)

        # Check shapes
        print(f"Features shape: {features.shape}")
        print(f"Coordinates shape: {coordinates.shape}")
        print(f"Metadata shape: {metadata.shape}")

        assert features.ndim == 2, f"Features should be 2D, got {features.ndim}D"
        assert features.shape[0] == MAX_TRS, f"Expected {MAX_TRS} TRs, got {features.shape[0]}"
        assert coordinates.shape[1] == 3, f"Coordinates should have 3 columns, got {coordinates.shape[1]}"
        assert features.shape[1] == coordinates.shape[0], "Feature and coordinate dimensions mismatch"

        # Check metadata
        assert 'tr_index' in metadata.columns
        assert 'time' in metadata.columns
        assert len(metadata) == MAX_TRS

        # Check that features are not all zeros
        n_nonzero = np.sum(np.any(features != 0, axis=0))
        print(f"Non-zero voxels: {n_nonzero:,}")
        assert n_nonzero > 0, "All features are zero"

        # Check time values
        expected_times = np.arange(MAX_TRS) * TR
        np.testing.assert_allclose(metadata['time'].values, expected_times)

    def test_nii_to_features_all_subjects(self, processor, nii_files):
        """Test extracting features from all subjects."""
        print(f"\n=== Testing NIfTI to Features (All Subjects) ===")

        # Create mask first
        processor.create_shared_mask(nii_files)
        n_voxels = processor.get_mask_info()['n_voxels']

        print(f"Processing {len(nii_files)} subjects...")

        all_features = []
        for i, nii_file in enumerate(nii_files):
            print(f"  {i+1}/{len(nii_files)}: {nii_file.name}")
            features, coordinates, metadata = processor.nii_to_features(nii_file)

            # Check consistent shape
            assert features.shape == (MAX_TRS, n_voxels), \
                f"Inconsistent shape for {nii_file.name}: {features.shape}"

            all_features.append(features)

        print(f"Successfully processed all {len(nii_files)} subjects")

    def test_features_to_nii_roundtrip(self, processor, nii_files, temp_dir):
        """Test bidirectional conversion: NIfTI -> features -> NIfTI."""
        print(f"\n=== Testing Round-trip Conversion ===")

        # Create mask
        processor.create_shared_mask(nii_files)

        # Load original
        test_file = nii_files[0]
        print(f"Original: {test_file.name}")

        # Convert to features
        features, coordinates, metadata = processor.nii_to_features(test_file)
        print(f"Extracted features: {features.shape}")

        # Convert back to NIfTI
        output_path = temp_dir / "roundtrip_test.nii.gz"
        reconstructed_img = processor.features_to_nii(
            features,
            coordinates,
            output_path
        )

        print(f"Reconstructed: {output_path}")

        # Load reconstructed
        reconstructed_data = reconstructed_img.get_fdata()
        print(f"Reconstructed shape: {reconstructed_data.shape}")

        # Load original with truncation
        original_img = nib.load(str(test_file))
        original_data = original_img.get_fdata()[:, :, :, :MAX_TRS]
        print(f"Original truncated shape: {original_data.shape}")

        # Compare shapes
        assert reconstructed_data.shape == original_data.shape, \
            f"Shape mismatch: {reconstructed_data.shape} vs {original_data.shape}"

        # Get mask
        mask = processor._shared_mask

        # Compare brain voxels
        for t in range(MAX_TRS):
            original_brain = original_data[:, :, :, t][mask]
            reconstructed_brain = reconstructed_data[:, :, :, t][mask]

            # Should be exactly equal (within floating point precision)
            np.testing.assert_allclose(
                reconstructed_brain,
                original_brain,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Brain voxels differ at timepoint {t}"
            )

        # Check that non-brain voxels are zero in reconstruction
        reconstructed_nonbrain = reconstructed_data[~mask]
        assert np.all(reconstructed_nonbrain == 0), "Non-brain voxels should be zero"

        print("Round-trip test passed!")

    def test_average_across_subjects(self, processor, nii_files):
        """Test cross-subject averaging."""
        print(f"\n=== Testing Cross-Subject Averaging ===")

        # Create mask
        processor.create_shared_mask(nii_files)

        # Extract features from first 3 subjects
        n_test_subjects = 3
        print(f"Testing with {n_test_subjects} subjects...")

        features_list = []
        for i in range(n_test_subjects):
            features, _, _ = processor.nii_to_features(nii_files[i])
            features_list.append(features)
            print(f"  Subject {i+1}: {features.shape}")

        # Average
        averaged = processor.average_across_subjects(features_list)
        print(f"Averaged shape: {averaged.shape}")

        # Check shape
        assert averaged.shape == features_list[0].shape, "Averaged shape mismatch"

        # Check that averaging actually happened (not just copying one subject)
        for features in features_list:
            # Should not be identical to any single subject
            # (extremely unlikely if there's noise)
            assert not np.allclose(averaged, features), \
                "Averaged features identical to single subject"

        # Sanity check: manual averaging should match
        manual_avg = np.mean(np.stack(features_list, axis=0), axis=0)
        np.testing.assert_allclose(averaged, manual_avg)

        print("Averaging test passed!")

    def test_load_all_subjects(self, processor):
        """Test load_all_subjects convenience method."""
        print(f"\n=== Testing Load All Subjects ===")

        features_list, coordinates_list, metadata_list, subject_ids = \
            processor.load_all_subjects(DATA_DIR)

        # Check counts
        assert len(features_list) == N_SUBJECTS, f"Expected {N_SUBJECTS} subjects"
        assert len(coordinates_list) == N_SUBJECTS
        assert len(metadata_list) == N_SUBJECTS
        assert len(subject_ids) == N_SUBJECTS

        print(f"Loaded {len(features_list)} subjects")
        print(f"Subject IDs: {subject_ids}")

        # Check subject IDs
        expected_ids = [f"s{i}" for i in range(1, N_SUBJECTS + 1)]
        assert subject_ids == expected_ids, f"Subject ID mismatch"

        # Check shapes
        n_voxels = features_list[0].shape[1]
        for i, features in enumerate(features_list):
            assert features.shape[0] == MAX_TRS, f"Subject {i+1}: wrong number of TRs"
            assert features.shape[1] == n_voxels, f"Subject {i+1}: wrong number of voxels"

        # Check that all coordinates are identical
        for i in range(1, len(coordinates_list)):
            np.testing.assert_array_equal(
                coordinates_list[0],
                coordinates_list[i],
                err_msg=f"Subject {i+1} coordinates differ from subject 1"
            )

        print("Load all subjects test passed!")

    def test_save_and_load_mask(self, processor, nii_files, temp_dir):
        """Test saving and loading mask."""
        print(f"\n=== Testing Save/Load Mask ===")

        # Create mask
        processor.create_shared_mask(nii_files)
        original_info = processor.get_mask_info()

        # Save mask
        mask_path = temp_dir / "test_mask.nii.gz"
        processor.save_mask(mask_path)
        print(f"Saved mask to: {mask_path}")

        # Create new processor and load mask
        processor2 = FMRIProcessor(tr=TR, max_trs=MAX_TRS)
        processor2.load_mask(mask_path)
        loaded_info = processor2.get_mask_info()

        # Compare info
        assert original_info['n_voxels'] == loaded_info['n_voxels']
        assert original_info['mask_shape'] == loaded_info['mask_shape']

        # Compare masks
        np.testing.assert_array_equal(
            processor._shared_mask,
            processor2._shared_mask
        )

        # Compare coordinates
        np.testing.assert_array_equal(
            processor._coordinates,
            processor2._coordinates
        )

        print("Save/load mask test passed!")

    def test_exact_voxel_count(self, processor, nii_files):
        """Test and report exact voxel count."""
        print(f"\n=== EXACT VOXEL COUNT TEST ===")

        processor.create_shared_mask(nii_files)
        info = processor.get_mask_info()

        print(f"\nFinal Mask Statistics:")
        print(f"  Brain voxels: {info['n_voxels']:,}")
        print(f"  Total voxels: {info['total_voxels']:,}")
        print(f"  Proportion brain: {info['proportion_brain']:.2%}")
        print(f"  Proportion zero: {info['proportion_zero']:.2%}")
        print(f"  Mask shape: {info['mask_shape']}")

        # Report exact count
        print(f"\n*** EXACT VOXEL COUNT: {info['n_voxels']:,} ***")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

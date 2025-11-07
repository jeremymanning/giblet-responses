"""
fMRI processing module for multimodal autoencoder project.

Handles bidirectional conversion between fMRI NIfTI files and feature matrices:
- NIfTI → Features: Extract timeseries from brain voxels using shared mask
- Features → NIfTI: Reconstruct NIfTI files from feature matrices
- Cross-subject averaging and alignment

Default TR = 1.5 seconds (configurable).
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from tqdm import tqdm


class FMRIProcessor:
    """
    Process fMRI data for multimodal autoencoder training.

    Handles:
    - Creation of shared brain mask across all subjects
    - Extraction of timeseries from brain voxels
    - Temporal truncation to match stimulus duration
    - Bidirectional conversion between NIfTI and feature matrices
    - Cross-subject averaging

    Parameters
    ----------
    tr : float, default=1.5
        fMRI repetition time in seconds
    max_trs : int, optional
        Maximum number of TRs to extract (for truncating to stimulus duration)
    mask_threshold : float, default=0.5
        Threshold for shared mask computation (proportion of subjects)
    normalize : bool, default=True
        Whether to z-score normalize fMRI data (per-subject normalization)
    """

    def __init__(
        self,
        tr: float = 1.5,
        max_trs: Optional[int] = None,
        mask_threshold: float = 0.5,
        normalize: bool = True,
    ):
        self.tr = tr
        self.max_trs = max_trs
        self.mask_threshold = mask_threshold
        self.normalize = normalize
        self._shared_mask = None
        self._shared_mask_img = None
        self._coordinates = None
        self._template_affine = None
        self._template_header = None

    def create_shared_mask(
        self, nii_files: List[Union[str, Path]]
    ) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """
        Create shared brain mask across all subjects.

        Creates individual masks for each subject based on temporal variability
        (standard deviation), then combines them using a voting scheme.

        Parameters
        ----------
        nii_files : list of str or Path
            List of paths to NIfTI files for all subjects

        Returns
        -------
        mask_array : np.ndarray
            3D boolean mask array (True = brain voxel)
        mask_img : nib.Nifti1Image
            NIfTI image object of the mask

        Notes
        -----
        - Uses temporal std > 10th percentile for each subject
        - Votes across subjects with threshold (default 0.5 = >50% of subjects)
        - Typical result: ~83,300 brain voxels (varies by dataset)
        """
        print(f"Creating shared mask from {len(nii_files)} subjects...")
        print(f"Using voting threshold: {self.mask_threshold}")

        # Load first file to get dimensions and template info
        first_img = nib.load(str(nii_files[0]))
        template_shape = first_img.shape[:3]
        template_affine = first_img.affine
        template_header = first_img.header

        # Create individual masks for each subject
        individual_masks = []
        print("Creating individual masks based on temporal variability...")

        for i, nii_file in enumerate(tqdm(nii_files, desc="Processing subjects")):
            img = nib.load(str(nii_file))
            data = img.get_fdata()

            # Use first 200 timepoints for speed (enough to assess variability)
            n_sample_trs = min(200, data.shape[3])
            sample_data = data[:, :, :, :n_sample_trs]

            # Calculate temporal standard deviation
            std_vol = np.std(sample_data, axis=3)

            # Create mask: voxels with std > 10th percentile of non-zero stds
            non_zero_stds = std_vol[std_vol > 0]
            if len(non_zero_stds) > 0:
                threshold = np.percentile(non_zero_stds, 10)
                mask = std_vol > threshold
            else:
                mask = np.zeros(template_shape, dtype=bool)

            individual_masks.append(mask)
            n_voxels = np.sum(mask)
            # print(f"  Subject {i+1}: {n_voxels:,} voxels")

        # Combine masks using voting
        print(f"Combining {len(individual_masks)} individual masks...")
        vote_counts = np.sum(np.stack(individual_masks, axis=0).astype(int), axis=0)
        vote_threshold = int(np.ceil(len(nii_files) * self.mask_threshold))

        # Final mask: voxels present in at least threshold proportion of subjects
        mask_array = vote_counts >= vote_threshold

        # Count non-zero voxels
        n_voxels = np.sum(mask_array)
        print(f"Shared mask created: {n_voxels:,} brain voxels")
        print(
            f"  Vote threshold: voxels in >={vote_threshold}/{len(nii_files)} subjects"
        )

        # Create NIfTI image for mask
        mask_img = nib.Nifti1Image(
            mask_array.astype(np.uint8), template_affine, template_header
        )

        # Store mask and template info
        self._shared_mask = mask_array
        self._shared_mask_img = mask_img
        self._template_affine = template_affine
        self._template_header = template_header

        # Compute and store voxel coordinates
        self._compute_voxel_coordinates()

        return mask_array, mask_img

    def _compute_voxel_coordinates(self):
        """
        Compute and store (x, y, z) coordinates for each brain voxel.

        Coordinates are stored in the order they appear when flattening
        the masked volume.
        """
        if self._shared_mask is None:
            raise RuntimeError("Shared mask must be created first")

        # Get coordinates of all True voxels
        coords = np.array(np.where(self._shared_mask)).T
        self._coordinates = coords
        print(f"Stored coordinates for {len(coords):,} voxels")

    def nii_to_features(
        self, nii_path: Union[str, Path], brain_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Extract timeseries features from NIfTI file.

        Parameters
        ----------
        nii_path : str or Path
            Path to NIfTI file
        brain_mask : np.ndarray, optional
            3D boolean brain mask. If None, uses shared mask.

        Returns
        -------
        features : np.ndarray
            Shape (n_trs, n_voxels) timeseries matrix
        coordinates : np.ndarray
            Shape (n_voxels, 3) array of (x, y, z) coordinates
        metadata : pd.DataFrame
            DataFrame with: tr_index, time

        Notes
        -----
        - Applies brain mask to extract only brain voxels
        - Truncates to max_trs if specified
        - Output is in (time, voxels) format for compatibility with models
        """
        nii_path = Path(nii_path)

        # Use shared mask if not provided
        if brain_mask is None:
            if self._shared_mask is None:
                raise RuntimeError("No mask provided and shared mask not created")
            brain_mask = self._shared_mask
            coordinates = self._coordinates
        else:
            # Compute coordinates for provided mask
            coordinates = np.array(np.where(brain_mask)).T

        # Load NIfTI file
        print(f"Loading {nii_path.name}...")
        img = nib.load(str(nii_path))
        data = img.get_fdata()

        # Store template info from first file
        if self._template_affine is None:
            self._template_affine = img.affine
            self._template_header = img.header

        # Get dimensions
        n_x, n_y, n_z, n_timepoints = data.shape
        print(f"  Shape: {data.shape}")
        print(f"  Timepoints: {n_timepoints}")

        # Truncate if needed
        if self.max_trs is not None:
            n_trs = min(n_timepoints, self.max_trs)
            data = data[:, :, :, :n_trs]
            print(f"  Truncated to {n_trs} TRs")
        else:
            n_trs = n_timepoints

        # Apply mask to extract brain voxels
        # Shape: (n_trs, n_voxels)
        features = np.zeros((n_trs, len(coordinates)), dtype=np.float32)

        for t in range(n_trs):
            volume = data[:, :, :, t]
            features[t] = volume[brain_mask]

        # Create metadata
        metadata = pd.DataFrame(
            {"tr_index": np.arange(n_trs), "time": np.arange(n_trs) * self.tr}
        )

        print(f"  Extracted features: {features.shape}")
        print(f"  Non-zero voxels: {np.sum(np.any(features != 0, axis=0)):,}")

        # Apply z-score normalization if requested
        norm_stats = None
        if self.normalize:
            # Compute mean and std across time (per voxel)
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)

            # Avoid division by zero for constant voxels
            std = np.where(std == 0, 1.0, std)

            # Store normalization statistics (per-participant)
            # These will be saved for applying to validation/test data
            norm_stats = {
                'mean': mean.squeeze(),  # (n_voxels,)
                'std': std.squeeze(),    # (n_voxels,)
                'subject_id': nii_path.stem  # e.g., 'sherlock_movie_s1'
            }

            # Z-score normalization: (x - mean) / std
            features = (features - mean) / std

            print(f"  Applied z-score normalization (per-voxel)")
            print(f"  Normalized mean: {np.mean(features):.6f} (should be ~0)")
            print(f"  Normalized std: {np.std(features):.6f} (should be ~1)")

        return features, coordinates, metadata, norm_stats

    def features_to_nii(
        self,
        features: np.ndarray,
        coordinates: np.ndarray,
        output_path: Union[str, Path],
        template_img: Optional[nib.Nifti1Image] = None,
        template_shape: Optional[Tuple[int, int, int]] = None,
    ) -> nib.Nifti1Image:
        """
        Reconstruct NIfTI file from feature matrix.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_voxels) timeseries matrix
        coordinates : np.ndarray
            Shape (n_voxels, 3) array of (x, y, z) coordinates
        output_path : str or Path
            Path for output NIfTI file
        template_img : nib.Nifti1Image, optional
            Template NIfTI image for affine and header info
        template_shape : tuple of int, optional
            3D shape (x, y, z) of original volume. If None, infers from coordinates.

        Returns
        -------
        img : nib.Nifti1Image
            Reconstructed NIfTI image

        Notes
        -----
        - Reconstructs full 4D volume from masked voxels
        - Non-brain voxels are set to zero
        - Uses template affine and header if provided
        - Uses template_shape if provided, otherwise infers from max coordinates
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trs, n_voxels = features.shape

        # Get template info
        if template_img is not None:
            affine = template_img.affine
            header = template_img.header.copy()
            if template_shape is None:
                template_shape = template_img.shape[:3]
        elif self._template_affine is not None:
            affine = self._template_affine
            header = self._template_header.copy()
            if template_shape is None and self._shared_mask is not None:
                template_shape = self._shared_mask.shape
        elif self._shared_mask_img is not None:
            affine = self._shared_mask_img.affine
            header = self._shared_mask_img.header.copy()
            if template_shape is None:
                template_shape = self._shared_mask_img.shape[:3]
        else:
            raise RuntimeError("No template image or affine available")

        # Determine volume shape
        if template_shape is not None:
            vol_shape = template_shape
        else:
            # Fallback: infer from coordinates (may result in cropped volume)
            max_coords = np.max(coordinates, axis=0) + 1
            vol_shape = tuple(max_coords.astype(int))

        # Reconstruct 4D volume
        data = np.zeros((*vol_shape, n_trs), dtype=np.float32)

        # Fill in brain voxels
        for t in range(n_trs):
            for i, (x, y, z) in enumerate(coordinates):
                x, y, z = int(x), int(y), int(z)
                if x < vol_shape[0] and y < vol_shape[1] and z < vol_shape[2]:
                    data[x, y, z, t] = features[t, i]

        # Create NIfTI image
        img = nib.Nifti1Image(data, affine, header)

        # Save
        print(f"Saving reconstructed NIfTI to {output_path}")
        nib.save(img, str(output_path))

        return img

    def average_across_subjects(
        self, subject_features_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Average timeseries across multiple subjects.

        Parameters
        ----------
        subject_features_list : list of np.ndarray
            List of feature matrices, each shape (n_trs, n_voxels)

        Returns
        -------
        averaged_features : np.ndarray
            Shape (n_trs, n_voxels) averaged timeseries

        Notes
        -----
        - All subjects must have same number of voxels
        - Truncates to minimum number of TRs across subjects
        - Simple arithmetic mean across subjects
        """
        # Validate inputs
        if len(subject_features_list) == 0:
            raise ValueError("No subject features provided")

        n_voxels_list = [f.shape[1] for f in subject_features_list]
        if len(set(n_voxels_list)) > 1:
            raise ValueError(f"Inconsistent number of voxels: {set(n_voxels_list)}")

        # Find minimum TRs
        min_trs = min(f.shape[0] for f in subject_features_list)

        # Truncate all subjects to same length
        truncated = [f[:min_trs] for f in subject_features_list]

        # Stack and average
        stacked = np.stack(truncated, axis=0)  # (n_subjects, n_trs, n_voxels)
        averaged = np.mean(stacked, axis=0)  # (n_trs, n_voxels)

        print(f"Averaged {len(subject_features_list)} subjects")
        print(f"  Shape: {averaged.shape}")
        print(f"  TRs used: {min_trs}")

        return averaged

    def load_all_subjects(
        self, data_dir: Union[str, Path], pattern: str = "sherlock_movie_s*.nii.gz"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.DataFrame], List[str]]:
        """
        Load all subject fMRI data from a directory.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing NIfTI files
        pattern : str, default="sherlock_movie_s*.nii.gz"
            Glob pattern for matching subject files

        Returns
        -------
        features_list : list of np.ndarray
            List of feature matrices, each shape (n_trs, n_voxels)
        coordinates_list : list of np.ndarray
            List of coordinate arrays (should all be identical)
        metadata_list : list of pd.DataFrame
            List of metadata DataFrames
        subject_ids : list of str
            List of subject IDs extracted from filenames

        Notes
        -----
        - Creates shared mask if not already created
        - Processes all subjects in sorted order (numeric, not alphabetic)
        - Extracts subject IDs from filenames
        """
        data_dir = Path(data_dir)
        # Sort numerically by subject ID (e.g., s1, s2, ..., s17)
        # not alphabetically (which would give s1, s10, s11, ..., s2)
        nii_files = sorted(
            data_dir.glob(pattern),
            key=lambda f: int(f.name.split("_")[-1].split(".")[0][1:]),
        )

        if len(nii_files) == 0:
            raise FileNotFoundError(f"No files found matching {pattern} in {data_dir}")

        print(f"Found {len(nii_files)} subjects")

        # Create shared mask if needed
        if self._shared_mask is None:
            self.create_shared_mask(nii_files)

        # Process each subject
        features_list = []
        coordinates_list = []
        metadata_list = []
        subject_ids = []

        for nii_file in tqdm(nii_files, desc="Loading subjects"):
            # Extract subject ID from filename
            # e.g., "sherlock_movie_s1.nii.gz" -> "s1"
            subject_id = nii_file.stem.replace(".nii", "").split("_")[-1]
            subject_ids.append(subject_id)

            # Extract features
            features, coordinates, metadata = self.nii_to_features(nii_file)
            features_list.append(features)
            coordinates_list.append(coordinates)
            metadata_list.append(metadata)

        return features_list, coordinates_list, metadata_list, subject_ids

    def get_mask_info(self) -> Dict:
        """
        Get information about the shared mask.

        Returns
        -------
        info : dict
            Dictionary with mask statistics and properties
        """
        if self._shared_mask is None:
            raise RuntimeError("Shared mask not created yet")

        mask_array = self._shared_mask
        n_voxels = np.sum(mask_array)
        total_voxels = np.prod(mask_array.shape)
        proportion = n_voxels / total_voxels

        return {
            "n_voxels": int(n_voxels),
            "total_voxels": int(total_voxels),
            "proportion_brain": float(proportion),
            "proportion_zero": float(1 - proportion),
            "mask_shape": mask_array.shape,
            "coordinates_shape": self._coordinates.shape
            if self._coordinates is not None
            else None,
        }

    def save_mask(self, output_path: Union[str, Path]):
        """
        Save shared mask to NIfTI file.

        Parameters
        ----------
        output_path : str or Path
            Path for output mask file
        """
        if self._shared_mask_img is None:
            raise RuntimeError("Shared mask not created yet")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nib.save(self._shared_mask_img, str(output_path))
        print(f"Saved mask to {output_path}")

    def load_mask(self, mask_path: Union[str, Path]):
        """
        Load shared mask from NIfTI file.

        Parameters
        ----------
        mask_path : str or Path
            Path to mask file
        """
        mask_path = Path(mask_path)
        print(f"Loading mask from {mask_path}")

        mask_img = nib.load(str(mask_path))
        mask_array = mask_img.get_fdata().astype(bool)

        self._shared_mask = mask_array
        self._shared_mask_img = mask_img
        self._template_affine = mask_img.affine
        self._template_header = mask_img.header

        self._compute_voxel_coordinates()

        n_voxels = np.sum(mask_array)
        print(f"Loaded mask: {n_voxels:,} brain voxels")

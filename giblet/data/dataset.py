"""
PyTorch Dataset for multimodal fMRI data (video, audio, text, fMRI).

This module provides a PyTorch Dataset class that loads and aligns all modalities
(video, audio, text, fMRI) for training the multimodal autoencoder. It handles:
- Loading preprocessed data or processing raw data
- Caching aligned features
- Train/validation splits
- Per-subject and cross-subject modes
- HRF convolution for stimulus features

The dataset returns samples at the TR level (1.5s resolution).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict
import pickle

from .video import VideoProcessor
from .audio import AudioProcessor
from .fmri import FMRIProcessor
from ..alignment.sync import align_all_modalities

# Try to import text processor, use dummy features if unavailable
try:
    from .text import TextProcessor

    TEXT_PROCESSOR_AVAILABLE = True
except Exception as e:
    print(f"Warning: TextProcessor not available ({e}). Will use dummy text features.")
    TEXT_PROCESSOR_AVAILABLE = False
    TextProcessor = None


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal fMRI data (video, audio, text, fMRI).

    This dataset loads all four modalities (video, audio, text, fMRI) aligned
    to a common temporal grid based on fMRI TRs. Each sample corresponds to
    one TR (~1.5 seconds).

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing data files:
        - data_dir/sherlock_nii/*.nii.gz (fMRI data)
        - data_dir/stimuli_*.m4v or stimuli_*.mp4 (video file)
        - data_dir/annotations.xlsx (text annotations)
    subjects : str, int, or list, default='all'
        Which subjects to load:
        - 'all': Load all 17 subjects
        - int: Load single subject by number (1-17)
        - list: Load specific subjects [1, 2, 5, ...]
    split : str, optional
        Data split: 'train', 'val', or None (all data).
        If specified, uses 80/20 train/val split on TRs.
    apply_hrf : bool, default=True
        Whether to convolve stimulus features with HRF.
        If True, stimulus features predict BOLD response.
        If False, uses raw stimulus features.
    mode : str, default='per_subject'
        Dataset mode:
        - 'per_subject': Each sample is (subject, TR) pair
        - 'cross_subject': TRs averaged across subjects
    cache_dir : str or Path, optional
        Directory for caching processed features.
        If None, uses data_dir/cache/
    preprocess : bool, default=True
        Whether to preprocess features on initialization.
        If False, features are loaded on-the-fly (slower).
    tr : float, default=1.5
        fMRI repetition time in seconds.
    max_trs : int, optional
        Maximum number of TRs to load (for debugging).
        If None, loads all available TRs.
    use_encodec : bool, default=True
        Whether to use EnCodec for audio processing.
    encodec_bandwidth : float, default=3.0
        EnCodec bandwidth in kbps.
    encodec_sample_rate : int, default=12000
        EnCodec sample rate in Hz.
    frame_skip : int, default=2
        Frame skip factor for video (memory optimization).
    shared_data : dict, optional
        Shared data cache between train/val splits.
    normalize_fmri : bool, default=True
        Whether to z-score normalize fMRI data per subject.
        Fixes scale mismatch between fMRI and other modalities.

    Attributes
    ----------
    n_subjects : int
        Number of subjects loaded
    n_trs : int
        Number of TRs per subject
    n_samples : int
        Total number of samples (n_subjects × n_trs for per_subject mode)
    feature_dims : dict
        Dictionary with feature dimensions for each modality

    Examples
    --------
    >>> # Load all subjects with HRF convolution
    >>> dataset = MultimodalDataset('data/', subjects='all', apply_hrf=True)
    >>> len(dataset)  # 17 subjects × 920 TRs
    15640
    >>> sample = dataset[0]
    >>> sample['video'].shape
    torch.Size([43200])

    >>> # Load single subject for validation
    >>> val_dataset = MultimodalDataset('data/', subjects=1, split='val')
    >>> len(val_dataset)
    184  # 20% of 920 TRs

    >>> # Cross-subject averaged dataset
    >>> avg_dataset = MultimodalDataset('data/', mode='cross_subject')
    >>> len(avg_dataset)
    920  # Just TRs, averaged across subjects
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        subjects: Union[str, int, List[int]] = "all",
        split: Optional[str] = None,
        apply_hrf: bool = True,
        mode: str = "per_subject",
        cache_dir: Optional[Union[str, Path]] = None,
        preprocess: bool = True,
        tr: float = 1.5,
        max_trs: Optional[int] = None,
        use_encodec: bool = True,
        encodec_bandwidth: float = 3.0,
        encodec_sample_rate: int = 12000,
        frame_skip: int = 2,  # Issue #30: Memory optimization via frame skipping
        shared_data: Optional[Dict] = None,  # For sharing cache between train/val
        normalize_fmri: bool = True,  # Issue #32: Normalize fMRI data to fix scale mismatch
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.apply_hrf = apply_hrf
        self.mode = mode
        self.tr = tr
        self.max_trs = max_trs
        self.use_encodec = use_encodec
        self.encodec_bandwidth = encodec_bandwidth
        self.encodec_sample_rate = encodec_sample_rate
        self.frame_skip = frame_skip  # Issue #30
        self.shared_data = shared_data  # Shared data for memory efficiency
        self.normalize_fmri = normalize_fmri  # Issue #32

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Parse subjects
        if subjects == "all":
            self.subject_ids = list(range(1, 18))  # 1-17
        elif isinstance(subjects, int):
            self.subject_ids = [subjects]
        elif isinstance(subjects, list):
            self.subject_ids = subjects
        else:
            raise ValueError(f"Invalid subjects: {subjects}")

        self.n_subjects = len(self.subject_ids)

        # Initialize processors
        self.video_processor = VideoProcessor(tr=tr, frame_skip=frame_skip)  # Issue #30
        self.audio_processor = AudioProcessor(
            tr=tr,
            use_encodec=use_encodec,
            encodec_bandwidth=encodec_bandwidth,
            sample_rate=encodec_sample_rate,
        )
        if TEXT_PROCESSOR_AVAILABLE:
            self.text_processor = TextProcessor(tr=tr)
        else:
            self.text_processor = None
            print("Warning: Using dummy text features (TextProcessor unavailable)")
        self.fmri_processor = FMRIProcessor(
            tr=tr, max_trs=max_trs, normalize=normalize_fmri
        )

        # Data containers
        self.video_features = None
        self.audio_features = None
        self.text_features = None
        self.fmri_features = None
        self.metadata = None

        # Feature dimensions (will be set during loading)
        self.feature_dims = {}
        self.n_trs = None
        self.n_samples = None

        # MEMORY OPTIMIZATION (Issue #30): Determine target dtype once during init
        # Check bfloat16 support here to avoid CUDA re-initialization in DataLoader workers
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.target_dtype = torch.bfloat16
            else:
                self.target_dtype = torch.float32
        except Exception:
            # If CUDA check fails (e.g., in worker process), default to float32
            self.target_dtype = torch.float32

        # Load or preprocess data
        if self.shared_data is not None:
            # Use shared data (for memory efficiency with train/val splits)
            self._load_from_shared_data(self.shared_data)
        elif preprocess:
            self._load_or_preprocess_data()
        else:
            raise NotImplementedError("On-the-fly loading not yet implemented")

        # Apply train/val split if requested
        if split is not None:
            self._apply_split()

    def _get_cache_path(self) -> Path:
        """Get path for cached features."""
        # Create unique cache filename based on parameters
        subjects_str = (
            "all"
            if len(self.subject_ids) == 17
            else f"s{'_'.join(map(str, self.subject_ids))}"
        )
        hrf_str = "hrf" if self.apply_hrf else "nohrf"
        mode_str = self.mode

        # Include EnCodec parameters in cache name if using EnCodec
        if self.use_encodec:
            audio_str = f"encodec_{self.encodec_sample_rate//1000}khz_{self.encodec_bandwidth}kbps"
        else:
            audio_str = "mel"

        # Include frame_skip in cache name (Issue #30)
        frame_skip_str = f"skip{self.frame_skip}"

        cache_name = f"sherlock_{subjects_str}_{hrf_str}_{mode_str}_{audio_str}_{frame_skip_str}.pkl"
        return self.cache_dir / cache_name

    def _get_encodec_cache_path(self, video_path: Path) -> Path:
        """Get path for cached EnCodec features."""
        encodec_cache_dir = (
            self.cache_dir
            / f"encodec_{self.encodec_sample_rate//1000}khz_{self.encodec_bandwidth}kbps"
        )
        encodec_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{video_path.stem}_encodec.npz"
        return encodec_cache_dir / cache_name

    def _load_or_preprocess_data(self):
        """Load cached data or preprocess from raw files."""
        import torch.distributed as dist

        cache_path = self._get_cache_path()

        # DISTRIBUTED FIX (Issue #30): Synchronize caching across ranks
        # Only rank 0 should preprocess and write cache, others wait
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if is_distributed:
            # Check if cache exists (all ranks check)
            cache_exists = cache_path.exists()

            if not cache_exists and rank == 0:
                # Only rank 0 preprocesses and creates cache
                print("[Rank 0] Preprocessing data from raw files...")
                self._preprocess_data()
                self._save_to_cache(cache_path)
                print(f"[Rank 0] Cache saved to {cache_path}")

            # Barrier: all ranks wait for rank 0 to finish caching
            dist.barrier()

            # PERFORMANCE FIX: Sequential cache loading to prevent memory spikes
            # Loading 58GB cache on all 8 ranks simultaneously causes OOM/timeouts
            # Instead, load one rank at a time with barriers between each
            world_size = dist.get_world_size()

            if cache_path.exists():
                # Each rank loads sequentially to avoid memory pressure
                for loading_rank in range(world_size):
                    if rank == loading_rank:
                        print(f"[Rank {rank}] Loading cached features from {cache_path}")
                        self._load_from_cache(cache_path)
                        print(f"[Rank {rank}] Cache loaded successfully")
                    # Wait for this rank to finish loading before next rank starts
                    dist.barrier()
            else:
                raise FileNotFoundError(
                    f"Cache file not created by rank 0: {cache_path}"
                )
        else:
            # Non-distributed: original logic
            if cache_path.exists():
                print(f"Loading cached features from {cache_path}")
                self._load_from_cache(cache_path)
            else:
                print("Preprocessing data from raw files...")
                self._preprocess_data()
                self._save_to_cache(cache_path)

    def _preprocess_data(self):
        """Preprocess all modalities and align them."""
        print("\nPreprocessing multimodal dataset:")
        print(f"  Subjects: {self.subject_ids}")
        print(f"  Mode: {self.mode}")
        print(f"  Apply HRF: {self.apply_hrf}")

        # Paths to data files
        video_path = self.data_dir / "stimuli_Sherlock.m4v"
        annotations_path = self.data_dir / "annotations.xlsx"
        fmri_dir = self.data_dir / "sherlock_nii"

        # Check files exist
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        if not fmri_dir.exists():
            raise FileNotFoundError(f"fMRI directory not found: {fmri_dir}")

        # 1. Process video (shared across all subjects)
        print("\n1. Processing video...")
        video_features, video_meta = self.video_processor.video_to_features(
            video_path, max_trs=self.max_trs
        )
        print(f"  Video features: {video_features.shape}")

        # 2. Process audio (shared across all subjects)
        print("\n2. Processing audio...")

        # Check for cached EnCodec features
        if self.use_encodec:
            encodec_cache_path = self._get_encodec_cache_path(video_path)
            if encodec_cache_path.exists():
                print(f"  Loading cached EnCodec features from {encodec_cache_path}")
                cached_data = np.load(encodec_cache_path)
                audio_features = cached_data["features"]
                # Recreate metadata DataFrame
                audio_meta = pd.DataFrame(
                    {
                        "tr_index": cached_data["tr_indices"],
                        "start_time": cached_data["start_times"],
                        "end_time": cached_data["end_times"],
                        "n_frames": cached_data["n_frames"],
                        "encoding_mode": ["encodec"] * len(cached_data["tr_indices"]),
                    }
                )
                # Apply max_trs if specified
                if self.max_trs is not None:
                    audio_features = audio_features[: self.max_trs]
                    audio_meta = audio_meta.iloc[: self.max_trs]
                print(f"  Audio features (cached): {audio_features.shape}")
            else:
                print(
                    f"  Computing EnCodec features (will cache to {encodec_cache_path})..."
                )
                audio_features, audio_meta = self.audio_processor.audio_to_features(
                    video_path, max_trs=self.max_trs, from_video=True
                )
                # Save to cache
                print("  Caching EnCodec features...")
                np.savez_compressed(
                    encodec_cache_path,
                    features=audio_features,
                    tr_indices=audio_meta["tr_index"].values,
                    start_times=audio_meta["start_time"].values,
                    end_times=audio_meta["end_time"].values,
                    n_frames=audio_meta["n_frames"].values,
                )
                print(
                    f"  Cached {encodec_cache_path.stat().st_size / 1024 / 1024:.1f} MB"
                )
                print(f"  Audio features: {audio_features.shape}")
        else:
            # Mel spectrogram mode (no caching)
            audio_features, audio_meta = self.audio_processor.audio_to_features(
                video_path, max_trs=self.max_trs, from_video=True
            )
            print(f"  Audio features: {audio_features.shape}")

        # 3. Process text (shared across all subjects)
        print("\n3. Processing text annotations...")
        # Get number of TRs from video
        n_trs_target = video_features.shape[0]

        if self.text_processor is not None:
            text_features, text_meta = self.text_processor.annotations_to_embeddings(
                annotations_path, n_trs=n_trs_target
            )
            print(f"  Text features: {text_features.shape}")
        else:
            # Create dummy text features (1024-dim like BGE embeddings)
            print("  Creating dummy text features (TextProcessor unavailable)...")
            text_features = np.random.randn(n_trs_target, 1024).astype(np.float32)
            # Normalize like real embeddings
            text_features = text_features / np.linalg.norm(
                text_features, axis=1, keepdims=True
            )
            print(f"  Text features (dummy): {text_features.shape}")

        # 4. Process fMRI for each subject
        print(f"\n4. Processing fMRI for {self.n_subjects} subjects...")
        fmri_patterns = [f"sherlock_movie_s{sid}.nii.gz" for sid in self.subject_ids]
        fmri_paths = [fmri_dir / pattern for pattern in fmri_patterns]

        # Check all files exist
        missing = [p for p in fmri_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing fMRI files: {missing}")

        # Create shared mask across subjects
        print("  Creating shared brain mask...")
        mask_array, mask_img = self.fmri_processor.create_shared_mask(fmri_paths)

        # Extract features for each subject
        fmri_features_list = []
        fmri_norm_stats = []  # Store normalization stats per subject
        for sid, fmri_path in zip(self.subject_ids, fmri_paths):
            print(f"  Loading subject {sid}...")
            features, coords, meta, norm_stats = self.fmri_processor.nii_to_features(fmri_path)
            fmri_features_list.append(features)
            if norm_stats is not None:
                fmri_norm_stats.append(norm_stats)

        # Stack into (n_subjects, n_trs, n_voxels) array
        fmri_features_stacked = np.stack(fmri_features_list, axis=0)
        print(f"  fMRI features: {fmri_features_stacked.shape}")

        # Save normalization statistics for validation/test
        if fmri_norm_stats:
            norm_stats_path = self.cache_dir / "fmri_normalization_stats.npz"
            print(f"  Saving normalization stats to {norm_stats_path}")
            np.savez(
                norm_stats_path,
                **{f"subject_{stats['subject_id']}_mean": stats['mean'] for stats in fmri_norm_stats},
                **{f"subject_{stats['subject_id']}_std": stats['std'] for stats in fmri_norm_stats},
                subject_ids=[stats['subject_id'] for stats in fmri_norm_stats]
            )

        # 5. Align all modalities
        print("\n5. Aligning all modalities to common TR grid...")

        if self.mode == "per_subject":
            # Align each subject separately
            aligned_data = []
            for sub_idx in range(self.n_subjects):
                result = align_all_modalities(
                    video_features=video_features,
                    audio_features=audio_features,
                    text_features=text_features,
                    fmri_features=fmri_features_stacked[sub_idx],
                    apply_hrf_conv=self.apply_hrf,
                    tr=self.tr,
                )
                aligned_data.append(result)

            # Stack aligned features
            # Shape: (n_subjects, n_trs, n_features)
            self.n_trs = aligned_data[0]["n_trs"]
            self.video_features = np.stack([d["video"] for d in aligned_data], axis=0)
            self.audio_features = np.stack([d["audio"] for d in aligned_data], axis=0)
            self.text_features = np.stack([d["text"] for d in aligned_data], axis=0)
            self.fmri_features = np.stack([d["fmri"] for d in aligned_data], axis=0)

            self.n_samples = self.n_subjects * self.n_trs

        elif self.mode == "cross_subject":
            # Average fMRI across subjects first
            fmri_avg = np.mean(fmri_features_stacked, axis=0)

            # Align with averaged fMRI
            result = align_all_modalities(
                video_features=video_features,
                audio_features=audio_features,
                text_features=text_features,
                fmri_features=fmri_avg,
                apply_hrf_conv=self.apply_hrf,
                tr=self.tr,
            )

            # Store aligned features (no subject dimension)
            # Shape: (n_trs, n_features)
            self.n_trs = result["n_trs"]
            self.video_features = result["video"][np.newaxis, ...]
            self.audio_features = result["audio"][np.newaxis, ...]
            self.text_features = result["text"][np.newaxis, ...]
            self.fmri_features = result["fmri"][np.newaxis, ...]

            self.n_samples = self.n_trs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Store feature dimensions
        # Note: audio features are now 3D (n_codebooks/n_mels, frames_per_tr)
        if self.audio_features.ndim == 4:
            # New format with 4D: (n_subjects/1, n_trs, n_codebooks/n_mels, frames_per_tr)
            audio_dim = self.audio_features.shape[
                -2:
            ]  # (n_codebooks/n_mels, frames_per_tr)
        elif self.audio_features.ndim == 3:
            # Legacy 3D format for individual subject: (n_trs, n_codebooks/n_mels, frames_per_tr)
            audio_dim = self.audio_features.shape[
                -2:
            ]  # (n_codebooks/n_mels, frames_per_tr)
        else:
            # Old 2D format: (n_subjects/1, n_trs, n_features)
            audio_dim = self.audio_features.shape[-1]

        self.feature_dims = {
            "video": self.video_features.shape[-1],
            "audio": audio_dim,
            "text": self.text_features.shape[-1],
            "fmri": self.fmri_features.shape[-1],
        }

        # Create metadata
        self.metadata = {
            "subject_ids": self.subject_ids,
            "n_subjects": self.n_subjects,
            "n_trs": self.n_trs,
            "n_samples": self.n_samples,
            "feature_dims": self.feature_dims,
            "tr": self.tr,
            "apply_hrf": self.apply_hrf,
            "mode": self.mode,
        }

        print("\nPreprocessing complete:")
        print(f"  N subjects: {self.n_subjects}")
        print(f"  N TRs: {self.n_trs}")
        print(f"  N samples: {self.n_samples}")
        print(f"  Feature dims: {self.feature_dims}")

    def _save_to_cache(self, cache_path: Path):
        """Save processed features to cache."""
        print(f"\nSaving to cache: {cache_path}")

        cache_data = {
            "video_features": self.video_features,
            "audio_features": self.audio_features,
            "text_features": self.text_features,
            "fmri_features": self.fmri_features,
            "metadata": self.metadata,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Cached {cache_path.stat().st_size / 1024 / 1024:.1f} MB")

    def _load_from_cache(self, cache_path: Path):
        """Load processed features from cache."""
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self.video_features = cache_data["video_features"]
        self.audio_features = cache_data["audio_features"]
        self.text_features = cache_data["text_features"]
        self.fmri_features = cache_data["fmri_features"]
        self.metadata = cache_data["metadata"]

        self.n_subjects = self.metadata["n_subjects"]
        self.n_trs = self.metadata["n_trs"]
        self.n_samples = self.metadata["n_samples"]
        self.feature_dims = self.metadata["feature_dims"]

        print(f"  Loaded {self.n_samples} samples")
        print(f"  Feature dims: {self.feature_dims}")

    def _load_from_shared_data(self, shared_data: Dict):
        """Load features from shared data dictionary (avoids duplicate cache loading)."""
        # Directly reference shared arrays (no copy)
        self.video_features = shared_data["video_features"]
        self.audio_features = shared_data["audio_features"]
        self.text_features = shared_data["text_features"]
        self.fmri_features = shared_data["fmri_features"]
        self.metadata = shared_data["metadata"].copy()  # Copy metadata to allow per-split modifications

        self.n_subjects = self.metadata["n_subjects"]
        self.n_trs = self.metadata["n_trs"]
        self.n_samples = self.metadata["n_samples"]
        self.feature_dims = self.metadata["feature_dims"]

        print(f"  Using shared data: {self.n_samples} samples")
        print(f"  Feature dims: {self.feature_dims}")

    def get_data_dict(self) -> Dict:
        """
        Export data as dictionary for sharing between train/val datasets.

        Returns
        -------
        dict
            Dictionary containing feature arrays and metadata.
            Arrays are references (not copies) for memory efficiency.
        """
        return {
            "video_features": self.video_features,
            "audio_features": self.audio_features,
            "text_features": self.text_features,
            "fmri_features": self.fmri_features,
            "metadata": self.metadata,
        }

    def _apply_split(self):
        """Apply train/validation split to the data."""
        if self.split not in ["train", "val"]:
            raise ValueError(f"Invalid split: {self.split}. Use 'train' or 'val'.")

        # Use 80/20 split on TRs
        split_idx = int(0.8 * self.n_trs)

        if self.split == "train":
            tr_start, tr_end = 0, split_idx
        else:  # val
            tr_start, tr_end = split_idx, self.n_trs

        # Slice along TR dimension (axis=1)
        self.video_features = self.video_features[:, tr_start:tr_end, :]
        self.audio_features = self.audio_features[:, tr_start:tr_end, :]
        self.text_features = self.text_features[:, tr_start:tr_end, :]
        self.fmri_features = self.fmri_features[:, tr_start:tr_end, :]

        # Update counts
        self.n_trs = tr_end - tr_start
        if self.mode == "per_subject":
            self.n_samples = self.n_subjects * self.n_trs
        else:
            self.n_samples = self.n_trs

        print(f"\nApplied {self.split} split:")
        print(f"  TRs: {tr_start}-{tr_end} ({self.n_trs} total)")
        print(f"  Samples: {self.n_samples}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index (0 to n_samples-1)

        Returns
        -------
        sample : dict
            Dictionary with keys:
            - 'video': Video features (torch.Tensor, bfloat16)
            - 'audio': Audio features (torch.Tensor, int64 for EnCodec or bfloat16 for mel)
            - 'text': Text features (torch.Tensor, bfloat16)
            - 'fmri': fMRI features (torch.Tensor, bfloat16)
            - 'subject_id': Subject ID (int), only for per_subject mode
            - 'tr_index': TR index (int)

        Notes
        -----
        MEMORY OPTIMIZATION (Issue #30): Returns bfloat16 tensors instead of float32
        to reduce memory usage by ~50%. This matches the model's bfloat16 dtype.
        """
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_samples})")

        # Use target dtype determined during initialization (Issue #30 fix)
        # This avoids CUDA re-initialization in DataLoader worker processes
        target_dtype = self.target_dtype

        if self.mode == "per_subject":
            # Compute subject and TR indices
            subject_idx = idx // self.n_trs
            tr_idx = idx % self.n_trs

            # Convert audio to appropriate dtype (int64 for EnCodec, bfloat16/float32 for mel)
            audio_feat = self.audio_features[subject_idx, tr_idx]
            if audio_feat.dtype in [np.int32, np.int64]:
                # EnCodec discrete codes - keep as int64
                audio_tensor = torch.from_numpy(audio_feat).long()
            else:
                # Mel spectrogram - convert to bfloat16 or float32
                audio_tensor = torch.from_numpy(audio_feat).to(target_dtype)

            sample = {
                "video": torch.from_numpy(self.video_features[subject_idx, tr_idx]).to(
                    target_dtype
                ),
                "audio": audio_tensor,
                "text": torch.from_numpy(self.text_features[subject_idx, tr_idx]).to(
                    target_dtype
                ),
                "fmri": torch.from_numpy(self.fmri_features[subject_idx, tr_idx]).to(
                    target_dtype
                ),
                "subject_id": self.subject_ids[subject_idx],
                "tr_index": tr_idx,
            }

        else:  # cross_subject
            tr_idx = idx

            # Convert audio to appropriate dtype
            audio_feat = self.audio_features[0, tr_idx]
            if audio_feat.dtype in [np.int32, np.int64]:
                # EnCodec discrete codes - keep as int64
                audio_tensor = torch.from_numpy(audio_feat).long()
            else:
                # Mel spectrogram - convert to bfloat16 or float32
                audio_tensor = torch.from_numpy(audio_feat).to(target_dtype)

            sample = {
                "video": torch.from_numpy(self.video_features[0, tr_idx]).to(
                    target_dtype
                ),
                "audio": audio_tensor,
                "text": torch.from_numpy(self.text_features[0, tr_idx]).to(
                    target_dtype
                ),
                "fmri": torch.from_numpy(self.fmri_features[0, tr_idx]).to(
                    target_dtype
                ),
                "tr_index": tr_idx,
            }

        return sample

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get a batch of samples.

        Parameters
        ----------
        indices : list of int
            Sample indices

        Returns
        -------
        batch : dict
            Dictionary with batched tensors
        """
        samples = [self[i] for i in indices]

        # Stack into batch
        batch = {
            "video": torch.stack([s["video"] for s in samples]),
            "audio": torch.stack([s["audio"] for s in samples]),
            "text": torch.stack([s["text"] for s in samples]),
            "fmri": torch.stack([s["fmri"] for s in samples]),
            "tr_index": torch.tensor([s["tr_index"] for s in samples]),
        }

        if self.mode == "per_subject":
            batch["subject_id"] = torch.tensor([s["subject_id"] for s in samples])

        return batch

    def get_subject_data(self, subject_id: int) -> Dict[str, np.ndarray]:
        """
        Get all data for a specific subject.

        Parameters
        ----------
        subject_id : int
            Subject ID (1-17)

        Returns
        -------
        data : dict
            Dictionary with full timeseries for the subject
        """
        if self.mode != "per_subject":
            raise ValueError("get_subject_data only available in per_subject mode")

        if subject_id not in self.subject_ids:
            raise ValueError(f"Subject {subject_id} not in dataset")

        subject_idx = self.subject_ids.index(subject_id)

        return {
            "video": self.video_features[subject_idx],
            "audio": self.audio_features[subject_idx],
            "text": self.text_features[subject_idx],
            "fmri": self.fmri_features[subject_idx],
            "subject_id": subject_id,
            "n_trs": self.n_trs,
        }

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each modality.

        Returns
        -------
        stats : dict
            Dictionary with mean, std, min, max for each modality
        """
        stats = {}

        for modality in ["video", "audio", "text", "fmri"]:
            features = getattr(self, f"{modality}_features")

            stats[modality] = {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "shape": features.shape,
            }

        return stats

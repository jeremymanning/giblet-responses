"""
PyTorch Dataset for Sherlock multimodal fMRI data (without text processor).

This is a temporary version that works without sentence-transformers.
Once sentence-transformers issue is resolved, use dataset.py instead.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from tqdm import tqdm
import pickle

from .video import VideoProcessor
from .audio import AudioProcessor
from .fmri import FMRIProcessor
from ..alignment.sync import align_all_modalities


class SherlockDataset(Dataset):
    """
    PyTorch Dataset for Sherlock multimodal fMRI data.

    NOTE: This version uses dummy text features (random normalized vectors)
    instead of real text embeddings. Once sentence-transformers is working,
    switch to the full dataset.py version.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing data files
    subjects : str, int, or list, default='all'
        Which subjects to load
    split : str, optional
        Data split: 'train', 'val', or None
    apply_hrf : bool, default=True
        Whether to convolve stimulus features with HRF
    mode : str, default='per_subject'
        Dataset mode: 'per_subject' or 'cross_subject'
    cache_dir : str or Path, optional
        Directory for caching processed features
    preprocess : bool, default=True
        Whether to preprocess features on initialization
    tr : float, default=1.5
        fMRI repetition time in seconds
    max_trs : int, optional
        Maximum number of TRs to load
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        subjects: Union[str, int, List[int]] = 'all',
        split: Optional[str] = None,
        apply_hrf: bool = True,
        mode: str = 'per_subject',
        cache_dir: Optional[Union[str, Path]] = None,
        preprocess: bool = True,
        tr: float = 1.5,
        max_trs: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.apply_hrf = apply_hrf
        self.mode = mode
        self.tr = tr
        self.max_trs = max_trs

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = self.data_dir / 'cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Parse subjects
        if subjects == 'all':
            self.subject_ids = list(range(1, 18))  # 1-17
        elif isinstance(subjects, int):
            self.subject_ids = [subjects]
        elif isinstance(subjects, list):
            self.subject_ids = subjects
        else:
            raise ValueError(f"Invalid subjects: {subjects}")

        self.n_subjects = len(self.subject_ids)

        # Initialize processors
        self.video_processor = VideoProcessor(tr=tr)
        self.audio_processor = AudioProcessor(tr=tr)
        self.fmri_processor = FMRIProcessor(tr=tr, max_trs=max_trs)

        # Data containers
        self.video_features = None
        self.audio_features = None
        self.text_features = None
        self.fmri_features = None
        self.metadata = None

        # Feature dimensions
        self.feature_dims = {}
        self.n_trs = None
        self.n_samples = None

        # Load or preprocess data
        if preprocess:
            self._load_or_preprocess_data()
        else:
            raise NotImplementedError("On-the-fly loading not yet implemented")

        # Apply train/val split if requested
        if split is not None:
            self._apply_split()

    def _get_cache_path(self) -> Path:
        """Get path for cached features."""
        subjects_str = 'all' if len(self.subject_ids) == 17 else \
                       f"s{'_'.join(map(str, self.subject_ids))}"
        hrf_str = 'hrf' if self.apply_hrf else 'nohrf'
        mode_str = self.mode
        cache_name = f"sherlock_{subjects_str}_{hrf_str}_{mode_str}_notxt.pkl"
        return self.cache_dir / cache_name

    def _load_or_preprocess_data(self):
        """Load cached data or preprocess from raw files."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Loading cached features from {cache_path}")
            self._load_from_cache(cache_path)
        else:
            print("Preprocessing data from raw files...")
            self._preprocess_data()
            self._save_to_cache(cache_path)

    def _preprocess_data(self):
        """Preprocess all modalities and align them."""
        print(f"\nPreprocessing Sherlock dataset:")
        print(f"  Subjects: {self.subject_ids}")
        print(f"  Mode: {self.mode}")
        print(f"  Apply HRF: {self.apply_hrf}")
        print(f"  Using dummy text features (sentence-transformers unavailable)")

        # Paths to data files
        video_path = self.data_dir / 'stimuli_Sherlock.m4v'
        fmri_dir = self.data_dir / 'sherlock_nii'

        # Check files exist
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not fmri_dir.exists():
            raise FileNotFoundError(f"fMRI directory not found: {fmri_dir}")

        # 1. Process video
        print("\n1. Processing video...")
        video_features, video_meta = self.video_processor.video_to_features(
            video_path, max_trs=self.max_trs
        )
        print(f"  Video features: {video_features.shape}")

        # 2. Process audio
        print("\n2. Processing audio...")
        audio_features, audio_meta = self.audio_processor.audio_to_features(
            video_path, max_trs=self.max_trs, from_video=True
        )
        print(f"  Audio features: {audio_features.shape}")

        # 3. Create dummy text features
        print("\n3. Creating dummy text features...")
        n_trs_target = video_features.shape[0]
        text_features = np.random.randn(n_trs_target, 1024).astype(np.float32)
        # Normalize like real embeddings
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        print(f"  Text features (dummy): {text_features.shape}")

        # 4. Process fMRI
        print(f"\n4. Processing fMRI for {self.n_subjects} subjects...")
        fmri_patterns = [f"sherlock_movie_s{sid}.nii.gz" for sid in self.subject_ids]
        fmri_paths = [fmri_dir / pattern for pattern in fmri_patterns]

        missing = [p for p in fmri_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing fMRI files: {missing}")

        print("  Creating shared brain mask...")
        mask_array, mask_img = self.fmri_processor.create_shared_mask(fmri_paths)

        fmri_features_list = []
        for sid, fmri_path in zip(self.subject_ids, fmri_paths):
            print(f"  Loading subject {sid}...")
            features, coords, meta = self.fmri_processor.nii_to_features(fmri_path)
            fmri_features_list.append(features)

        fmri_features_stacked = np.stack(fmri_features_list, axis=0)
        print(f"  fMRI features: {fmri_features_stacked.shape}")

        # 5. Align all modalities
        print("\n5. Aligning all modalities...")

        if self.mode == 'per_subject':
            aligned_data = []
            for sub_idx in range(self.n_subjects):
                result = align_all_modalities(
                    video_features=video_features,
                    audio_features=audio_features,
                    text_features=text_features,
                    fmri_features=fmri_features_stacked[sub_idx],
                    apply_hrf_conv=self.apply_hrf,
                    tr=self.tr
                )
                aligned_data.append(result)

            self.n_trs = aligned_data[0]['n_trs']
            self.video_features = np.stack([d['video'] for d in aligned_data], axis=0)
            self.audio_features = np.stack([d['audio'] for d in aligned_data], axis=0)
            self.text_features = np.stack([d['text'] for d in aligned_data], axis=0)
            self.fmri_features = np.stack([d['fmri'] for d in aligned_data], axis=0)
            self.n_samples = self.n_subjects * self.n_trs

        elif self.mode == 'cross_subject':
            fmri_avg = np.mean(fmri_features_stacked, axis=0)
            result = align_all_modalities(
                video_features=video_features,
                audio_features=audio_features,
                text_features=text_features,
                fmri_features=fmri_avg,
                apply_hrf_conv=self.apply_hrf,
                tr=self.tr
            )

            self.n_trs = result['n_trs']
            self.video_features = result['video'][np.newaxis, ...]
            self.audio_features = result['audio'][np.newaxis, ...]
            self.text_features = result['text'][np.newaxis, ...]
            self.fmri_features = result['fmri'][np.newaxis, ...]
            self.n_samples = self.n_trs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.feature_dims = {
            'video': self.video_features.shape[-1],
            'audio': self.audio_features.shape[-1],
            'text': self.text_features.shape[-1],
            'fmri': self.fmri_features.shape[-1]
        }

        self.metadata = {
            'subject_ids': self.subject_ids,
            'n_subjects': self.n_subjects,
            'n_trs': self.n_trs,
            'n_samples': self.n_samples,
            'feature_dims': self.feature_dims,
            'tr': self.tr,
            'apply_hrf': self.apply_hrf,
            'mode': self.mode
        }

        print(f"\nPreprocessing complete:")
        print(f"  N subjects: {self.n_subjects}")
        print(f"  N TRs: {self.n_trs}")
        print(f"  N samples: {self.n_samples}")
        print(f"  Feature dims: {self.feature_dims}")

    def _save_to_cache(self, cache_path: Path):
        """Save processed features to cache."""
        print(f"\nSaving to cache: {cache_path}")

        cache_data = {
            'video_features': self.video_features,
            'audio_features': self.audio_features,
            'text_features': self.text_features,
            'fmri_features': self.fmri_features,
            'metadata': self.metadata
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Cached {cache_path.stat().st_size / 1024 / 1024:.1f} MB")

    def _load_from_cache(self, cache_path: Path):
        """Load processed features from cache."""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        self.video_features = cache_data['video_features']
        self.audio_features = cache_data['audio_features']
        self.text_features = cache_data['text_features']
        self.fmri_features = cache_data['fmri_features']
        self.metadata = cache_data['metadata']

        self.n_subjects = self.metadata['n_subjects']
        self.n_trs = self.metadata['n_trs']
        self.n_samples = self.metadata['n_samples']
        self.feature_dims = self.metadata['feature_dims']

        print(f"  Loaded {self.n_samples} samples")
        print(f"  Feature dims: {self.feature_dims}")

    def _apply_split(self):
        """Apply train/validation split."""
        if self.split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {self.split}")

        split_idx = int(0.8 * self.n_trs)

        if self.split == 'train':
            tr_start, tr_end = 0, split_idx
        else:
            tr_start, tr_end = split_idx, self.n_trs

        self.video_features = self.video_features[:, tr_start:tr_end, :]
        self.audio_features = self.audio_features[:, tr_start:tr_end, :]
        self.text_features = self.text_features[:, tr_start:tr_end, :]
        self.fmri_features = self.fmri_features[:, tr_start:tr_end, :]

        self.n_trs = tr_end - tr_start
        if self.mode == 'per_subject':
            self.n_samples = self.n_subjects * self.n_trs
        else:
            self.n_samples = self.n_trs

        print(f"\nApplied {self.split} split:")
        print(f"  TRs: {tr_start}-{tr_end} ({self.n_trs} total)")
        print(f"  Samples: {self.n_samples}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range")

        if self.mode == 'per_subject':
            subject_idx = idx // self.n_trs
            tr_idx = idx % self.n_trs

            sample = {
                'video': torch.from_numpy(self.video_features[subject_idx, tr_idx]).float(),
                'audio': torch.from_numpy(self.audio_features[subject_idx, tr_idx]).float(),
                'text': torch.from_numpy(self.text_features[subject_idx, tr_idx]).float(),
                'fmri': torch.from_numpy(self.fmri_features[subject_idx, tr_idx]).float(),
                'subject_id': self.subject_ids[subject_idx],
                'tr_index': tr_idx
            }

        else:
            tr_idx = idx

            sample = {
                'video': torch.from_numpy(self.video_features[0, tr_idx]).float(),
                'audio': torch.from_numpy(self.audio_features[0, tr_idx]).float(),
                'text': torch.from_numpy(self.text_features[0, tr_idx]).float(),
                'fmri': torch.from_numpy(self.fmri_features[0, tr_idx]).float(),
                'tr_index': tr_idx
            }

        return sample

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        samples = [self[i] for i in indices]

        batch = {
            'video': torch.stack([s['video'] for s in samples]),
            'audio': torch.stack([s['audio'] for s in samples]),
            'text': torch.stack([s['text'] for s in samples]),
            'fmri': torch.stack([s['fmri'] for s in samples]),
            'tr_index': torch.tensor([s['tr_index'] for s in samples])
        }

        if self.mode == 'per_subject':
            batch['subject_id'] = torch.tensor([s['subject_id'] for s in samples])

        return batch

    def get_subject_data(self, subject_id: int) -> Dict[str, np.ndarray]:
        if self.mode != 'per_subject':
            raise ValueError("get_subject_data only available in per_subject mode")

        if subject_id not in self.subject_ids:
            raise ValueError(f"Subject {subject_id} not in dataset")

        subject_idx = self.subject_ids.index(subject_id)

        return {
            'video': self.video_features[subject_idx],
            'audio': self.audio_features[subject_idx],
            'text': self.text_features[subject_idx],
            'fmri': self.fmri_features[subject_idx],
            'subject_id': subject_id,
            'n_trs': self.n_trs
        }

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}

        for modality in ['video', 'audio', 'text', 'fmri']:
            features = getattr(self, f'{modality}_features')

            stats[modality] = {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'shape': features.shape
            }

        return stats

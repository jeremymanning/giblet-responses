"""
Text processing module for multimodal fMRI autoencoder project.

Handles bidirectional conversion between text annotations and embeddings:
- Text → Embeddings: Extract text embeddings aligned to fMRI TRs
- Embeddings → Text: Recover text using nearest-neighbor search

Uses BAAI/bge-large-en-v1.5 model (1024-dim, top MTEB performance).
Default temporal alignment uses TR = 1.5 seconds (configurable).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from tqdm import tqdm

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install to use TextProcessor.")

from sklearn.metrics.pairwise import cosine_similarity


class TextProcessor:
    """
    Process text annotations for multimodal autoencoder training.

    Handles:
    - Loading annotations from Excel files
    - Text embedding using BAAI/bge-large-en-v1.5 (1024-dim)
    - Temporal alignment to fMRI TRs (1.5s bins)
    - Temporal concatenation of embeddings within windows
    - Handling overlapping segments and gaps
    - Text recovery using nearest-neighbor search

    Parameters
    ----------
    model_name : str, default='BAAI/bge-large-en-v1.5'
        Sentence-transformers model for text embedding
    tr : float, default=1.5
        fMRI repetition time in seconds
    aggregation : str, default='mean'
        How to handle overlapping segments: 'mean', 'first', 'last', 'max'
        Used in 'average' mode or for legacy compatibility
    gap_fill : str, default='forward_fill'
        How to handle gaps: 'forward_fill', 'zero', 'interpolate'
    device : str, optional
        Device for model inference ('cuda', 'cpu', or None for auto-detect)
    temporal_mode : str, default='concatenate'
        Temporal aggregation mode: 'concatenate' or 'average'
        - 'concatenate': Stack embeddings from temporal window (Issue #26)
        - 'average': Average embeddings (legacy behavior)
    max_annotations_per_tr : int, default=3
        Maximum annotations to concatenate per TR (used in 'concatenate' mode)
        Output will be padded/truncated to this size
    temporal_window : float, default=1.0
        Temporal window size in TR units (future extensibility)
        Currently uses [t - window*TR, t] for each TR
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-large-en-v1.5',
        tr: float = 1.5,
        aggregation: str = 'mean',
        gap_fill: str = 'forward_fill',
        device: Optional[str] = None,
        temporal_mode: str = 'concatenate',
        max_annotations_per_tr: int = 3,
        temporal_window: float = 1.0
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for TextProcessor. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.tr = tr
        self.aggregation = aggregation
        self.gap_fill = gap_fill
        self.device = device
        self.temporal_mode = temporal_mode  # 'concatenate' or 'average'
        self.max_annotations_per_tr = max_annotations_per_tr
        self.temporal_window = temporal_window  # Window size in TR units
        self.n_features = 1024  # BGE-large-en-v1.5 embedding dimension

        # Calculate effective feature dimension based on mode
        if temporal_mode == 'concatenate':
            self.effective_dim = self.n_features * max_annotations_per_tr
        else:
            self.effective_dim = self.n_features

        # Lazy load model
        self._model = None
        self._annotation_cache = None  # Cache for annotations

    def _load_model(self):
        """Load sentence embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded (embedding dim: {self.n_features})")

    def load_annotations(
        self,
        xlsx_path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Load annotations from Excel file.

        Parameters
        ----------
        xlsx_path : str or Path
            Path to annotations Excel file

        Returns
        -------
        annotations : pd.DataFrame
            Loaded annotations with cleaned column names
        """
        xlsx_path = Path(xlsx_path)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {xlsx_path}")

        # Load Excel file
        df = pd.read_excel(xlsx_path)

        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()

        # Verify required columns exist
        required_cols = ['Start Time (s)', 'End Time (s)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def combine_text_columns(
        self,
        annotations: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        separator: str = '; '
    ) -> pd.Series:
        """
        Combine multiple text columns into single text per segment.

        Parameters
        ----------
        annotations : pd.DataFrame
            Annotations dataframe
        text_columns : list of str, optional
            Columns to combine. If None, uses default set:
            ['Scene Details - A Level', 'Name - All', 'Location']
        separator : str, default='; '
            Separator between column values

        Returns
        -------
        combined_text : pd.Series
            Combined text for each segment
        """
        if text_columns is None:
            # Default columns with rich descriptive information
            text_columns = ['Scene Details - A Level', 'Name - All', 'Location']

        # Verify columns exist
        available_cols = [col for col in text_columns if col in annotations.columns]
        if not available_cols:
            raise ValueError(f"None of the text columns found: {text_columns}")

        # Combine columns, handling NaN values
        combined = annotations[available_cols].fillna('').astype(str)
        combined_text = combined.apply(
            lambda row: separator.join([x for x in row if x]),
            axis=1
        )

        # Remove entries that are empty after combination
        combined_text = combined_text.replace('', np.nan)

        return combined_text

    def align_to_trs(
        self,
        annotations: pd.DataFrame,
        embeddings: np.ndarray,
        n_trs: int
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Align segment embeddings to TR grid.

        Parameters
        ----------
        annotations : pd.DataFrame
            Annotations with Start Time (s) and End Time (s) columns
        embeddings : np.ndarray
            Shape (n_segments, embedding_dim) embeddings for each segment
        n_trs : int
            Number of TRs in the target grid

        Returns
        -------
        tr_embeddings : np.ndarray
            Shape (n_trs, effective_dim) aligned embeddings
            - 'concatenate' mode: (n_trs, n_features * max_annotations_per_tr)
            - 'average' mode: (n_trs, n_features)
        metadata : pd.DataFrame
            Metadata for each TR: tr_index, start_time, end_time,
            n_segments_contributing, segment_indices

        Notes
        -----
        - Handles overlapping segments using self.aggregation strategy
        - Fills gaps using self.gap_fill strategy
        - Each TR corresponds to time window [tr_idx * TR, (tr_idx + 1) * TR)
        - In 'concatenate' mode: stacks embeddings with padding/truncation
        - In 'average' mode: averages embeddings (legacy behavior)
        """
        # Choose alignment mode
        if self.temporal_mode == 'concatenate':
            return self._align_to_trs_concatenate(annotations, embeddings, n_trs)
        else:
            return self._align_to_trs_average(annotations, embeddings, n_trs)

    def _align_to_trs_average(
        self,
        annotations: pd.DataFrame,
        embeddings: np.ndarray,
        n_trs: int
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Legacy averaging mode for backward compatibility."""
        tr_embeddings = np.zeros((n_trs, self.n_features), dtype=np.float32)
        tr_metadata = []

        # Track which segments contribute to each TR
        tr_contributors = [[] for _ in range(n_trs)]

        # Map segments to TRs
        for seg_idx, row in annotations.iterrows():
            start_time = row['Start Time (s)']
            end_time = row['End Time (s)']

            # Find TRs that overlap with this segment
            # TR i covers time [i * TR, (i+1) * TR)
            start_tr = int(np.floor(start_time / self.tr))
            end_tr = int(np.ceil(end_time / self.tr))

            # Clamp to valid TR range
            start_tr = max(0, start_tr)
            end_tr = min(n_trs, end_tr)

            # Add this segment to all overlapping TRs
            for tr_idx in range(start_tr, end_tr):
                tr_contributors[tr_idx].append(seg_idx)

        # Aggregate embeddings for each TR
        for tr_idx in range(n_trs):
            start_time = tr_idx * self.tr
            end_time = start_time + self.tr

            contributing_segments = tr_contributors[tr_idx]

            if len(contributing_segments) > 0:
                # Get embeddings for contributing segments
                contributing_embeddings = embeddings[contributing_segments]

                # Aggregate based on strategy
                if self.aggregation == 'mean':
                    tr_embeddings[tr_idx] = np.mean(contributing_embeddings, axis=0)
                elif self.aggregation == 'first':
                    tr_embeddings[tr_idx] = contributing_embeddings[0]
                elif self.aggregation == 'last':
                    tr_embeddings[tr_idx] = contributing_embeddings[-1]
                elif self.aggregation == 'max':
                    tr_embeddings[tr_idx] = np.max(contributing_embeddings, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")

            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_segments_contributing': len(contributing_segments),
                'segment_indices': contributing_segments
            })

        # Fill gaps
        tr_embeddings = self._fill_gaps(tr_embeddings, tr_metadata)

        metadata_df = pd.DataFrame(tr_metadata)

        return tr_embeddings, metadata_df

    def _align_to_trs_concatenate(
        self,
        annotations: pd.DataFrame,
        embeddings: np.ndarray,
        n_trs: int
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Concatenate embeddings from temporal window [t-TR, t].

        For each TR at time t, includes all annotations that overlap
        with the temporal window [t - temporal_window*TR, t].
        Concatenates up to max_annotations_per_tr embeddings.
        """
        tr_embeddings = np.zeros((n_trs, self.effective_dim), dtype=np.float32)
        tr_metadata = []

        for tr_idx in range(n_trs):
            # Define temporal window for this TR
            tr_end = (tr_idx + 1) * self.tr
            tr_start = tr_end - (self.temporal_window * self.tr)
            tr_start = max(0, tr_start)  # Clamp to non-negative time

            # Find annotations overlapping this temporal window
            overlapping_mask = (
                (annotations['Start Time (s)'] < tr_end) &
                (annotations['End Time (s)'] > tr_start)
            )
            overlapping_indices = annotations[overlapping_mask].index.tolist()

            # Get embeddings for overlapping annotations
            n_overlapping = len(overlapping_indices)

            if n_overlapping > 0:
                # Take up to max_annotations_per_tr embeddings
                selected_indices = overlapping_indices[:self.max_annotations_per_tr]
                selected_embeddings = embeddings[selected_indices]

                # Concatenate embeddings
                concat_embedding = selected_embeddings.flatten()

                # Handle padding/truncation
                if len(concat_embedding) < self.effective_dim:
                    # Pad with zeros
                    padded = np.zeros(self.effective_dim, dtype=np.float32)
                    padded[:len(concat_embedding)] = concat_embedding
                    tr_embeddings[tr_idx] = padded
                elif len(concat_embedding) > self.effective_dim:
                    # Truncate (shouldn't happen if max_annotations_per_tr is correct)
                    tr_embeddings[tr_idx] = concat_embedding[:self.effective_dim]
                else:
                    # Exact match
                    tr_embeddings[tr_idx] = concat_embedding

            # Store metadata
            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': tr_start,
                'end_time': tr_end,
                'n_segments_contributing': n_overlapping,
                'segment_indices': overlapping_indices
            })

        # Fill gaps (for TRs with no annotations)
        tr_embeddings = self._fill_gaps_concatenate(tr_embeddings, tr_metadata)

        metadata_df = pd.DataFrame(tr_metadata)

        return tr_embeddings, metadata_df

    def _fill_gaps(
        self,
        tr_embeddings: np.ndarray,
        tr_metadata: List[Dict]
    ) -> np.ndarray:
        """
        Fill gaps where no segments contributed (average mode).

        Parameters
        ----------
        tr_embeddings : np.ndarray
            Shape (n_trs, embedding_dim)
        tr_metadata : list of dict
            Metadata with n_segments_contributing for each TR

        Returns
        -------
        filled_embeddings : np.ndarray
            Embeddings with gaps filled
        """
        filled = tr_embeddings.copy()
        n_trs = len(tr_metadata)

        # Find gaps (TRs with no contributing segments)
        gaps = [i for i, meta in enumerate(tr_metadata)
                if meta['n_segments_contributing'] == 0]

        if len(gaps) == 0:
            return filled

        if self.gap_fill == 'forward_fill':
            # Forward fill from last valid value
            last_valid = None
            for tr_idx in range(n_trs):
                if tr_metadata[tr_idx]['n_segments_contributing'] > 0:
                    last_valid = filled[tr_idx].copy()
                elif last_valid is not None:
                    filled[tr_idx] = last_valid

        elif self.gap_fill == 'zero':
            # Gaps remain zero (already initialized to zero)
            pass

        elif self.gap_fill == 'interpolate':
            # Linear interpolation between valid values
            for gap_idx in gaps:
                # Find previous and next valid TRs
                prev_idx = gap_idx - 1
                while prev_idx >= 0 and tr_metadata[prev_idx]['n_segments_contributing'] == 0:
                    prev_idx -= 1

                next_idx = gap_idx + 1
                while next_idx < n_trs and tr_metadata[next_idx]['n_segments_contributing'] == 0:
                    next_idx += 1

                if prev_idx >= 0 and next_idx < n_trs:
                    # Interpolate between prev and next
                    alpha = (gap_idx - prev_idx) / (next_idx - prev_idx)
                    filled[gap_idx] = (1 - alpha) * filled[prev_idx] + alpha * filled[next_idx]
                elif prev_idx >= 0:
                    # Only previous valid, forward fill
                    filled[gap_idx] = filled[prev_idx]
                elif next_idx < n_trs:
                    # Only next valid, backward fill
                    filled[gap_idx] = filled[next_idx]

        else:
            raise ValueError(f"Unknown gap_fill strategy: {self.gap_fill}")

        return filled

    def _fill_gaps_concatenate(
        self,
        tr_embeddings: np.ndarray,
        tr_metadata: List[Dict]
    ) -> np.ndarray:
        """
        Fill gaps where no segments contributed (concatenate mode).

        Parameters
        ----------
        tr_embeddings : np.ndarray
            Shape (n_trs, effective_dim)
        tr_metadata : list of dict
            Metadata with n_segments_contributing for each TR

        Returns
        -------
        filled_embeddings : np.ndarray
            Embeddings with gaps filled
        """
        filled = tr_embeddings.copy()
        n_trs = len(tr_metadata)

        # Find gaps (TRs with no contributing segments)
        gaps = [i for i, meta in enumerate(tr_metadata)
                if meta['n_segments_contributing'] == 0]

        if len(gaps) == 0:
            return filled

        if self.gap_fill == 'forward_fill':
            # Forward fill from last valid value
            last_valid = None
            for tr_idx in range(n_trs):
                if tr_metadata[tr_idx]['n_segments_contributing'] > 0:
                    last_valid = filled[tr_idx].copy()
                elif last_valid is not None:
                    filled[tr_idx] = last_valid

        elif self.gap_fill == 'zero':
            # Gaps remain zero (already initialized to zero)
            pass

        elif self.gap_fill == 'interpolate':
            # Linear interpolation between valid values (works for concatenated embeddings too)
            for gap_idx in gaps:
                # Find previous and next valid TRs
                prev_idx = gap_idx - 1
                while prev_idx >= 0 and tr_metadata[prev_idx]['n_segments_contributing'] == 0:
                    prev_idx -= 1

                next_idx = gap_idx + 1
                while next_idx < n_trs and tr_metadata[next_idx]['n_segments_contributing'] == 0:
                    next_idx += 1

                if prev_idx >= 0 and next_idx < n_trs:
                    # Interpolate between prev and next
                    alpha = (gap_idx - prev_idx) / (next_idx - prev_idx)
                    filled[gap_idx] = (1 - alpha) * filled[prev_idx] + alpha * filled[next_idx]
                elif prev_idx >= 0:
                    # Only previous valid, forward fill
                    filled[gap_idx] = filled[prev_idx]
                elif next_idx < n_trs:
                    # Only next valid, backward fill
                    filled[gap_idx] = filled[next_idx]

        else:
            raise ValueError(f"Unknown gap_fill strategy: {self.gap_fill}")

        return filled

    def annotations_to_embeddings(
        self,
        xlsx_path: Union[str, Path],
        n_trs: int,
        text_columns: Optional[List[str]] = None,
        cache_annotations: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert annotations to embeddings aligned to TR grid.

        Parameters
        ----------
        xlsx_path : str or Path
            Path to annotations Excel file
        n_trs : int
            Number of TRs in target grid (typically ~920-950 for Sherlock)
        text_columns : list of str, optional
            Text columns to combine and embed
        cache_annotations : bool, default=True
            Whether to cache annotations for reuse

        Returns
        -------
        embeddings : np.ndarray
            Shape (n_trs, 1024) embedding matrix
        metadata : pd.DataFrame
            TR-level metadata with timing and segment information

        Examples
        --------
        >>> processor = TextProcessor()
        >>> embeddings, metadata = processor.annotations_to_embeddings(
        ...     'data/annotations.xlsx',
        ...     n_trs=950
        ... )
        >>> embeddings.shape
        (950, 1024)
        """
        # Load model
        self._load_model()

        # Load annotations
        if cache_annotations and self._annotation_cache is not None:
            annotations = self._annotation_cache
        else:
            annotations = self.load_annotations(xlsx_path)
            if cache_annotations:
                self._annotation_cache = annotations

        # Combine text columns
        combined_text = self.combine_text_columns(annotations, text_columns)

        # Filter out segments with no text
        valid_mask = combined_text.notna()
        valid_annotations = annotations[valid_mask].copy()
        valid_text = combined_text[valid_mask].values

        print(f"Loaded {len(annotations)} segments, {len(valid_annotations)} with text")

        # Embed text segments
        print("Embedding text segments...")
        segment_embeddings = self._model.encode(
            valid_text,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Align to TR grid
        print("Aligning embeddings to TR grid...")
        tr_embeddings, metadata = self.align_to_trs(
            valid_annotations,
            segment_embeddings,
            n_trs
        )

        # Store segment embeddings and text for later retrieval
        metadata['segment_embeddings'] = [segment_embeddings] * len(metadata)
        metadata['segment_texts'] = [valid_text] * len(metadata)

        return tr_embeddings, metadata

    def embeddings_to_text(
        self,
        embeddings: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        method: str = 'nearest_neighbor',
        top_k: int = 1
    ) -> Union[List[str], List[List[str]]]:
        """
        Recover text from embeddings using nearest-neighbor search.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_trs, embedding_dim) embeddings to decode
        metadata : pd.DataFrame, optional
            Metadata from annotations_to_embeddings with segment_embeddings
            and segment_texts. If None, uses cached annotations.
        method : str, default='nearest_neighbor'
            Decoding method: 'nearest_neighbor'
        top_k : int, default=1
            Number of nearest neighbors to return

        Returns
        -------
        texts : list of str or list of list of str
            If top_k=1: List of recovered text for each TR
            If top_k>1: List of top-k texts for each TR

        Examples
        --------
        >>> processor = TextProcessor()
        >>> embeddings, metadata = processor.annotations_to_embeddings(
        ...     'data/annotations.xlsx', n_trs=950
        ... )
        >>> texts = processor.embeddings_to_text(embeddings, metadata)
        >>> len(texts)
        950
        """
        if metadata is None:
            raise ValueError(
                "metadata required for text recovery. "
                "Provide metadata from annotations_to_embeddings()"
            )

        if 'segment_embeddings' not in metadata.columns:
            raise ValueError(
                "metadata missing segment_embeddings. "
                "Ensure metadata is from annotations_to_embeddings()"
            )

        # Extract reference embeddings and texts
        segment_embeddings = metadata['segment_embeddings'].iloc[0]
        segment_texts = metadata['segment_texts'].iloc[0]

        if method == 'nearest_neighbor':
            # Compute cosine similarity between each TR and all segments
            similarities = cosine_similarity(embeddings, segment_embeddings)

            # Find top-k nearest neighbors
            if top_k == 1:
                nearest_indices = np.argmax(similarities, axis=1)
                texts = [segment_texts[idx] for idx in nearest_indices]
            else:
                # Get top-k for each TR
                top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
                texts = [
                    [segment_texts[idx] for idx in tr_indices]
                    for tr_indices in top_indices
                ]

            return texts

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_embedding_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns
        -------
        info : dict
            Model name, embedding dimension, device, temporal mode, etc.
        """
        self._load_model()

        return {
            'model_name': self.model_name,
            'embedding_dim': self.n_features,
            'effective_dim': self.effective_dim,
            'device': str(self._model.device),
            'tr': self.tr,
            'temporal_mode': self.temporal_mode,
            'max_annotations_per_tr': self.max_annotations_per_tr,
            'temporal_window': self.temporal_window,
            'aggregation': self.aggregation,
            'gap_fill': self.gap_fill
        }

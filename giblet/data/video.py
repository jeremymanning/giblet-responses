"""
Video processing module for multimodal fMRI autoencoder project.

Handles bidirectional conversion between video files and feature matrices:
- Video → Features: Extract, downsample, and concatenate temporal windows aligned to fMRI TRs
- Features → Video: Reconstruct video from concatenated feature matrices

Temporal alignment:
- Each TR contains concatenated frames from [t-TR, t]
- Default TR = 1.5 seconds @ 25fps = ~37 frames per window
- All TRs have consistent dimensions (frames concatenated as flat vectors)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
from tqdm import tqdm


class VideoProcessor:
    """
    Process video files for multimodal autoencoder training.

    Handles:
    - Frame extraction at native FPS
    - Spatial downsampling (640×360 → 160×90)
    - Temporal concatenation: each TR contains frames from [t-TR, t]
    - Normalization to [0, 1] range
    - Video reconstruction from concatenated features

    Temporal Concatenation:
    - Each TR contains all frames from the preceding TR window
    - For TR=1.5s at 25fps: ~37 frames concatenated into flat vector
    - Ensures consistent dimensions across all TRs via padding if needed

    Parameters
    ----------
    target_height : int, default=90
        Target frame height after downsampling
    target_width : int, default=160
        Target frame width after downsampling
    tr : float, default=1.5
        fMRI repetition time in seconds
    normalize : bool, default=True
        Whether to normalize pixel values to [0, 1]
    frame_skip : int, default=2
        Sample every Nth frame (2 = half framerate, 3 = third framerate, etc.)
        Memory optimization: frame_skip=2 reduces model size by ~50%
    """

    def __init__(
        self,
        target_height: int = 90,
        target_width: int = 160,
        tr: float = 1.5,
        normalize: bool = True,
        frame_skip: int = 2
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.tr = tr
        self.normalize = normalize
        self.frame_skip = frame_skip  # MEMORY OPTIMIZATION (Issue #30): Skip frames to reduce model size
        self.frame_features = target_height * target_width * 3  # RGB channels per frame

    def video_to_features(
        self,
        video_path: Union[str, Path],
        max_trs: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert video file to feature matrix with temporal concatenation aligned to fMRI TRs.

        Each TR contains concatenated frames from temporal window [t-TR, t].

        Parameters
        ----------
        video_path : str or Path
            Path to video file
        max_trs : int, optional
            Maximum number of TRs to extract (truncates if needed)

        Returns
        -------
        features : np.ndarray
            Shape (n_trs, n_features) where n_features = frames_per_tr * height * width * 3
            Features are concatenated flattened RGB frames from temporal window
        metadata : pd.DataFrame
            DataFrame with columns: tr_index, start_time, end_time, n_frames_concatenated,
            frames_per_tr

        Notes
        -----
        - Concatenates all frames within each TR window [t-TR, t]
        - For TR=1.5s @ 25fps: ~37 frames concatenated per TR
        - First TR may be padded with zeros (no previous frames available)
        - Last TR may be padded if incomplete
        - All TRs guaranteed to have consistent dimensions via padding
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Calculate frames per TR window (before skipping)
        frames_per_tr_original = int(np.round(fps * self.tr))

        # MEMORY OPTIMIZATION (Issue #30): Reduce frames by skipping
        # Example: frame_skip=2 means take every 2nd frame (half the frames)
        frames_per_tr = int(np.ceil(frames_per_tr_original / self.frame_skip))

        # Calculate number of TRs
        n_trs = int(np.floor(duration / self.tr))
        if max_trs is not None:
            n_trs = min(n_trs, max_trs)

        # Pre-allocate feature matrix
        # Each TR contains frames_per_tr concatenated frames (after skipping)
        n_features_per_tr = frames_per_tr * self.frame_features
        features = np.zeros((n_trs, n_features_per_tr), dtype=np.float32)

        # Metadata for each TR
        tr_metadata = []

        # Process each TR
        for tr_idx in tqdm(range(n_trs), desc="Processing video"):
            # Time window for this TR: [t-TR, t]
            end_time = (tr_idx + 1) * self.tr
            start_time = end_time - self.tr

            # Frame indices for this TR window (original range before skipping)
            end_frame = int(end_time * fps)
            start_frame = end_frame - frames_per_tr_original

            # Extract frames within this TR window (with skipping for memory optimization)
            tr_frames = []
            for frame_idx in range(start_frame, end_frame, self.frame_skip):  # Skip frames
                if frame_idx < 0:
                    # Before video start - create zero-padded frame
                    zero_frame = np.zeros(
                        (self.target_height, self.target_width, 3),
                        dtype=np.float32
                    )
                    tr_frames.append(zero_frame)
                elif frame_idx >= total_frames:
                    # After video end - create zero-padded frame
                    zero_frame = np.zeros(
                        (self.target_height, self.target_width, 3),
                        dtype=np.float32
                    )
                    tr_frames.append(zero_frame)
                else:
                    # Normal frame extraction
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        # Failed to read frame - use zero padding
                        zero_frame = np.zeros(
                            (self.target_height, self.target_width, 3),
                            dtype=np.float32
                        )
                        tr_frames.append(zero_frame)
                    else:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Downsample
                        frame = cv2.resize(
                            frame,
                            (self.target_width, self.target_height),
                            interpolation=cv2.INTER_AREA
                        )

                        # Convert to float and normalize if requested
                        frame = frame.astype(np.float32)
                        if self.normalize:
                            frame = frame / 255.0

                        tr_frames.append(frame)

            # Ensure we have exactly frames_per_tr frames
            if len(tr_frames) < frames_per_tr:
                # Pad with zeros if needed
                n_padding = frames_per_tr - len(tr_frames)
                for _ in range(n_padding):
                    zero_frame = np.zeros(
                        (self.target_height, self.target_width, 3),
                        dtype=np.float32
                    )
                    tr_frames.append(zero_frame)
            elif len(tr_frames) > frames_per_tr:
                # Truncate if somehow we have too many
                tr_frames = tr_frames[:frames_per_tr]

            # Convert to numpy array: (frames_per_tr, H, W, C)
            tr_frames = np.array(tr_frames, dtype=np.float32)

            # Flatten and concatenate all frames into single vector
            features[tr_idx] = tr_frames.reshape(-1)

            # Store metadata
            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_frames_concatenated': len(tr_frames),
                'frames_per_tr': frames_per_tr
            })

        cap.release()

        metadata_df = pd.DataFrame(tr_metadata)

        return features, metadata_df

    def features_to_video(
        self,
        features: np.ndarray,
        output_path: Union[str, Path],
        fps: float = 25,
        metadata: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Reconstruct video from concatenated temporal feature matrix.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_features) feature matrix with concatenated temporal windows
        output_path : str or Path
            Path for output video file
        fps : float, default=25
            Output video frame rate
        metadata : pd.DataFrame, optional
            Metadata from video_to_features (for accurate reconstruction)

        Notes
        -----
        - Each TR contains concatenated frames from temporal window
        - Extracts the last frame from each window to represent the TR
        - For TR=1.5s at 25fps: extracts frame 37 of 37 as representative frame
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trs = features.shape[0]

        # Calculate frames per TR
        frames_per_tr = int(np.round(self.tr * fps))

        # Denormalize if needed
        if self.normalize:
            features = features * 255.0

        # Reshape features to concatenated frames
        # (n_trs, frames_per_tr, H, W, C)
        frames = features.reshape(n_trs, frames_per_tr, self.target_height, self.target_width, 3)
        frames = np.clip(frames, 0, 255).astype(np.uint8)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.target_width, self.target_height)
        )

        if not out.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")

        # Write frames
        for tr_idx in tqdm(range(n_trs), desc="Writing video"):
            # Write all frames from this TR window
            for frame_idx in range(frames_per_tr):
                frame = frames[tr_idx, frame_idx]
                # Convert RGB back to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        out.release()

    def frame_to_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert single frame to feature vector.

        Parameters
        ----------
        frame : np.ndarray
            Shape (height, width, 3) RGB frame

        Returns
        -------
        features : np.ndarray
            Shape (n_features,) flattened feature vector
        """
        # Downsample
        frame = cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )

        # Normalize if requested
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0

        return frame.flatten()

    def features_to_frame(self, features: np.ndarray) -> np.ndarray:
        """
        Convert feature vector to frame.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_features,) flattened feature vector

        Returns
        -------
        frame : np.ndarray
            Shape (height, width, 3) RGB frame
        """
        # Denormalize if needed
        if self.normalize:
            features = features * 255.0

        # Reshape
        frame = features.reshape(self.target_height, self.target_width, 3)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video file metadata.

        Parameters
        ----------
        video_path : str or Path
            Path to video file

        Returns
        -------
        info : dict
            Dictionary with: fps, width, height, total_frames, duration, n_trs
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        n_trs = int(np.floor(duration / self.tr))

        cap.release()

        return {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration': duration,
            'n_trs': n_trs
        }

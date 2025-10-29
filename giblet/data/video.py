"""
Video processing module for Sherlock fMRI project.

Handles bidirectional conversion between video files and feature matrices:
- Video → Features: Extract, downsample, and align frames to fMRI TRs
- Features → Video: Reconstruct video from feature matrices

All temporal alignment uses TR = 1.5 seconds (fMRI repetition time).
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
    - Temporal aggregation to match fMRI TR (1.5s bins)
    - Normalization to [0, 1] range
    - Video reconstruction from features

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
    """

    def __init__(
        self,
        target_height: int = 90,
        target_width: int = 160,
        tr: float = 1.5,
        normalize: bool = True
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.tr = tr
        self.normalize = normalize
        self.n_features = target_height * target_width * 3  # RGB channels

    def video_to_features(
        self,
        video_path: Union[str, Path],
        max_trs: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert video file to feature matrix aligned to fMRI TRs.

        Parameters
        ----------
        video_path : str or Path
            Path to video file
        max_trs : int, optional
            Maximum number of TRs to extract (truncates if needed)

        Returns
        -------
        features : np.ndarray
            Shape (n_trs, n_features) where n_features = height * width * 3
            Features are flattened RGB frames
        metadata : pd.DataFrame
            DataFrame with columns: tr_index, start_time, end_time, n_frames_aggregated

        Notes
        -----
        - Aggregates multiple video frames into each TR bin
        - Uses average pooling across frames within each TR
        - First TR starts at t=0, subsequent TRs at t=1.5, 3.0, ...
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

        # Calculate number of TRs
        n_trs = int(np.floor(duration / self.tr))
        if max_trs is not None:
            n_trs = min(n_trs, max_trs)

        # Pre-allocate feature matrix
        features = np.zeros((n_trs, self.n_features), dtype=np.float32)

        # Metadata for each TR
        tr_metadata = []

        # Process each TR
        for tr_idx in tqdm(range(n_trs), desc="Processing video"):
            # Time window for this TR
            start_time = tr_idx * self.tr
            end_time = start_time + self.tr

            # Frame indices for this TR
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Extract and aggregate frames within this TR
            tr_frames = []
            for frame_idx in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Downsample
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA
                )

                tr_frames.append(frame)

            if len(tr_frames) == 0:
                # No frames in this TR (shouldn't happen but handle gracefully)
                print(f"Warning: No frames found for TR {tr_idx}")
                continue

            # Average frames within TR
            tr_frame = np.mean(tr_frames, axis=0).astype(np.float32)

            # Normalize if requested
            if self.normalize:
                tr_frame = tr_frame / 255.0

            # Flatten and store
            features[tr_idx] = tr_frame.flatten()

            # Store metadata
            tr_metadata.append({
                'tr_index': tr_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_frames_aggregated': len(tr_frames)
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
        Reconstruct video from feature matrix.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_trs, n_features) feature matrix
        output_path : str or Path
            Path for output video file
        fps : float, default=25
            Output video frame rate
        metadata : pd.DataFrame, optional
            Metadata from video_to_features (for accurate reconstruction)

        Notes
        -----
        - Each TR's features are duplicated to fill the TR duration at target FPS
        - For TR=1.5s at 25fps, each feature frame is duplicated ~37-38 times
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trs = features.shape[0]

        # Denormalize if needed
        if self.normalize:
            features = features * 255.0

        # Reshape features to frames
        frames = features.reshape(n_trs, self.target_height, self.target_width, 3)
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
            # Convert RGB back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frames[tr_idx], cv2.COLOR_RGB2BGR)

            # Duplicate frame to fill TR duration
            frames_per_tr = int(np.round(self.tr * fps))
            for _ in range(frames_per_tr):
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

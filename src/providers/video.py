import cv2
import numpy as np
from typing import Tuple, Optional
from .base import VisionProvider

class VideoProvider(VisionProvider):
    """
    VisionProvider implementation for reading frames from a video file.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        self.frame_id = 0

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        current_id = self.frame_id
        self.frame_id += 1
        return frame, current_id

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_metadata(self) -> dict:
        """Returns video metadata for output synchronization."""
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

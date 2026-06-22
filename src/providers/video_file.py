"""
VideoFileProvider - Temporary provider for video file playback.

This provider reads from a video file on disk, matching the original main.py
behavior. It serves as a bridge implementation until RTSP, PiCamera, and
USBWebcam providers are fully implemented.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import cv2

from src.providers.base import VisionProvider

logger = logging.getLogger(__name__)


class VideoFileProvider(VisionProvider):
    """
    Video file provider for testing and development.

    Reads frames from a video file on disk using OpenCV's VideoCapture.
    Useful for testing the inference pipeline with recorded footage before
    deploying to live camera or RTSP sources.
    """

    def __init__(self, file_path: str, loop: bool = True):
        """
        Initialize video file provider.

        Args:
            file_path: Path to video file (e.g., "test.mp4", "footage.avi")
            loop: If True, restart video when it reaches the end (default: True)

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
        """
        self.file_path = file_path
        self.loop = loop
        self.capture: Optional[cv2.VideoCapture] = None
        self._connected = False

        logger.info(f"Initializing VideoFileProvider with file: {file_path}")

        # Attempt initial connection
        if not self._open_video():
            raise ValueError(f"Failed to open video file: {file_path}")

        logger.info("VideoFileProvider initialized successfully")

    def _open_video(self) -> bool:
        """
        Open the video file and validate it.

        Returns:
            bool: True if video opened successfully, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.file_path)

            if not self.capture.isOpened():
                logger.error(f"Could not open video file: {self.file_path}")
                self._connected = False
                return False

            # Validate by reading properties
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Video opened: {width}x{height} @ {fps:.1f} FPS, "
                f"{frame_count} frames"
            )

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            self._connected = False
            return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve the next frame from the video file.

        If loop mode is enabled and the video reaches its end, it will
        automatically restart from the beginning.

        Returns:
            tuple[bool, np.ndarray]: (success, frame) where frame is BGR image
                                     or None if read failed
        """
        if not self._connected or self.capture is None:
            logger.warning("Attempted to read frame from disconnected provider")
            return False, None

        ret, frame = self.capture.read()

        # Handle end of video
        if not ret:
            if self.loop:
                logger.debug("End of video reached, looping back to start")
                # Reset to beginning
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.capture.read()

                if not ret:
                    logger.error("Failed to read frame after looping")
                    self._connected = False
                    return False, None
            else:
                logger.info("End of video reached (loop disabled)")
                self._connected = False
                return False, None

        return True, frame

    def is_connected(self) -> bool:
        """
        Check if the video file is currently open and readable.

        Returns:
            bool: True if video is open, False otherwise
        """
        return self._connected and self.capture is not None and self.capture.isOpened()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the video file.

        Releases the current capture and attempts to reopen the video file.

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect to video file")

        # Release existing capture
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        # Attempt to reopen
        success = self._open_video()

        if success:
            logger.info("Video file reconnection successful")
        else:
            logger.error("Video file reconnection failed")

        return success

    def release(self) -> None:
        """
        Release the video file and clean up resources.

        This method is idempotent and safe to call multiple times.
        """
        if self.capture is not None:
            logger.info("Releasing video file resources")
            self.capture.release()
            self.capture = None

        self._connected = False
        logger.debug("VideoFileProvider released")

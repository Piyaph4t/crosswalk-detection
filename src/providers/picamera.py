"""
PiCamera provider for CSI-connected camera modules.

This module implements the VisionProvider interface for Raspberry Pi Camera
modules connected via the CSI (Camera Serial Interface) port. It uses OpenCV's
VideoCapture with the V4L2 backend for optimal performance on Raspberry Pi.

CSI connections are stable once established, so this provider implements
fail-fast behavior at startup with minimal runtime error recovery.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .base import VisionProvider

logger = logging.getLogger(__name__)


class PiCameraProvider(VisionProvider):
    """
    Vision provider for Raspberry Pi Camera modules via CSI connection.

    This provider assumes a stable CSI connection and fails fast if the camera
    is not detected at startup. It does not implement reconnection logic since
    CSI cameras rarely disconnect during runtime.

    Attributes:
        resolution: Camera resolution as [width, height]
        framerate: Target frames per second
        _capture: OpenCV VideoCapture instance
        _connected: Connection status flag
    """

    def __init__(self, resolution: list[int] = None, framerate: int = 30):
        """
        Initialize the PiCamera provider.

        Args:
            resolution: Camera resolution [width, height]. Defaults to [640, 480]
            framerate: Target FPS. Defaults to 30

        Raises:
            RuntimeError: If camera cannot be detected or opened
        """
        self.resolution = resolution or [640, 480]
        self.framerate = framerate
        self._capture: Optional[cv2.VideoCapture] = None
        self._connected = False

        logger.info(
            f"Initializing PiCamera provider with resolution {self.resolution} @ {self.framerate} FPS"
        )

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """
        Initialize the camera with V4L2 backend.

        Raises:
            RuntimeError: If camera cannot be opened or configured
        """
        try:
            # Open camera with V4L2 backend (device index 0)
            self._capture = cv2.VideoCapture(0, cv2.CAP_V4L2)

            if not self._capture.isOpened():
                raise RuntimeError(
                    "Failed to open PiCamera on device index 0. "
                    "Ensure camera is properly connected via CSI port and enabled in raspi-config."
                )

            # Configure camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._capture.set(cv2.CAP_PROP_FPS, self.framerate)

            # Verify configuration by reading actual values
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._capture.get(cv2.CAP_PROP_FPS))

            logger.info(
                f"PiCamera configured: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )

            # Test frame capture to ensure camera is working
            success, frame = self._capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    "Camera opened but failed to capture test frame. "
                    "Check camera module connection and permissions."
                )

            self._connected = True
            logger.info("PiCamera provider initialized successfully")

        except Exception as e:
            self._connected = False
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            logger.error(f"Failed to initialize PiCamera: {e}")
            raise RuntimeError(f"PiCamera initialization failed: {e}") from e

    def get_frame(self) -> tuple[bool, np.ndarray]:
        """
        Retrieve the next frame from the PiCamera.

        Returns:
            tuple[bool, np.ndarray]: (success, frame) where success is True if
                frame was captured successfully, and frame is a BGR numpy array
                with shape (H, W, 3) or None on failure
        """
        if not self._connected or self._capture is None:
            logger.error("Cannot get frame: camera not connected")
            return False, None

        try:
            success, frame = self._capture.read()

            if not success or frame is None:
                logger.error(
                    "Failed to read frame from PiCamera. "
                    "CSI cable may have disconnected or camera encountered hardware error."
                )
                self._connected = False
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Exception during frame capture: {e}")
            self._connected = False
            return False, None

    def is_connected(self) -> bool:
        """
        Check if the camera is currently connected.

        Returns:
            bool: True if camera is connected and ready, False otherwise
        """
        return self._connected and self._capture is not None and self._capture.isOpened()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera.

        Note: CSI connections are stable and rarely disconnect during runtime.
        This method is provided for interface compliance but is not expected
        to be used frequently. If the camera disconnects, it typically requires
        physical intervention to reconnect the CSI cable.

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.warning(
            "Reconnect called on PiCamera provider. "
            "CSI disconnections typically require physical cable reconnection."
        )

        try:
            # Release existing resources
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            # Attempt to reinitialize
            self._initialize_camera()
            return self._connected

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def release(self) -> None:
        """
        Release camera resources and clean up.

        This method is idempotent and safe to call multiple times.
        """
        if self._capture is not None:
            logger.info("Releasing PiCamera resources")
            self._capture.release()
            self._capture = None

        self._connected = False

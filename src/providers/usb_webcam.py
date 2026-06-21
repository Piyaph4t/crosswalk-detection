"""
USB Webcam provider for USB-connected camera devices.

This module implements the VisionProvider interface for USB webcams
connected to the Raspberry Pi via USB ports. It uses OpenCV's VideoCapture
for device access and implements fail-fast behavior at startup with runtime
error detection for USB disconnects.

USB connections can be less stable than CSI connections, so this provider
includes error detection and logging for disconnection events.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .base import VisionProvider

logger = logging.getLogger(__name__)


class USBWebcamProvider(VisionProvider):
    """
    Vision provider for USB webcams.

    This provider opens a USB webcam device by index and fails fast if the
    device is not detected at startup. It detects USB disconnections during
    runtime through failed frame reads and logs errors appropriately.

    Attributes:
        device_id: Video device index (typically 0 for first USB camera)
        resolution: Camera resolution as [width, height]
        _capture: OpenCV VideoCapture instance
        _connected: Connection status flag
    """

    def __init__(self, device_id: int = 0, resolution: list[int] = None):
        """
        Initialize the USB webcam provider.

        Args:
            device_id: Video device index. Defaults to 0 (first USB camera)
            resolution: Camera resolution [width, height]. Defaults to [640, 480]

        Raises:
            RuntimeError: If webcam device cannot be detected or opened
        """
        self.device_id = device_id
        self.resolution = resolution or [640, 480]
        self._capture: Optional[cv2.VideoCapture] = None
        self._connected = False

        logger.info(
            f"Initializing USB webcam provider with device {self.device_id}, "
            f"resolution {self.resolution}"
        )

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """
        Initialize the USB webcam device.

        Raises:
            RuntimeError: If device cannot be opened or configured
        """
        try:
            # Open USB webcam by device index
            self._capture = cv2.VideoCapture(self.device_id)

            if not self._capture.isOpened():
                raise RuntimeError(
                    f"Failed to open USB webcam on device index {self.device_id}. "
                    f"Ensure webcam is properly connected and not in use by another application. "
                    f"Check available devices with: ls /dev/video*"
                )

            # Configure camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Verify configuration by reading actual values
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"USB webcam configured: {actual_width}x{actual_height}"
            )

            # Test frame capture to ensure camera is working
            success, frame = self._capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    "Webcam opened but failed to capture test frame. "
                    "Check device permissions and ensure camera is functional."
                )

            self._connected = True
            logger.info(
                f"USB webcam provider initialized successfully on device {self.device_id}"
            )

        except Exception as e:
            self._connected = False
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            logger.error(f"Failed to initialize USB webcam: {e}")
            raise RuntimeError(f"USB webcam initialization failed: {e}") from e

    def get_frame(self) -> tuple[bool, np.ndarray]:
        """
        Retrieve the next frame from the USB webcam.

        Returns:
            tuple[bool, np.ndarray]: (success, frame) where success is True if
                frame was captured successfully, and frame is a BGR numpy array
                with shape (H, W, 3) or None on failure
        """
        if not self._connected or self._capture is None:
            logger.error("Cannot get frame: USB webcam not connected")
            return False, None

        try:
            success, frame = self._capture.read()

            if not success or frame is None:
                logger.error(
                    f"Failed to read frame from USB webcam (device {self.device_id}). "
                    "USB cable may have disconnected or device encountered an error."
                )
                self._connected = False
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Exception during frame capture from USB webcam: {e}")
            self._connected = False
            return False, None

    def is_connected(self) -> bool:
        """
        Check if the USB webcam is currently connected.

        Returns:
            bool: True if webcam is connected and ready, False otherwise
        """
        return self._connected and self._capture is not None and self._capture.isOpened()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the USB webcam.

        USB devices can be disconnected and reconnected during runtime, so this
        method attempts to re-establish the connection. If the USB device was
        physically disconnected, it must be reconnected before this will succeed.

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.warning(
            f"Reconnect called on USB webcam provider (device {self.device_id}). "
            "Attempting to re-establish connection."
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
            logger.error(f"USB webcam reconnection failed: {e}")
            return False

    def release(self) -> None:
        """
        Release webcam resources and clean up.

        This method is idempotent and safe to call multiple times.
        """
        if self._capture is not None:
            logger.info(f"Releasing USB webcam resources (device {self.device_id})")
            self._capture.release()
            self._capture = None

        self._connected = False

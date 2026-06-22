"""
RTSP stream provider for consuming video from network cameras.

This module implements the VisionProvider interface for RTSP streams, enabling
the system to connect to existing CCTV cameras via the RTSP protocol. It includes
robust error handling with exponential backoff reconnection, stale stream detection,
and comprehensive logging for production deployment.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .base import VisionProvider


logger = logging.getLogger(__name__)


class RTSPProvider(VisionProvider):
    """
    Vision provider for RTSP network streams.

    Connects to IP cameras or CCTV systems via RTSP protocol with automatic
    reconnection on network drops, exponential backoff retry logic, and
    stale stream detection.

    Attributes:
        url (str): RTSP stream URL (e.g., rtsp://192.168.1.100:554/stream1)
        reconnect_timeout (int): Base seconds to wait before reconnection retry
        read_timeout (int): Max seconds to wait for frame before considering connection stale
        buffer_size (int): OpenCV frame buffer size (1 = minimal latency)
    """

    # Exponential backoff parameters
    MIN_BACKOFF = 5  # Initial backoff in seconds
    MAX_BACKOFF = 60  # Maximum backoff in seconds
    BACKOFF_MULTIPLIER = 2  # Exponential growth factor

    def __init__(
        self,
        url: str,
        reconnect_timeout: int = 5,
        read_timeout: int = 10,
        buffer_size: int = 1
    ):
        """
        Initialize RTSP provider with connection parameters.

        Args:
            url: RTSP stream URL (must start with 'rtsp://')
            reconnect_timeout: Base seconds between reconnection attempts
            read_timeout: Seconds to wait for frame before considering stream stale
            buffer_size: OpenCV VideoCapture buffer size (lower = less latency)

        Raises:
            ValueError: If URL is invalid or missing rtsp:// scheme
        """
        # Validate URL format
        if not url or not url.strip():
            raise ValueError("RTSP URL cannot be empty")

        if not url.lower().startswith("rtsp://"):
            raise ValueError(
                f"Invalid RTSP URL: '{url}'. URL must start with 'rtsp://'"
            )

        self.url = url.strip()
        self.reconnect_timeout = max(1, reconnect_timeout)
        self.read_timeout = max(1, read_timeout)
        self.buffer_size = max(1, buffer_size)

        # Connection state
        self._capture: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._last_frame_time: Optional[float] = None
        self._reconnect_attempts = 0
        self._current_backoff = self.MIN_BACKOFF

        logger.info(
            f"Initializing RTSPProvider: url={self.url}, "
            f"reconnect_timeout={self.reconnect_timeout}s, "
            f"read_timeout={self.read_timeout}s, "
            f"buffer_size={self.buffer_size}"
        )

        # Attempt initial connection
        if not self._connect():
            raise ConnectionError(
                f"Failed to establish initial connection to RTSP stream: {self.url}"
            )

    def _connect(self) -> bool:
        """
        Internal method to establish connection to RTSP stream.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Attempting to connect to RTSP stream: {self.url}")

            # Release existing capture if present
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            # Create VideoCapture with FFMPEG backend for RTSP
            self._capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            if not self._capture.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.url}")
                self._connected = False
                return False

            # Configure buffer size for minimal latency
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Verify we can actually read a frame
            success, frame = self._capture.read()
            if not success or frame is None:
                logger.error(
                    f"RTSP stream opened but failed to read initial frame: {self.url}"
                )
                self._capture.release()
                self._capture = None
                self._connected = False
                return False

            # Connection successful
            self._connected = True
            self._last_frame_time = time.time()
            self._reconnect_attempts = 0
            self._current_backoff = self.MIN_BACKOFF

            logger.info(
                f"Successfully connected to RTSP stream: {self.url} "
                f"(frame shape: {frame.shape})"
            )
            return True

        except cv2.error as e:
            logger.error(f"OpenCV error connecting to RTSP stream: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to RTSP stream: {type(e).__name__}: {e}"
            )
            self._connected = False
            return False

    def get_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve the next frame from the RTSP stream.

        Implements stale stream detection using read_timeout. If no frame
        is received within the timeout period, the connection is considered
        stale and a reconnection is attempted.

        Returns:
            tuple[bool, Optional[np.ndarray]]: (success, frame) where success
                indicates if frame was retrieved successfully, and frame is
                a BGR numpy array or None on failure
        """
        # Check connection state
        if not self._connected or self._capture is None:
            logger.warning("Attempted to read frame while disconnected")
            return False, None

        # Check for stale stream
        if self._last_frame_time is not None:
            time_since_last_frame = time.time() - self._last_frame_time
            if time_since_last_frame > self.read_timeout:
                logger.warning(
                    f"Stream appears stale (no frames for {time_since_last_frame:.1f}s). "
                    f"Attempting reconnection..."
                )
                self._connected = False
                self.reconnect()
                return False, None

        try:
            # Attempt to read frame
            success, frame = self._capture.read()

            if not success or frame is None:
                logger.warning("Failed to read frame from RTSP stream")
                self._connected = False
                # Don't automatically reconnect here - let caller decide
                return False, None

            # Update last frame timestamp
            self._last_frame_time = time.time()
            return True, frame

        except cv2.error as e:
            logger.error(f"OpenCV error reading frame: {e}")
            self._connected = False
            return False, None
        except Exception as e:
            logger.error(
                f"Unexpected error reading frame: {type(e).__name__}: {e}"
            )
            self._connected = False
            return False, None

    def is_connected(self) -> bool:
        """
        Check if RTSP stream is currently connected and responding.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._connected and self._capture is not None and self._capture.isOpened()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to RTSP stream with exponential backoff.

        Implements exponential backoff strategy:
        - 1st attempt: 5s delay
        - 2nd attempt: 10s delay
        - 3rd attempt: 20s delay
        - 4th+ attempts: 60s delay (capped)

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        self._reconnect_attempts += 1

        # Calculate backoff delay
        if self._reconnect_attempts > 1:
            # Apply exponential backoff
            self._current_backoff = min(
                self._current_backoff * self.BACKOFF_MULTIPLIER,
                self.MAX_BACKOFF
            )

            logger.info(
                f"Reconnection attempt #{self._reconnect_attempts}. "
                f"Waiting {self._current_backoff}s before retry..."
            )
            time.sleep(self._current_backoff)
        else:
            logger.info(f"Reconnection attempt #{self._reconnect_attempts}")

        # Attempt connection
        success = self._connect()

        if success:
            logger.info(
                f"Reconnection successful after {self._reconnect_attempts} attempt(s)"
            )
        else:
            logger.error(
                f"Reconnection attempt #{self._reconnect_attempts} failed"
            )

        return success

    def release(self) -> None:
        """
        Release RTSP stream and clean up resources.

        This method is idempotent and safe to call multiple times.
        """
        if self._capture is not None:
            try:
                self._capture.release()
                logger.info(f"Released RTSP stream: {self.url}")
            except Exception as e:
                logger.warning(f"Error releasing RTSP stream: {e}")
            finally:
                self._capture = None

        self._connected = False
        self._last_frame_time = None

    def __del__(self):
        """Ensure resources are released when object is garbage collected."""
        self.release()

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "connected" if self._connected else "disconnected"
        return (
            f"RTSPProvider(url='{self.url}', status={status}, "
            f"reconnect_attempts={self._reconnect_attempts})"
        )

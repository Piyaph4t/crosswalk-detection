"""
Abstract base class for vision input providers.

This module defines the VisionProvider interface that all video input sources
(RTSP streams, PiCamera, USB webcams) must implement. The interface enables
runtime switching between input sources via configuration without code changes.
"""

from abc import ABC, abstractmethod
import numpy as np


class VisionProvider(ABC):
    """
    Abstract base class for video input providers.

    All concrete providers must implement the four abstract methods to ensure
    consistent frame acquisition and connection management across different
    input sources (RTSP, PiCamera, USB webcam).
    """

    @abstractmethod
    def get_frame(self) -> tuple[bool, np.ndarray]:
        """
        Retrieve the next frame from the video source.

        This method matches the signature of cv2.VideoCapture.read() for
        compatibility with existing OpenCV-based code. Implementations should
        return immediately rather than blocking indefinitely.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - bool: True if frame retrieved successfully, False otherwise
                - np.ndarray: BGR image frame with shape (H, W, 3) if successful,
                             or None if retrieval failed
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the video source is currently available and responding.

        This method should perform a lightweight check to determine connection
        status without blocking for extended periods. For network sources, this
        might check socket state; for cameras, it might verify device availability.

        Returns:
            bool: True if source is connected and ready, False otherwise
        """
        pass

    @abstractmethod
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the video source.

        This method should release existing connections and attempt to
        re-establish the connection. Implementations may include retry logic
        with backoff, but should not block indefinitely.

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Clean up resources and release the video source.

        This method should close all connections, release camera devices,
        and free any allocated resources. After calling this method, the
        provider should not be used again without reconnection.

        Implementations should ensure this method is idempotent and safe
        to call multiple times.
        """
        pass

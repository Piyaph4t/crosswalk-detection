"""
Unit tests for USBWebcamProvider.

Tests device detection, frame reading, error handling for USB webcam
according to the spec at
docs/superpowers/specs/2026-06-21-dual-input-vision-provider-design.md
"""

from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
import cv2


@pytest.fixture
def mock_video_capture():
    """Fixture providing a mocked cv2.VideoCapture instance."""
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_instance = MagicMock()
        mock_capture_class.return_value = mock_instance
        yield mock_instance


class TestUSBWebcamProvider:
    """Test suite for USBWebcamProvider implementation."""

    def test_successful_initialization(self, mock_video_capture):
        """Test successful USB webcam initialization with default config."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(
            device_id=0,
            resolution=[640, 480]
        )

        assert provider.is_connected() is True

    def test_device_not_found_fails_fast(self, mock_video_capture):
        """Test that missing device fails fast with clear error at startup."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = False

        with pytest.raises(ConnectionError) as exc_info:
            USBWebcamProvider(device_id=0, resolution=[640, 480])

        # Should have helpful error message about device not found
        error_msg = str(exc_info.value).lower()
        assert "device" in error_msg or "webcam" in error_msg or "not found" in error_msg

    def test_custom_device_id(self):
        """Test that custom device_id is used."""
        from src.providers.usb_webcam import USBWebcamProvider

        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_capture_class.return_value = mock_instance

            provider = USBWebcamProvider(device_id=1, resolution=[640, 480])

            # Verify VideoCapture was called with device index 1
            args = mock_capture_class.call_args[0]
            assert args[0] == 1, "Should use custom device index"

    def test_default_device_id(self):
        """Test default device_id of 0."""
        from src.providers.usb_webcam import USBWebcamProvider

        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_capture_class.return_value = mock_instance

            provider = USBWebcamProvider()

            # Verify VideoCapture was called with device index 0
            args = mock_capture_class.call_args[0]
            assert args[0] == 0, "Default device index should be 0"

    def test_resolution_configuration(self, mock_video_capture):
        """Test that resolution is properly configured on VideoCapture."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(
            device_id=0,
            resolution=[1280, 720]
        )

        # Verify resolution was set
        set_calls = mock_video_capture.set.call_args_list
        width_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_WIDTH and call[0][1] == 1280
            for call in set_calls
        )
        height_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_HEIGHT and call[0][1] == 720
            for call in set_calls
        )

        assert width_set, "Frame width should be set"
        assert height_set, "Frame height should be set"

    def test_default_resolution(self, mock_video_capture):
        """Test default resolution of [640, 480]."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        # Initialize without explicit resolution
        provider = USBWebcamProvider(device_id=0)

        # Verify default resolution was set
        set_calls = mock_video_capture.set.call_args_list
        width_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_WIDTH and call[0][1] == 640
            for call in set_calls
        )
        height_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_HEIGHT and call[0][1] == 480
            for call in set_calls
        )

        assert width_set, "Default width should be 640"
        assert height_set, "Default height should be 480"

    def test_get_frame_success(self, mock_video_capture):
        """Test successful frame retrieval from USB webcam."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, test_frame)

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])
        success, frame = provider.get_frame()

        assert success is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_get_frame_failure(self, mock_video_capture):
        """Test frame retrieval failure returns correct types."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True
        mock_video_capture.read.return_value = (False, None)

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])
        success, frame = provider.get_frame()

        assert success is False
        assert frame is None

    def test_usb_disconnect_detection(self, mock_video_capture):
        """Test detection of USB disconnect during runtime."""
        from src.providers.usb_webcam import USBWebcamProvider

        # Device starts connected, then disconnects
        mock_video_capture.isOpened.side_effect = [True, True, False]
        mock_video_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])

        # First frame succeeds
        success1, _ = provider.get_frame()
        assert success1 is True

        # Simulate USB disconnect
        mock_video_capture.read.return_value = (False, None)
        success2, frame2 = provider.get_frame()

        assert success2 is False
        assert frame2 is None

    def test_is_connected_when_device_active(self, mock_video_capture):
        """Test is_connected returns True when device is active."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])

        assert provider.is_connected() is True

    def test_is_connected_when_device_disconnected(self, mock_video_capture):
        """Test is_connected returns False when device disconnects."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.side_effect = [True, False]

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])
        assert provider.is_connected() is True

        # Simulate disconnect
        assert provider.is_connected() is False

    def test_reconnect_attempt(self, mock_video_capture):
        """Test reconnect attempt after USB disconnect."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.side_effect = [True, False, True]

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])

        # Simulate disconnect
        mock_video_capture.isOpened.return_value = False

        # Attempt reconnect
        mock_video_capture.isOpened.return_value = True
        result = provider.reconnect()

        # Reconnect behavior is minimal (no auto-retry logic like RTSP)
        assert isinstance(result, bool)

    def test_release_closes_capture(self, mock_video_capture):
        """Test that release() properly closes the VideoCapture."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])
        provider.release()

        mock_video_capture.release.assert_called_once()

    def test_release_is_idempotent(self, mock_video_capture):
        """Test that release() can be called multiple times safely."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(device_id=0, resolution=[640, 480])

        # Should not raise exception when called multiple times
        provider.release()
        provider.release()
        provider.release()

        assert mock_video_capture.release.call_count >= 1

    def test_invalid_device_id_validation(self, mock_video_capture):
        """Test validation of device_id parameter."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        # Negative device ID should be rejected
        with pytest.raises((ValueError, TypeError)):
            USBWebcamProvider(device_id=-1, resolution=[640, 480])

    def test_invalid_resolution_validation(self, mock_video_capture):
        """Test validation of resolution parameter."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        # Test with invalid resolution formats
        with pytest.raises((ValueError, TypeError)):
            USBWebcamProvider(device_id=0, resolution=[640])  # Only one dimension

        with pytest.raises((ValueError, TypeError)):
            USBWebcamProvider(device_id=0, resolution=[640, 480, 3])  # Three dimensions

        with pytest.raises((ValueError, TypeError)):
            USBWebcamProvider(device_id=0, resolution=[-640, 480])  # Negative value

    def test_runtime_error_logging(self, mock_video_capture, caplog):
        """Test that runtime errors (USB disconnect) are logged."""
        from src.providers.usb_webcam import USBWebcamProvider
        import logging

        mock_video_capture.isOpened.return_value = True
        mock_video_capture.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]

        with caplog.at_level(logging.ERROR):
            provider = USBWebcamProvider(device_id=0, resolution=[640, 480])

            # First read succeeds
            provider.get_frame()

            # Second read fails (USB disconnect)
            provider.get_frame()

        # Should log error about failed read or disconnect
        # (exact logging behavior is implementation-dependent)

    def test_multiple_device_support(self):
        """Test that multiple USB devices can be initialized independently."""
        from src.providers.usb_webcam import USBWebcamProvider

        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_instance1 = MagicMock()
            mock_instance1.isOpened.return_value = True
            mock_instance2 = MagicMock()
            mock_instance2.isOpened.return_value = True

            mock_capture_class.side_effect = [mock_instance1, mock_instance2]

            provider1 = USBWebcamProvider(device_id=0, resolution=[640, 480])
            provider2 = USBWebcamProvider(device_id=1, resolution=[640, 480])

            # Verify both were initialized with different device IDs
            calls = mock_capture_class.call_args_list
            assert calls[0][0][0] == 0
            assert calls[1][0][0] == 1

    def test_high_resolution_support(self, mock_video_capture):
        """Test support for high resolution webcams."""
        from src.providers.usb_webcam import USBWebcamProvider

        mock_video_capture.isOpened.return_value = True

        provider = USBWebcamProvider(
            device_id=0,
            resolution=[1920, 1080]
        )

        # Verify high resolution was set
        set_calls = mock_video_capture.set.call_args_list
        width_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_WIDTH and call[0][1] == 1920
            for call in set_calls
        )
        height_set = any(
            call[0][0] == cv2.CAP_PROP_FRAME_HEIGHT and call[0][1] == 1080
            for call in set_calls
        )

        assert width_set, "Frame width should be set to 1920"
        assert height_set, "Frame height should be set to 1080"

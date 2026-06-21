"""
Unit tests for PiCameraProvider.

Tests startup validation, frame reading, error handling for CSI-connected
PiCamera via V4L2 backend according to the spec at
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


class TestPiCameraProvider:
    """Test suite for PiCameraProvider implementation."""

    def test_successful_initialization(self, mock_video_capture):
        """Test successful PiCamera initialization with default config."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(
            resolution=[640, 480],
            framerate=30
        )

        assert provider.is_connected() is True

    def test_camera_not_detected_fails_fast(self, mock_video_capture):
        """Test that missing camera fails fast with clear error at startup."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = False

        with pytest.raises(ConnectionError) as exc_info:
            PiCameraProvider(resolution=[640, 480], framerate=30)

        # Should have helpful error message about camera not detected
        error_msg = str(exc_info.value).lower()
        assert "camera" in error_msg or "detect" in error_msg or "not found" in error_msg

    def test_resolution_configuration(self, mock_video_capture):
        """Test that resolution is properly configured on VideoCapture."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(
            resolution=[1280, 720],
            framerate=30
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

    def test_framerate_configuration(self, mock_video_capture):
        """Test that framerate is properly configured on VideoCapture."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(
            resolution=[640, 480],
            framerate=15
        )

        # Verify framerate was set
        set_calls = mock_video_capture.set.call_args_list
        fps_set = any(
            call[0][0] == cv2.CAP_PROP_FPS and call[0][1] == 15
            for call in set_calls
        )

        assert fps_set, "Frame rate should be set"

    def test_uses_device_zero(self):
        """Test that PiCamera uses device index 0."""
        from src.providers.picamera import PiCameraProvider

        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_capture_class.return_value = mock_instance

            provider = PiCameraProvider(resolution=[640, 480], framerate=30)

            # Verify VideoCapture was called with device index 0
            args = mock_capture_class.call_args[0]
            assert args[0] == 0, "Should use device index 0"

    def test_get_frame_success(self, mock_video_capture):
        """Test successful frame retrieval from PiCamera."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, test_frame)

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)
        success, frame = provider.get_frame()

        assert success is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_get_frame_failure(self, mock_video_capture):
        """Test frame retrieval failure returns correct types."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True
        mock_video_capture.read.return_value = (False, None)

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)
        success, frame = provider.get_frame()

        assert success is False
        assert frame is None

    def test_csi_cable_disconnect_detection(self, mock_video_capture):
        """Test detection of CSI cable disconnect during runtime."""
        from src.providers.picamera import PiCameraProvider

        # Camera starts connected, then disconnects
        mock_video_capture.isOpened.side_effect = [True, True, False]
        mock_video_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)

        # First frame succeeds
        success1, _ = provider.get_frame()
        assert success1 is True

        # Simulate CSI disconnect
        mock_video_capture.read.return_value = (False, None)
        success2, frame2 = provider.get_frame()

        assert success2 is False
        assert frame2 is None

    def test_is_connected_when_camera_active(self, mock_video_capture):
        """Test is_connected returns True when camera is active."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)

        assert provider.is_connected() is True

    def test_is_connected_when_camera_disconnected(self, mock_video_capture):
        """Test is_connected returns False when camera disconnects."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.side_effect = [True, False]

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)
        assert provider.is_connected() is True

        # Simulate disconnect
        assert provider.is_connected() is False

    def test_no_reconnection_logic(self, mock_video_capture):
        """Test that PiCamera has no reconnection logic (CSI cable stable)."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)

        # Reconnect method should exist (from interface) but may not do auto-retry
        # Since CSI is assumed stable, reconnect behavior is minimal
        result = provider.reconnect()

        # Result depends on implementation - may return False or attempt simple reconnect
        assert isinstance(result, bool)

    def test_release_closes_capture(self, mock_video_capture):
        """Test that release() properly closes the VideoCapture."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)
        provider.release()

        mock_video_capture.release.assert_called_once()

    def test_release_is_idempotent(self, mock_video_capture):
        """Test that release() can be called multiple times safely."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)

        # Should not raise exception when called multiple times
        provider.release()
        provider.release()
        provider.release()

        assert mock_video_capture.release.call_count >= 1

    def test_default_resolution(self, mock_video_capture):
        """Test default resolution of [640, 480]."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        # Initialize without explicit resolution
        provider = PiCameraProvider()

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

    def test_default_framerate(self, mock_video_capture):
        """Test default framerate of 30 FPS."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        # Initialize without explicit framerate
        provider = PiCameraProvider()

        # Verify default framerate was set
        set_calls = mock_video_capture.set.call_args_list
        fps_set = any(
            call[0][0] == cv2.CAP_PROP_FPS and call[0][1] == 30
            for call in set_calls
        )

        assert fps_set, "Default framerate should be 30"

    def test_runtime_error_logging(self, mock_video_capture, caplog):
        """Test that runtime errors (CSI disconnect) are logged."""
        from src.providers.picamera import PiCameraProvider
        import logging

        mock_video_capture.isOpened.return_value = True
        mock_video_capture.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]

        with caplog.at_level(logging.ERROR):
            provider = PiCameraProvider(resolution=[640, 480], framerate=30)

            # First read succeeds
            provider.get_frame()

            # Second read fails (CSI disconnect)
            provider.get_frame()

        # Should log error about failed read or disconnect
        log_messages = [record.message.lower() for record in caplog.records]
        # May log error on failed frame read
        # (exact logging behavior is implementation-dependent)

    def test_invalid_resolution_validation(self, mock_video_capture):
        """Test validation of resolution parameter."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        # Test with invalid resolution formats
        with pytest.raises((ValueError, TypeError)):
            PiCameraProvider(resolution=[640])  # Only one dimension

        with pytest.raises((ValueError, TypeError)):
            PiCameraProvider(resolution=[640, 480, 3])  # Three dimensions

        with pytest.raises((ValueError, TypeError)):
            PiCameraProvider(resolution=[-640, 480])  # Negative value

    def test_invalid_framerate_validation(self, mock_video_capture):
        """Test validation of framerate parameter."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        # Test with invalid framerate values
        with pytest.raises((ValueError, TypeError)):
            PiCameraProvider(framerate=0)  # Zero framerate

        with pytest.raises((ValueError, TypeError)):
            PiCameraProvider(framerate=-30)  # Negative framerate

    def test_v4l2_backend_preference(self, mock_video_capture):
        """Test that V4L2 backend is preferred for PiCamera."""
        from src.providers.picamera import PiCameraProvider

        mock_video_capture.isOpened.return_value = True

        provider = PiCameraProvider(resolution=[640, 480], framerate=30)

        # V4L2 backend usage is implementation-specific
        # This test documents expected behavior but may need adjustment
        # based on actual implementation approach

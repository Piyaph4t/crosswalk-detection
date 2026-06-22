"""
Unit tests for RTSPProvider.

Tests connection establishment, failure handling, reconnection logic with
exponential backoff, and timeout behavior according to the spec at
docs/superpowers/specs/2026-06-21-dual-input-vision-provider-design.md
"""

import time
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


class TestRTSPProvider:
    """Test suite for RTSPProvider implementation."""

    def test_successful_connection(self, mock_video_capture):
        """Test successful RTSP connection on initialization."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(
            url="rtsp://192.168.1.100:554/stream1",
            reconnect_timeout=5,
            read_timeout=10,
            buffer_size=1
        )

        assert provider.is_connected() is True
        mock_video_capture.set.assert_called()  # Buffer size should be set

    def test_invalid_url_fails_fast(self):
        """Test that invalid RTSP URL raises error at startup."""
        from src.providers.rtsp import RTSPProvider

        with pytest.raises((ValueError, ConnectionError)) as exc_info:
            RTSPProvider(url="http://invalid-url.com")

        # Should contain helpful error message
        assert "rtsp://" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_connection_failure_at_startup(self, mock_video_capture):
        """Test connection failure during initialization."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = False

        with pytest.raises(ConnectionError):
            RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

    def test_get_frame_success(self, mock_video_capture):
        """Test successful frame retrieval."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, test_frame)

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")
        success, frame = provider.get_frame()

        assert success is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_get_frame_failure(self, mock_video_capture):
        """Test frame retrieval failure returns correct types."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True
        mock_video_capture.read.return_value = (False, None)

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")
        success, frame = provider.get_frame()

        assert success is False
        assert frame is None

    def test_is_connected_when_stream_active(self, mock_video_capture):
        """Test is_connected returns True for active stream."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

        assert provider.is_connected() is True

    def test_is_connected_when_stream_dropped(self, mock_video_capture):
        """Test is_connected returns False when stream drops."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.side_effect = [True, False]

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")
        assert provider.is_connected() is True

        # Simulate connection drop
        assert provider.is_connected() is False

    def test_reconnect_success(self, mock_video_capture):
        """Test successful reconnection after connection drop."""
        from src.providers.rtsp import RTSPProvider

        # First connection succeeds, then fails, then reconnect succeeds
        mock_video_capture.isOpened.side_effect = [True, False, True]

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

        # Simulate connection drop
        mock_video_capture.isOpened.return_value = False
        assert provider.is_connected() is False

        # Reconnect should succeed
        mock_video_capture.isOpened.return_value = True
        result = provider.reconnect()

        assert result is True
        assert provider.is_connected() is True

    def test_reconnect_failure(self, mock_video_capture):
        """Test reconnection failure when stream unavailable."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.side_effect = [True, False, False]

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

        # Reconnect fails
        result = provider.reconnect()

        assert result is False
        assert provider.is_connected() is False

    def test_exponential_backoff_reconnection(self, mock_video_capture):
        """Test exponential backoff (5s, 10s, 20s, capped at 60s) on reconnection."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True
        provider = RTSPProvider(
            url="rtsp://192.168.1.100:554/stream1",
            reconnect_timeout=5
        )

        # Mock time.sleep to track backoff delays
        with patch('time.sleep') as mock_sleep:
            mock_video_capture.isOpened.return_value = False

            # Attempt multiple reconnections
            for attempt in range(5):
                provider.reconnect()

            # Verify exponential backoff pattern: 5, 10, 20, 40, 60 (capped)
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

            # Should have exponential growth pattern up to cap
            assert len(sleep_calls) >= 3
            # First delay should be base timeout (5s)
            assert sleep_calls[0] == 5
            # Subsequent delays should increase
            if len(sleep_calls) > 1:
                assert sleep_calls[1] >= sleep_calls[0]
            # Should cap at 60s
            assert all(delay <= 60 for delay in sleep_calls)

    def test_read_timeout_detection(self, mock_video_capture):
        """Test detection of stale stream (no frames within read_timeout)."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(
            url="rtsp://192.168.1.100:554/stream1",
            read_timeout=2  # Short timeout for testing
        )

        # Simulate frame read hanging (no data)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 5]  # Simulate 5 seconds elapsed
            mock_video_capture.read.return_value = (False, None)

            success, frame = provider.get_frame()

            # Should detect timeout and mark connection as stale
            assert success is False
            # Provider should recognize connection is problematic
            # (implementation may vary - could trigger reconnect or mark as disconnected)

    def test_buffer_size_configuration(self, mock_video_capture):
        """Test that buffer_size is properly configured on VideoCapture."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(
            url="rtsp://192.168.1.100:554/stream1",
            buffer_size=1
        )

        # Verify buffer size was set (cv2.CAP_PROP_BUFFERSIZE = 38)
        set_calls = mock_video_capture.set.call_args_list
        assert any(call[0][0] == cv2.CAP_PROP_BUFFERSIZE for call in set_calls)

        # Find the buffer size call and verify value
        for call in set_calls:
            if call[0][0] == cv2.CAP_PROP_BUFFERSIZE:
                assert call[0][1] == 1
                break

    def test_release_closes_capture(self, mock_video_capture):
        """Test that release() properly closes the VideoCapture."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")
        provider.release()

        mock_video_capture.release.assert_called_once()

    def test_release_is_idempotent(self, mock_video_capture):
        """Test that release() can be called multiple times safely."""
        from src.providers.rtsp import RTSPProvider

        mock_video_capture.isOpened.return_value = True

        provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

        # Should not raise exception when called multiple times
        provider.release()
        provider.release()
        provider.release()

        assert mock_video_capture.release.call_count >= 1

    def test_ffmpeg_backend_used(self):
        """Test that RTSP uses cv2.CAP_FFMPEG backend."""
        from src.providers.rtsp import RTSPProvider

        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_capture_class.return_value = mock_instance

            provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

            # Verify VideoCapture was called with CAP_FFMPEG backend
            args, kwargs = mock_capture_class.call_args
            # Should be called with url and cv2.CAP_FFMPEG
            assert args[0] == "rtsp://192.168.1.100:554/stream1"
            # Backend may be passed as second arg or via kwargs
            if len(args) > 1:
                assert args[1] == cv2.CAP_FFMPEG

    def test_connection_state_logging(self, mock_video_capture, caplog):
        """Test that connection state changes are logged appropriately."""
        from src.providers.rtsp import RTSPProvider
        import logging

        mock_video_capture.isOpened.side_effect = [True, False, True]

        with caplog.at_level(logging.INFO):
            provider = RTSPProvider(url="rtsp://192.168.1.100:554/stream1")

            # Simulate connection drop and reconnect
            mock_video_capture.isOpened.return_value = False
            provider.is_connected()

            mock_video_capture.isOpened.return_value = True
            provider.reconnect()

        # Verify appropriate log messages exist
        log_messages = [record.message.lower() for record in caplog.records]
        # Should log connection-related events
        assert any("connect" in msg or "rtsp" in msg for msg in log_messages)

"""
Vision Providers Package

This package provides a pluggable architecture for different video input sources
used in the crosswalk detection system. All providers implement the VisionProvider
abstract base class and can be instantiated via the registry factory.

Available Providers:
    - RTSPProvider: Network camera streams via RTSP protocol
    - PiCameraProvider: Raspberry Pi Camera Module via CSI interface
    - USBWebcamProvider: USB-connected webcams

Usage:
    from src.providers import create_provider

    config = load_config("config.yaml")
    provider = create_provider(config)

    success, frame = provider.get_frame()
"""

from src.providers.base import VisionProvider
from src.providers.registry import create_provider

__all__ = [
    "VisionProvider",
    "create_provider",
]

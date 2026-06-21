"""
Provider Registry - Factory for creating vision providers based on configuration.

This module provides a factory function to instantiate the appropriate VisionProvider
implementation based on the runtime configuration.
"""

from typing import Dict

from src.providers.base import VisionProvider
from src.providers.video_file import VideoFileProvider
from src.providers.picamera import PiCameraProvider
from src.providers.usb_webcam import USBWebcamProvider

# TODO: Implement this provider
# from src.providers.rtsp import RTSPProvider


def create_provider(config: dict) -> VisionProvider:
    """
    Factory function to create a vision provider based on configuration.

    This function maps the provider type string from the config to the appropriate
    provider class, extracts the relevant configuration section, and instantiates
    the provider with proper error handling.

    Args:
        config: Loaded configuration dictionary containing a 'vision' section
                with 'provider' field and provider-specific settings.

    Returns:
        VisionProvider: Instantiated and initialized vision provider.

    Raises:
        ValueError: If provider type is unknown or configuration is invalid.
        ConnectionError: If provider initialization or connection fails.

    Example:
        config = {
            "vision": {
                "provider": "rtsp",
                "rtsp": {
                    "url": "rtsp://192.168.1.100:554/stream1",
                    "reconnect_timeout": 5
                }
            }
        }
        provider = create_provider(config)
    """
    # Extract vision configuration section
    if "vision" not in config:
        raise ValueError("Configuration missing 'vision' section")

    vision_config = config["vision"]

    # Get provider type
    if "provider" not in vision_config:
        raise ValueError("Configuration missing 'vision.provider' field")

    provider_type = vision_config["provider"]

    # Provider mapping
    provider_map = {
        "video_file": (VideoFileProvider, "video_file"),
        "picamera": (PiCameraProvider, "picamera"),
        "usb_webcam": (USBWebcamProvider, "usb_webcam"),
        # TODO: Add this provider once implemented
        # "rtsp": (RTSPProvider, "rtsp"),
    }

    # Validate provider type
    if provider_type not in provider_map:
        valid_types = ", ".join(provider_map.keys())
        raise ValueError(
            f"Unknown provider type '{provider_type}'. "
            f"Valid options: {valid_types}"
        )

    # Get provider class and config key
    provider_class, config_key = provider_map[provider_type]

    # Extract provider-specific configuration
    if config_key not in vision_config:
        raise ValueError(
            f"Configuration missing 'vision.{config_key}' section "
            f"for provider type '{provider_type}'"
        )

    provider_config = vision_config[config_key]

    # Instantiate provider with error handling
    try:
        provider = provider_class(**provider_config)
        return provider
    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize {provider_type} provider: {str(e)}"
        ) from e

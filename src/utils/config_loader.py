"""Configuration loader with strict validation for vision provider system."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Supported provider types
VALID_PROVIDERS = {"rtsp", "picamera", "usb_webcam"}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to configuration file (default: "config.yaml")

    Returns:
        dict: Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If config validation fails
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)

    # Check file exists
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {config_file.absolute()}"
        )

    # Load YAML
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")

    # Validate structure
    _validate_config(config)

    logger.info(f"Configuration loaded successfully from {config_path}")
    logger.info(f"Selected provider: {config['vision']['provider']}")

    return config


def _validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration structure and values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    # Check top-level sections exist
    if "vision" not in config:
        raise ConfigValidationError(
            "Missing required 'vision' section in configuration"
        )

    if "inference" not in config:
        raise ConfigValidationError(
            "Missing required 'inference' section in configuration"
        )

    vision_config = config["vision"]

    # Validate provider type
    if "provider" not in vision_config:
        raise ConfigValidationError(
            "Missing required 'vision.provider' field in configuration"
        )

    provider_type = vision_config["provider"]

    if provider_type not in VALID_PROVIDERS:
        raise ConfigValidationError(
            f"Invalid provider type: '{provider_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_PROVIDERS))}"
        )

    # Validate provider-specific configuration
    if provider_type == "rtsp":
        _validate_rtsp_config(vision_config.get("rtsp", {}))
    elif provider_type == "picamera":
        _validate_picamera_config(vision_config.get("picamera", {}))
    elif provider_type == "usb_webcam":
        _validate_usb_webcam_config(vision_config.get("usb_webcam", {}))

    # Validate inference configuration
    _validate_inference_config(config["inference"])


def _validate_rtsp_config(rtsp_config: dict[str, Any]) -> None:
    """
    Validate RTSP provider configuration.

    Args:
        rtsp_config: RTSP configuration section

    Raises:
        ConfigValidationError: If validation fails
    """
    if not rtsp_config:
        raise ConfigValidationError(
            "Missing 'vision.rtsp' configuration section for RTSP provider"
        )

    # Validate URL
    if "url" not in rtsp_config:
        raise ConfigValidationError(
            "Missing required 'vision.rtsp.url' field"
        )

    url = rtsp_config["url"]
    if not isinstance(url, str):
        raise ConfigValidationError(
            f"'vision.rtsp.url' must be a string, got {type(url).__name__}"
        )

    if not url.startswith("rtsp://"):
        raise ConfigValidationError(
            f"Invalid RTSP URL: '{url}'. URL must start with 'rtsp://'"
        )

    # Validate optional timeout fields
    if "reconnect_timeout" in rtsp_config:
        _validate_positive_int(
            rtsp_config["reconnect_timeout"],
            "vision.rtsp.reconnect_timeout"
        )

    if "read_timeout" in rtsp_config:
        _validate_positive_int(
            rtsp_config["read_timeout"],
            "vision.rtsp.read_timeout"
        )

    if "buffer_size" in rtsp_config:
        _validate_positive_int(
            rtsp_config["buffer_size"],
            "vision.rtsp.buffer_size"
        )


def _validate_picamera_config(picamera_config: dict[str, Any]) -> None:
    """
    Validate PiCamera provider configuration.

    Args:
        picamera_config: PiCamera configuration section

    Raises:
        ConfigValidationError: If validation fails
    """
    if not picamera_config:
        raise ConfigValidationError(
            "Missing 'vision.picamera' configuration section for PiCamera provider"
        )

    # Validate resolution
    if "resolution" in picamera_config:
        _validate_resolution(
            picamera_config["resolution"],
            "vision.picamera.resolution"
        )

    # Validate framerate
    if "framerate" in picamera_config:
        _validate_positive_int(
            picamera_config["framerate"],
            "vision.picamera.framerate"
        )


def _validate_usb_webcam_config(usb_config: dict[str, Any]) -> None:
    """
    Validate USB webcam provider configuration.

    Args:
        usb_config: USB webcam configuration section

    Raises:
        ConfigValidationError: If validation fails
    """
    if not usb_config:
        raise ConfigValidationError(
            "Missing 'vision.usb_webcam' configuration section for USB webcam provider"
        )

    # Validate device_id
    if "device_id" in usb_config:
        device_id = usb_config["device_id"]
        if not isinstance(device_id, int):
            raise ConfigValidationError(
                f"'vision.usb_webcam.device_id' must be an integer, "
                f"got {type(device_id).__name__}"
            )
        if device_id < 0:
            raise ConfigValidationError(
                f"'vision.usb_webcam.device_id' must be non-negative, got {device_id}"
            )

    # Validate resolution
    if "resolution" in usb_config:
        _validate_resolution(
            usb_config["resolution"],
            "vision.usb_webcam.resolution"
        )


def _validate_inference_config(inference_config: dict[str, Any]) -> None:
    """
    Validate inference configuration.

    Args:
        inference_config: Inference configuration section

    Raises:
        ConfigValidationError: If validation fails
    """
    # Validate model_path
    if "model_path" not in inference_config:
        raise ConfigValidationError(
            "Missing required 'inference.model_path' field"
        )

    model_path = inference_config["model_path"]
    if not isinstance(model_path, str):
        raise ConfigValidationError(
            f"'inference.model_path' must be a string, got {type(model_path).__name__}"
        )

    # Validate confidence_threshold
    if "confidence_threshold" in inference_config:
        threshold = inference_config["confidence_threshold"]
        if not isinstance(threshold, (int, float)):
            raise ConfigValidationError(
                f"'inference.confidence_threshold' must be a number, "
                f"got {type(threshold).__name__}"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ConfigValidationError(
                f"'inference.confidence_threshold' must be between 0.0 and 1.0, "
                f"got {threshold}"
            )

    # Validate input_size
    if "input_size" in inference_config:
        _validate_positive_int(
            inference_config["input_size"],
            "inference.input_size"
        )


def _validate_resolution(resolution: Any, field_name: str) -> None:
    """
    Validate resolution format [width, height].

    Args:
        resolution: Resolution value to validate
        field_name: Name of the field for error messages

    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(resolution, list):
        raise ConfigValidationError(
            f"'{field_name}' must be a list of two integers [width, height], "
            f"got {type(resolution).__name__}"
        )

    if len(resolution) != 2:
        raise ConfigValidationError(
            f"'{field_name}' must contain exactly 2 elements [width, height], "
            f"got {len(resolution)} elements"
        )

    width, height = resolution

    if not isinstance(width, int) or not isinstance(height, int):
        raise ConfigValidationError(
            f"'{field_name}' must contain integers [width, height], "
            f"got [{type(width).__name__}, {type(height).__name__}]"
        )

    if width <= 0 or height <= 0:
        raise ConfigValidationError(
            f"'{field_name}' dimensions must be positive, got [{width}, {height}]"
        )


def _validate_positive_int(value: Any, field_name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ConfigValidationError(
            f"'{field_name}' must be an integer, got {type(value).__name__}"
        )

    if value <= 0:
        raise ConfigValidationError(
            f"'{field_name}' must be positive, got {value}"
        )

"""Configuration system for DART.

This module provides a unified configuration system for all hard-coded parameters
in the dart-mlci codebase. Configuration can be loaded from JSON files,
environment variables, or programmatically.

Example usage:
    >>> config = DARTConfig()  # Use defaults
    >>> config = DARTConfig.from_json("config.json")
    >>> print(config.detection.tolerance)
    60
    >>> print(config.calibration.pixel_size)
    0.065789
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AxisDirection(Enum):
    """Axis direction convention.

    POSITIVE: Standard direction (+X is right, +Y is down in images)
    NEGATIVE: Inverted direction
    """

    POSITIVE = 1
    NEGATIVE = -1


@dataclass
class CoordinateSystemConfig:
    """Configuration for a single coordinate system's axis conventions.

    Attributes:
        x_direction: Direction of the X axis
        y_direction: Direction of the Y axis
        flip_x: Whether to mirror around Y-axis
        flip_y: Whether to mirror around X-axis
    """

    x_direction: AxisDirection = AxisDirection.POSITIVE
    y_direction: AxisDirection = AxisDirection.POSITIVE
    flip_x: bool = False
    flip_y: bool = False


@dataclass
class CoordinatesConfig:
    """Configuration for all coordinate system conventions.

    Key insight (verified from code):
    - Blueprint uses Cartesian convention: +Y points UP
    - Image uses standard image convention: +Y points DOWN
    - This Y-inversion is handled explicitly in offset calculations
    - Any X-mirror from top-down vs bottom-up is handled by affine transform

    Attributes:
        blueprint: Blueprint (design) coordinate system config
        image: Image (camera/pixel) coordinate system config
        stage: Stage (microscope hardware) coordinate system config
        blueprint_to_image_invert_y: Whether Y is inverted from blueprint to image
    """

    # Blueprint: Design coordinates
    # Default: +X right, +Y UP (Cartesian convention)
    blueprint: CoordinateSystemConfig = field(
        default_factory=lambda: CoordinateSystemConfig(
            y_direction=AxisDirection.NEGATIVE  # Y increases upward (opposite of image)
        )
    )

    # Image: Camera/pixel coordinates
    # Default: +X right, +Y DOWN (standard image convention)
    image: CoordinateSystemConfig = field(default_factory=CoordinateSystemConfig)

    # Stage: Microscope stage coordinates (hardware-dependent)
    # Default: +X right, +Y down (but verify with your hardware!)
    stage: CoordinateSystemConfig = field(default_factory=CoordinateSystemConfig)

    # Whether blueprint->image requires Y inversion
    # True because blueprint uses Y-up, image uses Y-down
    blueprint_to_image_invert_y: bool = True


@dataclass
class DetectionConfig:
    """Configuration for marker detection and matching.

    Attributes:
        tolerance: Pixel tolerance for marker matching
        confidence: YOLO confidence threshold
    """

    tolerance: int = 60  # pixels
    confidence: float = 0.6


@dataclass
class PathConfig:
    """Configuration for file paths.

    Attributes:
        model_path: Path to the YOLO detection model
        structure_library_path: Path to chamber structure definitions (deprecated,
            use chip_config_path instead)
        blueprint_map_path: Path to blueprint map CSV (optional, deprecated,
            use chip_config_path instead)
        chip_config_path: Path to unified chip configuration JSON file.
            When set, this takes precedence over structure_library_path and
            blueprint_map_path.
    """

    model_path: Path = field(
        default_factory=lambda: Path("artifacts/models/v26_detect_s_imgsz1280.pt")
    )
    structure_library_path: Path = field(
        default_factory=lambda: Path("artifacts/chamber_structure.json")
    )
    blueprint_map_path: Path | None = None
    chip_config_path: Path | None = None

    def __post_init__(self):
        # Convert strings to Path objects
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.structure_library_path, str):
            self.structure_library_path = Path(self.structure_library_path)
        if isinstance(self.blueprint_map_path, str):
            self.blueprint_map_path = Path(self.blueprint_map_path)
        if isinstance(self.chip_config_path, str):
            self.chip_config_path = Path(self.chip_config_path)


@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters.

    Attributes:
        pixel_size: Pixel size in microns
        min_calibration_points: Minimum number of calibration points required
    """

    pixel_size: float = 0.065789  # microns per pixel
    min_calibration_points: int = 3


@dataclass
class DARTConfig:
    """Main configuration class for DMC masking.

    This class consolidates all configuration parameters used throughout
    the dart-mlci codebase. It can be instantiated with defaults,
    loaded from JSON, or configured via environment variables.

    Attributes:
        detection: Detection and matching configuration
        paths: File path configuration
        calibration: Calibration parameters
        coordinates: Coordinate system conventions

    Example:
        >>> config = DARTConfig()
        >>> config = DARTConfig.from_json("my_config.json")
        >>> config = DARTConfig.from_env()
    """

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    coordinates: CoordinatesConfig = field(default_factory=CoordinatesConfig)

    @classmethod
    def from_json(cls, path: Path | str) -> "DARTConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file

        Returns:
            DARTConfig instance with values from the file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> "DARTConfig":
        """Load configuration from environment variables.

        Environment variable mapping:
            DART_TOLERANCE -> detection.tolerance
            DART_CONFIDENCE -> detection.confidence
            DART_PIXEL_SIZE -> calibration.pixel_size
            DART_MODEL_PATH -> paths.model_path
            DART_STRUCTURE_LIBRARY_PATH -> paths.structure_library_path
            DART_BLUEPRINT_MAP_PATH -> paths.blueprint_map_path

        Returns:
            DARTConfig instance with values from environment variables
        """
        config = cls()

        # Detection config
        if "DART_TOLERANCE" in os.environ:
            config.detection.tolerance = int(os.environ["DART_TOLERANCE"])
        if "DART_CONFIDENCE" in os.environ:
            config.detection.confidence = float(os.environ["DART_CONFIDENCE"])

        # Calibration config
        if "DART_PIXEL_SIZE" in os.environ:
            config.calibration.pixel_size = float(os.environ["DART_PIXEL_SIZE"])

        # Path config
        if "DART_MODEL_PATH" in os.environ:
            config.paths.model_path = Path(os.environ["DART_MODEL_PATH"])
        if "DART_STRUCTURE_LIBRARY_PATH" in os.environ:
            config.paths.structure_library_path = Path(os.environ["DART_STRUCTURE_LIBRARY_PATH"])
        if "DART_BLUEPRINT_MAP_PATH" in os.environ:
            config.paths.blueprint_map_path = Path(os.environ["DART_BLUEPRINT_MAP_PATH"])
        if "DART_CHIP_CONFIG_PATH" in os.environ:
            config.paths.chip_config_path = Path(os.environ["DART_CHIP_CONFIG_PATH"])

        return config

    @classmethod
    def _from_dict(cls, data: dict) -> "DARTConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            DARTConfig instance
        """
        config = cls()

        # Detection config
        if "detection" in data:
            det = data["detection"]
            if "tolerance" in det:
                config.detection.tolerance = int(det["tolerance"])
            if "confidence" in det:
                config.detection.confidence = float(det["confidence"])

        # Calibration config
        if "calibration" in data:
            cal = data["calibration"]
            if "pixel_size" in cal:
                config.calibration.pixel_size = float(cal["pixel_size"])
            if "min_calibration_points" in cal:
                config.calibration.min_calibration_points = int(cal["min_calibration_points"])

        # Path config
        if "paths" in data:
            paths = data["paths"]
            if "model_path" in paths:
                config.paths.model_path = Path(paths["model_path"])
            if "structure_library_path" in paths:
                config.paths.structure_library_path = Path(paths["structure_library_path"])
            if paths.get("blueprint_map_path"):
                config.paths.blueprint_map_path = Path(paths["blueprint_map_path"])
            if paths.get("chip_config_path"):
                config.paths.chip_config_path = Path(paths["chip_config_path"])

        # Coordinates config
        if "coordinates" in data:
            coords = data["coordinates"]
            if "blueprint_to_image_invert_y" in coords:
                config.coordinates.blueprint_to_image_invert_y = bool(
                    coords["blueprint_to_image_invert_y"]
                )
            # Parse axis direction configs if present
            config.coordinates = cls._parse_coordinates_config(coords, config.coordinates)

        return config

    @classmethod
    def _parse_coordinates_config(cls, data: dict, default: CoordinatesConfig) -> CoordinatesConfig:
        """Parse coordinate system configuration from dictionary.

        Args:
            data: Coordinates configuration dictionary
            default: Default configuration to use for missing values

        Returns:
            CoordinatesConfig instance
        """
        config = default

        for system_name in ["blueprint", "image", "stage"]:
            if system_name in data:
                system_data = data[system_name]
                system_config = getattr(config, system_name)

                if "x_direction" in system_data:
                    direction = system_data["x_direction"].lower()
                    system_config.x_direction = (
                        AxisDirection.POSITIVE
                        if direction == "positive"
                        else AxisDirection.NEGATIVE
                    )

                if "y_direction" in system_data:
                    direction = system_data["y_direction"].lower()
                    system_config.y_direction = (
                        AxisDirection.POSITIVE
                        if direction == "positive"
                        else AxisDirection.NEGATIVE
                    )

                if "flip_x" in system_data:
                    system_config.flip_x = bool(system_data["flip_x"])

                if "flip_y" in system_data:
                    system_config.flip_y = bool(system_data["flip_y"])

        if "blueprint_to_image_invert_y" in data:
            config.blueprint_to_image_invert_y = bool(data["blueprint_to_image_invert_y"])

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "detection": {
                "tolerance": self.detection.tolerance,
                "confidence": self.detection.confidence,
            },
            "calibration": {
                "pixel_size": self.calibration.pixel_size,
                "min_calibration_points": self.calibration.min_calibration_points,
            },
            "paths": {
                "model_path": str(self.paths.model_path),
                "structure_library_path": str(self.paths.structure_library_path),
                "blueprint_map_path": (
                    str(self.paths.blueprint_map_path) if self.paths.blueprint_map_path else None
                ),
                "chip_config_path": (
                    str(self.paths.chip_config_path) if self.paths.chip_config_path else None
                ),
            },
            "coordinates": {
                "blueprint": {
                    "x_direction": self.coordinates.blueprint.x_direction.name.lower(),
                    "y_direction": self.coordinates.blueprint.y_direction.name.lower(),
                    "flip_x": self.coordinates.blueprint.flip_x,
                    "flip_y": self.coordinates.blueprint.flip_y,
                },
                "image": {
                    "x_direction": self.coordinates.image.x_direction.name.lower(),
                    "y_direction": self.coordinates.image.y_direction.name.lower(),
                    "flip_x": self.coordinates.image.flip_x,
                    "flip_y": self.coordinates.image.flip_y,
                },
                "stage": {
                    "x_direction": self.coordinates.stage.x_direction.name.lower(),
                    "y_direction": self.coordinates.stage.y_direction.name.lower(),
                    "flip_x": self.coordinates.stage.flip_x,
                    "flip_y": self.coordinates.stage.flip_y,
                },
                "blueprint_to_image_invert_y": self.coordinates.blueprint_to_image_invert_y,
            },
        }

    def to_json(self, path: Path | str) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global default configuration instance
_default_config: DARTConfig | None = None


def get_default_config() -> DARTConfig:
    """Get the global default configuration.

    Returns:
        The default DARTConfig instance (created on first call)
    """
    global _default_config
    if _default_config is None:
        _default_config = DARTConfig()
    return _default_config


def set_default_config(config: DARTConfig) -> None:
    """Set the global default configuration.

    Args:
        config: The configuration to use as default
    """
    global _default_config
    _default_config = config

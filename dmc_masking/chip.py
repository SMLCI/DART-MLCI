"""Unified chip configuration module.

This module provides a single-file configuration system for microfluidic chip
designs, replacing the scattered configuration across chamber_structure.json,
blueprint_map.csv, and hardcoded values in mask.py.

Example usage:
    >>> from dmc_masking.chip import ChipStructureLibrary
    >>> lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json")
    >>> structure_name, polygon, markers = lib("0050")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from .map import Map
from .mask import RoIPolygon


@dataclass
class ChamberTypeConfig:
    """Configuration for a single chamber type.

    Attributes:
        polygon: GeoJSON polygon dict (type + coordinates)
        markers: Dict mapping marker name to [x, y] position in microns
    """

    polygon: dict
    markers: dict[str, list[float]]


@dataclass
class ChipConfig:
    """Complete chip configuration loaded from a JSON file.

    Attributes:
        chip_name: Human-readable chip name (e.g., "SAK")
        version: Config file version string
        description: Optional description of the chip
        pixel_size: Default pixel size in microns per pixel
        chamber_types: Mapping of chamber type name to its configuration
        blueprint_map: List of dicts with roi_id, x, y positions
    """

    chip_name: str
    version: str
    description: str
    pixel_size: float
    chamber_types: dict[str, ChamberTypeConfig]
    blueprint_map: list[dict] = field(default_factory=list)


def load_chip_config(path: Path | str) -> ChipConfig:
    """Load and validate a chip configuration from a JSON file.

    Args:
        path: Path to the chip config JSON file

    Returns:
        ChipConfig instance

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config is missing required fields or is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chip config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate required top-level fields
    required = ["chip_name", "version", "pixel_size", "chamber_types"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Chip config missing required fields: {', '.join(missing)}")

    # Parse chamber types
    chamber_types = {}
    for name, ct_data in data["chamber_types"].items():
        # Validate chamber type fields
        ct_required = ["polygon", "markers"]
        ct_missing = [k for k in ct_required if k not in ct_data]
        if ct_missing:
            raise ValueError(
                f"Chamber type '{name}' missing required fields: {', '.join(ct_missing)}"
            )

        # Validate polygon has GeoJSON structure
        polygon = ct_data["polygon"]
        if "type" not in polygon or "coordinates" not in polygon:
            raise ValueError(
                f"Chamber type '{name}' polygon must have 'type' and 'coordinates' fields"
            )

        # Validate markers
        markers = ct_data["markers"]
        if not isinstance(markers, dict):
            raise ValueError(f"Chamber type '{name}' markers must be a dict")

        chamber_types[name] = ChamberTypeConfig(
            polygon=polygon,
            markers=markers,
        )

    # Parse blueprint map
    blueprint_map = data.get("blueprint_map", [])

    # Validate structure_type references in blueprint_map entries
    for entry in blueprint_map:
        st = entry.get("structure_type")
        if st is None:
            raise ValueError(
                f"Blueprint entry '{entry.get('roi_id', '?')}' missing 'structure_type'"
            )
        if st not in chamber_types:
            raise ValueError(
                f"Blueprint entry '{entry.get('roi_id', '?')}' references unknown "
                f"structure_type '{st}'"
            )

    return ChipConfig(
        chip_name=data["chip_name"],
        version=data["version"],
        description=data.get("description", ""),
        pixel_size=data["pixel_size"],
        chamber_types=chamber_types,
        blueprint_map=blueprint_map,
    )


class ChipStructureLibrary:
    """Structure library for any chip design, loaded from a unified config.

    This is the primary class for looking up chamber geometry and marker
    positions given an ROI ID. It replaces SAKRoIStructureLibrary.

    Attributes:
        chip_config: The loaded ChipConfig
        pixel_size: Pixel size used for scaling (microns per pixel)
        polygon_library: Dict of chamber name to scaled RoIPolygon
        marker_group_configs: Dict of chamber name to marker positions in pixels
    """

    def __init__(self, chip_config: ChipConfig, pixel_size: float | None = None):
        """Initialize from a ChipConfig.

        Args:
            chip_config: Loaded chip configuration
            pixel_size: Override pixel size (microns/pixel). If None, uses
                        the value from chip_config.
        """
        self.chip_config = chip_config
        self.pixel_size = pixel_size if pixel_size is not None else chip_config.pixel_size

        # Build polygon library: convert GeoJSON to scaled RoIPolygon objects
        self.polygon_library: dict[str, RoIPolygon] = {}
        for name, ct in chip_config.chamber_types.items():
            shapely_poly = shape(ct.polygon)
            rp = RoIPolygon(shapely_poly)

            # Scale from microns to pixels
            rp = rp.scale(1.0 / self.pixel_size)

            # Move polygon into positive coordinates
            xmin, ymin, _, _ = rp.roi_polygon.bounds
            rp = rp.translate(x=-xmin, y=-ymin)

            self.polygon_library[name] = rp

        # Build marker group configs: convert micron positions to pixel positions
        self.marker_group_configs: dict[str, dict[str, np.ndarray]] = {}
        for name, ct in chip_config.chamber_types.items():
            markers_px = {}
            for marker_name, pos_microns in ct.markers.items():
                markers_px[marker_name] = np.array(pos_microns, dtype=float) / self.pixel_size
            self.marker_group_configs[name] = markers_px

        # Build ROI-to-structure lookup from blueprint_map entries
        self._roi_to_structure: dict[str, str] = {
            entry["roi_id"]: entry["structure_type"] for entry in chip_config.blueprint_map
        }

    @classmethod
    def from_file(cls, path: Path | str, pixel_size: float | None = None) -> ChipStructureLibrary:
        """Load a ChipStructureLibrary from a chip config JSON file.

        Args:
            path: Path to the chip config JSON file
            pixel_size: Override pixel size. If None, uses the config's value.

        Returns:
            ChipStructureLibrary instance
        """
        config = load_chip_config(path)
        return cls(config, pixel_size=pixel_size)

    def __call__(self, roi_id: str) -> tuple[str, RoIPolygon, dict[str, np.ndarray]]:
        """Look up chamber structure for a given ROI ID.

        Args:
            roi_id: ROI identifier string (e.g., "0050")

        Returns:
            Tuple of (structure_name, RoIPolygon, marker_group_configs)

        Raises:
            ValueError: If no chamber type matches the ROI ID
        """
        roi_id = str(roi_id)

        name = self._roi_to_structure.get(roi_id)
        if name is None:
            raise ValueError(f"No structure found corresponding to the roi id {roi_id}!")

        return (
            name,
            self.polygon_library[name],
            self.marker_group_configs[name],
        )

    def get_blueprint_map(self) -> Map:
        """Get the blueprint map from the chip configuration.

        Returns:
            Map object with all ROI positions from the blueprint

        Raises:
            ValueError: If the chip config has no blueprint_map entries
        """
        if not self.chip_config.blueprint_map:
            raise ValueError("Chip config has no blueprint_map entries")

        return Map.from_dict_list(self.chip_config.blueprint_map)

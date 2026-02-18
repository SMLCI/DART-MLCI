#!/usr/bin/env python
"""Generate unified chip configuration file from legacy separate files.

This one-time migration script reads the existing chamber_structure.json and
sak_blueprint_map.csv, combines them with the hardcoded marker positions and
ROI patterns from mask.py, and outputs a single unified chip config JSON.

Usage:
    python scripts/generate_chip_config.py
    python scripts/generate_chip_config.py --output artifacts/chips/sak.json
"""

import argparse
import csv
import json
import re
from pathlib import Path


def gen_pattern(start_c: int, array: int) -> str:
    """Reproduce the gen_pattern function from mask.py."""
    return "|".join([rf"({c}{array}\d\d)" for c in range(start_c, 8, 2)])


def main():
    parser = argparse.ArgumentParser(description="Generate unified chip config from legacy files")
    parser.add_argument(
        "--chamber-structure",
        type=Path,
        default=Path("artifacts/chamber_structure.json"),
        help="Path to chamber_structure.json",
    )
    parser.add_argument(
        "--blueprint-map",
        type=Path,
        default=Path("artifacts/sak_blueprint_map.csv"),
        help="Path to sak_blueprint_map.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/chips/sak.json"),
        help="Output path for unified chip config",
    )
    args = parser.parse_args()

    # 1. Load chamber structure polygons
    print(f"Loading chamber structures from {args.chamber_structure}...")
    with open(args.chamber_structure, encoding="utf-8") as f:
        polygons = json.load(f)
    print(f"  Found {len(polygons)} chamber types")

    # 2. Load blueprint map
    print(f"Loading blueprint map from {args.blueprint_map}...")
    blueprint_entries = []
    with open(args.blueprint_map, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blueprint_entries.append(
                {
                    "roi_id": row["roi_id"].zfill(4),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                }
            )
    print(f"  Found {len(blueprint_entries)} ROI positions")

    # 3. Define hardcoded marker positions (in microns, from mask.py:285-318)
    marker_configs = {
        "NormaleBox-inner": {"cross": [4.0, 8.0], "circle": [56.0, 8.0]},
        "BigBox-inner": {"cross": [4.0, 8.0], "circle": [56.0, 8.0]},
        "OpenBox-inner": {"cross": [14.0, 8.0], "circle": [66.0, 8.0]},
        "Mothermachine-inner": {"cross": [14.0, 8.0], "circle": [66.0, 8.0]},
        "NormaleBox-pillar-inner": {"cross": [4.0, 8.0], "circle": [56.0, 8.0]},
        "BigBox-pillar-inner": {"cross": [4.0, 8.0], "circle": [56.0, 8.0]},
        "OpenBox-collector-inner": {"cross": [14.0, 8.0], "circle": [66.0, 8.0]},
        "Mothermachine-2x-inner": {"cross": [14.0, 8.0], "circle": [66.0, 8.0]},
    }

    # 4. Define ROI patterns (from mask.py:274-283, using gen_pattern)
    roi_patterns = {
        "NormaleBox-inner": gen_pattern(0, 0),
        "BigBox-inner": gen_pattern(0, 1),
        "OpenBox-inner": gen_pattern(0, 2),
        "Mothermachine-inner": gen_pattern(0, 3),
        "NormaleBox-pillar-inner": gen_pattern(1, 0),
        "BigBox-pillar-inner": gen_pattern(1, 1),
        "OpenBox-collector-inner": gen_pattern(1, 2),
        "Mothermachine-2x-inner": gen_pattern(1, 3),
    }

    # 5. Assemble chamber types (without roi_pattern)
    chamber_types = {}
    for name in polygons:
        if name not in marker_configs:
            print(f"  WARNING: No marker config for '{name}', skipping")
            continue
        if name not in roi_patterns:
            print(f"  WARNING: No ROI pattern for '{name}', skipping")
            continue

        chamber_types[name] = {
            "polygon": polygons[name],
            "markers": marker_configs[name],
        }

    # 5b. Assign structure_type to each blueprint entry via regex matching
    compiled_patterns = {name: re.compile(pattern) for name, pattern in roi_patterns.items()}
    for entry in blueprint_entries:
        roi_id = entry["roi_id"]
        matched = False
        for name, pattern in compiled_patterns.items():
            if pattern.match(roi_id) is not None:
                entry["structure_type"] = name
                matched = True
                break
        if not matched:
            print(f"  WARNING: No pattern match for ROI '{roi_id}'")

    # 6. Build unified config
    config = {
        "chip_name": "SAK",
        "version": "2.0",
        "description": "Standard Analysis Kit microfluidic chip",
        "pixel_size": 0.065789,
        "chamber_types": chamber_types,
        "blueprint_map": blueprint_entries,
    }

    # 7. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nUnified chip config written to {args.output}")

    # 8. Validate by loading it back
    print("Validating by loading config back...")
    from dmc_masking.chip import load_chip_config

    chip_config = load_chip_config(args.output)
    print(f"  Chip name: {chip_config.chip_name}")
    print(f"  Version: {chip_config.version}")
    print(f"  Chamber types: {len(chip_config.chamber_types)}")
    print(f"  Blueprint entries: {len(chip_config.blueprint_map)}")
    print("  Validation passed!")


if __name__ == "__main__":
    main()

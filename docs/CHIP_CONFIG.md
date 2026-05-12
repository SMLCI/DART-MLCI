# Unified Chip Configuration

The chip configuration system provides a single JSON file per chip design that serves as the complete source of truth for chamber geometry, marker positions, ROI patterns, and blueprint map positions.

## Quick Start

```python
from dart_mlci.chip import ChipStructureLibrary

# Load from chip config file
lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json")

# Look up chamber info for an ROI
structure_name, roi_polygon, marker_group = lib("0050")

# Get the full blueprint map
blueprint_map = lib.get_blueprint_map()
```

## JSON Schema

Each chip config file follows this structure:

```json
{
  "chip_name": "SAK",
  "version": "2.0",
  "description": "Swiss Army Knife microfluidic chip",

  "chamber_types": {
    "ChamberTypeName": {
      "polygon": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], ...]]
      },
      "markers": {
        "cross": [x_microns, y_microns],
        "circle": [x_microns, y_microns]
      }
    }
  },

  "blueprint_map": [
    {"roi_id": "0000", "x": 5278, "y": -37408, "structure_type": "ChamberTypeName"},
    {"roi_id": "0001", "x": 5278, "y": -37298, "structure_type": "ChamberTypeName"}
  ]
}
```

> **Note.** `pixel_size` is **not** part of the chip config. It is a property of
> the microscope, not the chip, and lives in `folder_config.json` instead. The
> authoritative list of required fields is the `required` list in
> [`dart_mlci/chip.py`](../dart_mlci/chip.py).

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chip_name` | string | Yes | Human-readable name (e.g., "SAK") |
| `version` | string | Yes | Config file version |
| `description` | string | No | Description of the chip design |
| `chamber_types` | object | Yes | Map of chamber type name to config |
| `blueprint_map` | array | No | List of ROI positions |

#### Chamber Type Fields

| Field | Type | Description |
|-------|------|-------------|
| `polygon` | GeoJSON | Chamber outline in microns (GeoJSON Polygon format) |
| `markers.cross` | [x, y] | Cross marker position in microns relative to chamber origin |
| `markers.circle` | [x, y] | Circle marker position in microns relative to chamber origin |

#### Blueprint Map Entry

| Field | Type | Description |
|-------|------|-------------|
| `roi_id` | string | ROI identifier (zero-padded to 4 digits) |
| `x` | number | X position in blueprint coordinates |
| `y` | number | Y position in blueprint coordinates |
| `structure_type` | string | Name of the chamber type (must match a key in `chamber_types`) |

## Creating a New Chip Design

### Step 1: Define Chamber Types

For each unique chamber shape on your chip, create a GeoJSON polygon with the outline coordinates in microns:

```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0], [0.0, 0.0]]]
  }
}
```

### Step 2: Measure Marker Positions

For each chamber type, determine where the cross and circle markers sit relative to the chamber polygon's origin, measured in microns:

```json
{
  "markers": {
    "cross": [4.0, 8.0],
    "circle": [56.0, 8.0]
  }
}
```

### Step 3: List Blueprint Positions

Add all ROI positions from your chip's blueprint, with each entry specifying its `structure_type`:

```json
{
  "blueprint_map": [
    {"roi_id": "0000", "x": 5278, "y": -37408, "structure_type": "SmallBox"},
    {"roi_id": "0001", "x": 5278, "y": -37298, "structure_type": "SmallBox"}
  ]
}
```

### Step 4: Validate

Load and validate your config:

```python
from dart_mlci.chip import load_chip_config

config = load_chip_config("my_chip.json")
print(f"Loaded {config.chip_name} with {len(config.chamber_types)} chamber types")
```

## Marker Detection: When You Need a New Model

The bundled YOLO weights in `artifacts/models/v26_detect_s_imgsz1280.pt` are
trained on the **SAK-style cross-and-circle fiducials**. The pipeline assumes
each chamber has exactly one `cross` and one `circle` marker, matched in pairs.

You can reuse the bundled detector if your chip uses the same fiducial shapes
(any size/orientation/spacing — the matcher tolerates that). If your chip uses
different markers, you must:

1. Collect annotated training data (cross/circle bounding boxes in your
   microscope's images).
2. Retrain the YOLO model — see Ultralytics' standard training flow
   (`yolo detect train ...`).
3. Point `folder_config.json` at the new weights via the `model_path` field.

Retraining is out of scope for this guide, but the API surface
(`MarkerDetectionModel(weights_path=...)`) is stable.

## Validation Checklist

Before considering a new chip config done, confirm each of these:

- [ ] `ChipStructureLibrary.from_file("my_chip.json")` loads without error.
- [ ] Every `blueprint_map` entry's `structure_type` matches a key in
  `chamber_types` (the loader will raise on a mismatch — but lint your map
  first if it is large).
- [ ] Polygons are closed (first coordinate equals last) and use the GeoJSON
  outer-ring convention.
- [ ] Marker coordinates fall inside, on, or near the polygon — they are
  drawn on the chip surface, not floating in space.
- [ ] An end-to-end smoke test passes on one real image:
  ```bash
  python scripts/process_image.py \
      --image one_frame.tif \
      --chip-config my_chip.json \
      --chamber-id 0000 \
      --output /tmp/out/
  ```
  Inspect `/tmp/out/cropped.png` and the marker overlay — the ROI should be
  axis-aligned and tightly cropped.

## Minimal Example

A toy 2-chamber chip config:

```json
{
  "chip_name": "MiniChip",
  "version": "2.0",
  "description": "Minimal example chip with 2 chamber types",
  "chamber_types": {
    "SmallBox": {
      "polygon": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]]]
      },
      "markers": {
        "cross": [4.0, 8.0],
        "circle": [36.0, 8.0]
      }
    },
    "LargeBox": {
      "polygon": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [80, 0], [80, 80], [0, 80], [0, 0]]]
      },
      "markers": {
        "cross": [8.0, 8.0],
        "circle": [72.0, 8.0]
      }
    }
  },
  "blueprint_map": [
    {"roi_id": "0000", "x": 100, "y": 200, "structure_type": "SmallBox"},
    {"roi_id": "0100", "x": 300, "y": 400, "structure_type": "LargeBox"}
  ]
}
```

## Usage Examples

### Python API

```python
from dart_mlci.chip import ChipStructureLibrary

lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json")

# Look up by ROI ID
name, polygon, markers = lib("0050")
print(f"Chamber type: {name}")
print(f"Polygon area: {polygon.area:.0f} px^2")

# Get blueprint map
blueprint = lib.get_blueprint_map()
```

### CLI Scripts

```bash
# Process an image with chip config
python scripts/process_image.py \
    --image my_image.tif \
    --chamber-id 0050 \
    --chip-config artifacts/chips/sak.json

# Calibrate with chip config
python scripts/calibrate_map.py \
    --config calibration.json \
    --chip-config artifacts/chips/sak.json \
    --output calibrated_map.csv
```

### REST API

Set the environment variable:
```bash
export DART_CHIP_CONFIG_PATH=artifacts/chips/sak.json
```

Or pass in the request body:
```json
{
  "image": "<base64>",
  "roi_id": "0050",
  "chip_config_path": "artifacts/chips/sak.json"
}
```

## Migration from Legacy Files

If you have existing `chamber_structure.json` and `sak_blueprint_map.csv` files, use the migration script:

```bash
python scripts/generate_chip_config.py \
    --chamber-structure artifacts/chamber_structure.json \
    --blueprint-map artifacts/sak_blueprint_map.csv \
    --output artifacts/chips/my_chip.json
```

The old files (`chamber_structure.json`, `sak_blueprint_map.csv`) are still supported but deprecated. They will be removed in a future release.

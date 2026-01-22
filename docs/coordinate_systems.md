# Coordinate Systems in DMC Masking

This document describes the coordinate systems used in the DMC masking calibration pipeline
and how they relate to each other.

## Overview

The calibration pipeline works with three coordinate systems:

```
+---------------------------+---------------------------+---------------------------+
| BLUEPRINT (Design)        | IMAGE PIXEL               | STAGE                     |
+---------------------------+---------------------------+---------------------------+
| Units: microns            | Units: pixels             | Units: microns            |
| Origin: Design top-left   | Origin: Image top-left    | Origin: Hardware ref      |
| Y: UP (Cartesian)         | Y: DOWN (standard image)  | Y: DOWN (typically)       |
| View: TOP-DOWN            | View: Camera (from below) | Hardware-dependent        |
+---------------------------+---------------------------+---------------------------+
```

## 1. Blueprint (Design) Coordinates

The blueprint represents the chip design as viewed from above (top-down view).

- **Units**: Microns
- **Convention**: Cartesian (+Y points UP)
- **Origin**: Typically top-left of the chip design

```
BLUEPRINT COORDINATE SYSTEM (Cartesian)

         +Y ^
            |
            |
   +--------+--------+
   |                 |
   |   chip design   |
   |   (top-down     |
   |    view)        |
   +--------+--------+---> +X
          origin
```

Marker positions in the blueprint (e.g., `cross=(14, 8)`) mean:
- X: 14 units to the right of origin
- Y: 8 units UP from origin (Cartesian convention)

## 2. Image Pixel Coordinates

Camera images use standard image coordinate convention.

- **Units**: Pixels
- **Convention**: Standard image (+Y points DOWN)
- **Origin**: Top-left corner of the image

```
IMAGE COORDINATE SYSTEM (Standard Image)

   origin (0,0)
      +---------------+---> +X
      |               |
      |   captured    |
      |   image       |
      |               |
      +---------------+
      |
      v +Y
```

## 3. Stage (Microscope) Coordinates

Physical microscope stage coordinates.

- **Units**: Microns
- **Convention**: Hardware-dependent (typically +Y DOWN)
- **Origin**: Hardware reference point

## The Y-Axis Inversion (Critical!)

The most important coordinate handling in this pipeline is the **Y-axis inversion**
between blueprint and image coordinates.

### Why Y is Inverted

- Blueprint uses **Cartesian convention**: +Y points UP
- Images use **standard image convention**: +Y points DOWN

This means a marker at Y=8 in the blueprint appears at a different Y position in
the image due to the opposite conventions.

### How the Code Handles This

When computing offsets from markers to polygon centers, the code uses **addition**
for Y instead of subtraction:

```python
# In compute_chamber_center (calibrate_map.py)
center_offset = np.array([
    polygon_center[0] - cross_local[0],    # X: normal subtraction
    polygon_center[1] + cross_local[1],    # Y: ADDITION (not subtraction!)
])
```

This is also seen in `apply_mask` (mask.py):

```python
rp = roi_polygon.translate(
    x=cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
    y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1],  # Uses +
)
```

### Visual Explanation

```
BLUEPRINT COORDINATES               IMAGE COORDINATES
(Cartesian: Y increases UP)         (Image: Y increases DOWN)

         +Y ^                              +-------+ (0,0)
            |                              |       |
   +--------+--------+                     | image |
   |   x marker at   |    Y-FLIP           |       |
   |   Y=8 (above    | ========>           +-------+
   |   origin)       |                         |
   +--------+--------+                         v +Y
          origin

In blueprint: marker Y=8 means 8 units ABOVE origin
In image:     same marker appears at Y > 0 (pixels DOWN from top)
```

### The Math

Given:
- `polygon_center = (50, 50)` - center of polygon in local coords
- `marker_position = (14, 8)` - marker position from blueprint

**Without Y-inversion (wrong):**
```
offset_x = 50 - 14 = 36
offset_y = 50 - 8 = 42   <-- This would be wrong!
```

**With Y-inversion (correct):**
```
offset_x = 50 - 14 = 36
offset_y = 50 + 8 = 58   <-- Correct: uses + for Y
```

## Transform Pipeline

The full transform pipeline from blueprint to stage coordinates:

```
                                      +-----------------+
                                      |                 |
+------------------+  x pixel_size    |  IMAGE (um)     |
| Pixel Coord (px) | ---------------+ |                 |
+------------------+                | +-----------------+
                                    |         |
                                    |         | + stage_position
                                    |         v
+------------------+   Affine       | +------------------+
| Blueprint (um)   | ---------------+ | Stage (um)       |
+------------------+   Transform      +------------------+
```

### Transform Classes

The `dmc_masking.calibration` module provides these transform classes:

1. **PixelToMicronTransform**: Scales pixel coordinates to microns
   ```python
   transform = PixelToMicronTransform(pixel_size=0.065789)
   microns = transform(pixels)
   ```

2. **ImageToStageTransform**: Adds stage position offset
   ```python
   transform = ImageToStageTransform(stage_position=np.array([6802.4, -4272.9]))
   stage_pos = transform(image_microns)
   ```

3. **AffineTransform2D**: General 2D affine transform
   ```python
   transform, fit_result = AffineTransform2D.from_point_pairs(source, target)
   transformed = transform(points)
   ```

## Physical View Considerations

### Top-Down vs Bottom-Up View

The blueprint shows the chip from **above** (top-down view), while the microscope
camera views from **below** (bottom-up view). This can create an X-mirror effect.

However, this mirror is **automatically captured** by the affine transform computed
from calibration point correspondences. The explicit Y-axis inversion (using +)
handles the coordinate convention difference, while any physical mirror/flip is
absorbed into the affine matrix.

```
BLUEPRINT (top-down)              MICROSCOPE VIEW (bottom-up)

    +------+                           +------+
    | A  B |    physical              | B  A |
    |      |    mirror  ==========>   |      |
    | C  D |    (X-flip)              | D  C |
    +------+                           +------+
```

## Configuration

The coordinate system behavior can be configured via `DMCConfig`:

```python
from dmc_masking.config import DMCConfig, CoordinatesConfig, AxisDirection

config = DMCConfig()
# Default: blueprint Y is NEGATIVE (up), image Y is POSITIVE (down)
print(config.coordinates.blueprint.y_direction)  # AxisDirection.NEGATIVE
print(config.coordinates.image.y_direction)      # AxisDirection.POSITIVE
print(config.coordinates.blueprint_to_image_invert_y)  # True
```

## Summary

| System     | X Direction | Y Direction | Notes                           |
|------------|-------------|-------------|----------------------------------|
| Blueprint  | +right      | +UP         | Cartesian convention            |
| Image      | +right      | +DOWN       | Standard image convention       |
| Stage      | +right      | varies      | Hardware-dependent              |

**Key Implementation Detail**: The Y-offset calculation uses **addition** (`+`)
instead of subtraction (`-`) to account for the Y-axis inversion between
blueprint (Y-up) and image (Y-down) coordinates.

## Code References

- `dmc_masking/calibration/coordinates.py` - Transform classes
- `dmc_masking/calibration/core.py:compute_chamber_center()` - Y-inversion handling
- `dmc_masking/mask.py:apply_mask()` - Y-inversion in polygon positioning
- `dmc_masking/config.py` - Coordinate system configuration

# Configuration Reference

This page is the reference for the runtime configuration consumed by
`scripts/process_folder.py` (`folder_config.json`) and a guide to the output
files the pipeline produces.

For the *chip* configuration (chamber geometry, marker positions, blueprint
map), see [`CHIP_CONFIG.md`](CHIP_CONFIG.md).

## `folder_config.json`

The folder config tells the pipeline **what to process**, **with what model**,
and **how to segment**. A minimal config looks like:

```json
{
  "input_dir": "data/experiment",
  "output_dir": "data/output",
  "pixel_size": 0.092766,
  "segmenter": "cellpose-sam",
  "folders": {
    "MyChamberFolder": "NormaleBox-inner"
  }
}
```

Omit `chip_config` / `model_path` to use the auto-downloaded defaults
(`chips/sak.json` and the bundled YOLO weights — both fetched on first use to
the per-user cache; see [`../README.md#installation`](../README.md#installation)).
Specify either field with an absolute path to point at a custom chip JSON or
your own model weights.

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_dir` | string | Yes | Path to the experiment directory containing chamber-type subfolders. |
| `output_dir` | string | Yes | Where per-stack results, summary, and timings are written. |
| `pixel_size` | float | Yes | Microscope pixel size in µm/px. This is a property of your microscope, not your chip. |
| `chip_config` | string | No | Path to the chip JSON (see [CHIP_CONFIG.md](CHIP_CONFIG.md)). Defaults to the auto-downloaded `chips/sak.json`. |
| `model_path` | string | No | Path to YOLO marker-detection weights. Defaults to the auto-downloaded `models/v26_detect_s_imgsz1280.pt`. |
| `folders` | object | Yes | Map of subfolder name (under `input_dir`) → chamber-type key (from the chip config's `chamber_types`). |
| `flip` | bool | No | Vertically flip frames before processing (microscope-orientation dependent). Default `false`. |
| `allow_truncation` | bool | No | Allow cropped ROIs that extend past the image boundary. Default `false`. |
| `registration` | bool | No | Enable timelapse registration. Default `false`. |
| `registration_method` | string | No | `"ncc"` (normalized cross-correlation) or `"phase"`. Default `"ncc"`. |
| `segmenter` | string | No | `"cellpose-sam"` (default) or `"omnipose"`. |
| `min_area_um2` | float | No | Drop labeled objects smaller than this area (µm²) after segmentation. |
| `max_area_um2` | float | No | Drop labeled objects larger than this area (µm²) after segmentation. |

CLI flags on `process_folder.py` override any field of the same name.

### Segmenter Choice

| Segmenter | Strengths | Notes |
|-----------|-----------|-------|
| `cellpose-sam` (default) | High-quality generalist; works out of the box on the SAK dataset. | Requires `pip install acia` (which pulls in Cellpose-SAM). |
| `omnipose` | Better for elongated cells (e.g., mother-machine channels); works well with timelapse registration enabled. | Requires `pip install cellpose_omni`. |

The configs `folder_config.json` (Omnipose + registration) and
`folder_config_cellpose_sam.json` (Cellpose-SAM, no registration) at the repo
root are reference templates for the two supported setups.

## Output Files

`process_folder.py` writes per-stack directories under `<output_dir>/<folder>/`
plus a batch-level summary at the root of `<output_dir>`.

### Per-stack directory

| File | Description |
|------|-------------|
| `stack.tif` | Segmentation masks (T × H × W, uint16, one label per cell instance). |
| `stack_chamber.tif` | Chamber boundary masks (T × H × W, uint8). |
| `stack_cropped.tif` | Cropped + rotated raw frames. Only written with `--save-cropped`. |
| `meta.csv` | Per-frame metadata: timings, registration offsets, cell counts. |
| `cells.csv` | Per-cell measurements: label, frame, area (pixels and µm²), bbox. |
| `timelapse.mp4` | Segmentation overlay video. Only with `--render-stacks`. |
| `timelapse_raw.mp4` | Raw rotated frames video. Only with `--render-stacks`. |

### Batch-level summary

| File | Description |
|------|-------------|
| `results.csv` | Per-stack success/failure, cell counts, error messages. |
| `timings.csv` | Frame-level pipeline-step timings across the whole experiment. |
| `summary.md` | Human-readable per-folder breakdown. |
| `timings_table.tex` | LaTeX table of mean step timings (for papers). |
| `pipeline_timings.png` | Waterfall chart of pipeline-step timings. |

## Common Flags for `process_folder.py`

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to a `folder_config.json`. |
| `--save-cropped` | Save `stack_cropped.tif`. |
| `--render-stacks` | Write `timelapse.mp4` and `timelapse_raw.mp4` per stack. |
| `--max-files N` | Process only the first N stacks per folder (smoke testing). |
| `--skip-existing` | Skip stacks that already have an output directory. |
| `--verbose` | Detailed per-frame logging. |
| `--segmenter NAME` | Override the config's segmenter (`cellpose-sam` or `omnipose`). |
| `--min-area-um2 X` | Drop segmentation objects smaller than X µm². |
| `--max-area-um2 X` | Drop segmentation objects larger than X µm². |

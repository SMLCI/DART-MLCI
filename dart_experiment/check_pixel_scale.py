"""Check pixel scale factor across all chamber types in the DART experiment."""

from pathlib import Path

import numpy as np
import tifffile

from dart_mlci.chip import ChipStructureLibrary
from dart_mlci.detection import MarkerDetectionModel
from dart_mlci.utils import normalize_image

chip_lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json", pixel_size=1.0)
mdm = MarkerDetectionModel("artifacts/models/v26_detect_s_imgsz1280.pt", verbose=False)

config_pixel_size = 0.0904140
folders = {
    "Small Chambers": "NormaleBox-inner",
    "Big Chambers": "BigBox-inner",
    "Big Chambers + Pillars": "BigBox-pillar-inner",
    "Open Chambers": "OpenBox-inner",
    "Open Chambers + Structures": "OpenBox-collector-inner",
    "Mother Machines": "Mothermachine-2x-inner",
    "Small Chambers + Pillar": "NormaleBox-pillar-inner",
}

input_dir = Path("dart_experiment/DART_Experiment")

header = (
    f"{'Folder':<30s} {'ChamberType':<25s} {'PhysDist(um)':>13s} "
    f"{'DetDist(px)':>13s} {'ExpDist(px)':>13s} {'NeededPxSz':>13s} "
    f"{'CfgPxSz':>13s} {'Error(%)':>10s}"
)
print(header)
print("-" * len(header))

all_needed = []

for folder_name, ctype in folders.items():
    folder_path = input_dir / folder_name
    if not folder_path.exists():
        print(f"{folder_name:<30s} FOLDER NOT FOUND")
        continue

    # Find first tif directly in the folder
    tifs = sorted(folder_path.glob("*.tif")) + sorted(folder_path.glob("*.tiff"))
    if not tifs:
        print(f"{folder_name:<30s} NO TIF FOUND")
        continue
    tif_path = tifs[0]

    # Load first frame, flip y, normalize to uint8
    stack = tifffile.imread(str(tif_path))
    frame = stack[0] if stack.ndim == 3 else stack
    frame = frame[::-1]  # y-flip
    frame = normalize_image(frame)  # uint16 -> uint8

    # Convert to 3-channel if needed
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    # Detect markers
    markers = mdm.predict_markers(frame)
    markers = [m for m in markers if m.get("conf", 0) >= 0.5]

    crosses = [m for m in markers if m["label"] == "cross"]
    circles = [m for m in markers if m["label"] == "circle"]

    if not crosses or not circles:
        print(
            f"{folder_name:<30s} {ctype:<25s} "
            f"No marker pairs (crosses={len(crosses)}, circles={len(circles)})"
        )
        continue

    # Get physical distance from chip config (access marker_group_configs directly)
    mg_phys = chip_lib.marker_group_configs[ctype]
    phys_dist = np.linalg.norm(np.array(mg_phys["cross"]) - np.array(mg_phys["circle"]))

    # Expected distance at current config pixel size
    expected_dist = phys_dist / config_pixel_size

    # Find best matching cross-circle pair
    best_det_dist = None
    best_err = float("inf")
    for c in crosses:
        for ci in circles:
            d = np.linalg.norm(np.array(c["bbox_center"]) - np.array(ci["bbox_center"]))
            err = abs(d - expected_dist)
            if err < best_err:
                best_err = err
                best_det_dist = d

    needed_px = phys_dist / best_det_dist
    error_pct = (needed_px - config_pixel_size) / config_pixel_size * 100
    all_needed.append(needed_px)

    print(
        f"{folder_name:<30s} {ctype:<25s} {phys_dist:>13.2f} "
        f"{best_det_dist:>13.1f} {expected_dist:>13.1f} {needed_px:>13.6f} "
        f"{config_pixel_size:>13.6f} {error_pct:>+10.2f}"
    )

if all_needed:
    print()
    mean_needed = np.mean(all_needed)
    std_needed = np.std(all_needed)
    print(f"Mean needed pixel size: {mean_needed:.6f} +/- {std_needed:.6f}")
    print(f"Current config pixel size: {config_pixel_size:.6f}")
    print(f"Difference: {(mean_needed - config_pixel_size) / config_pixel_size * 100:+.2f}%")

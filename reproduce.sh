#!/usr/bin/env bash
# reproduce.sh — Automate the DART experiment reproduction steps from the README.
#
# Usage:
#   bash reproduce.sh            # smoke test (processes 1 stack per folder)
#   bash reproduce.sh --full     # full experiment (all stacks)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
FULL=false
for arg in "$@"; do
    case "$arg" in
        --full) FULL=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: bash reproduce.sh [--full]"
            exit 1
            ;;
    esac
done

if $FULL; then
    echo "=== DART Experiment — FULL run ==="
else
    echo "=== DART Experiment — smoke test (--max-files 1) ==="
fi

CONFIG="dart_experiment/folder_config_cellpose_sam.json"
OUTPUT_DIR="dart_experiment/output_reproduce"
DATA_DIR="dart_experiment/DART_Experiment"
ZIP_URL="https://fz-juelich.sciebo.de/s/Tq5SW76WG9zqMJi/download"
ZIP_FILE="DART_Experiment.zip"

PASS=0
FAIL=0

step_pass() { PASS=$((PASS + 1)); echo "  -> PASS"; }
step_fail() { FAIL=$((FAIL + 1)); echo "  -> FAIL: $1"; }

# ---------------------------------------------------------------------------
# Step 1: Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Install dependencies ---"
pip install . && pip install acia
echo "  -> done"

# ---------------------------------------------------------------------------
# Step 2: Download dataset (skip if already present)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Download dataset ---"
if [ -d "$DATA_DIR" ]; then
    echo "  Dataset already exists at $DATA_DIR — skipping download."
else
    echo "  Downloading from sciebo..."
    wget -q --show-progress -O "$ZIP_FILE" "$ZIP_URL"
    echo "  Unzipping..."
    unzip -q "$ZIP_FILE" -d dart_experiment/
    rm -f "$ZIP_FILE"
    echo "  -> done"
fi

# ---------------------------------------------------------------------------
# Step 3: Validate data structure
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Validate data structure ---"
EXPECTED_FOLDERS=(
    "Small Chambers"
    "Big Chambers"
    "Big Chambers + Pillars"
    "Open Chambers"
    "Open Chambers + Structures"
    "Mother Machines"
    "Small Chambers + Pillar"
)
ALL_FOUND=true
for folder in "${EXPECTED_FOLDERS[@]}"; do
    if [ ! -d "$DATA_DIR/$folder" ]; then
        echo "  MISSING: $DATA_DIR/$folder"
        ALL_FOUND=false
    fi
done
if $ALL_FOUND; then
    step_pass
else
    step_fail "Some expected subfolders are missing"
fi

# ---------------------------------------------------------------------------
# Step 4: Pixel calibration
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Pixel calibration ---"
if python scripts/calibrate_pixel_scale.py --config "$CONFIG"; then
    step_pass
else
    step_fail "calibrate_pixel_scale.py failed"
fi

# ---------------------------------------------------------------------------
# Step 5: Patch config to use dedicated output directory
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Prepare output config ---"
REPRO_CONFIG="dart_experiment/folder_config_reproduce.json"
python -c "
import json, pathlib
cfg = json.loads(pathlib.Path('$CONFIG').read_text())
cfg['output_dir'] = '$OUTPUT_DIR'
pathlib.Path('$REPRO_CONFIG').write_text(json.dumps(cfg, indent=2))
print('  Wrote', '$REPRO_CONFIG', 'with output_dir =', '$OUTPUT_DIR')
"

# ---------------------------------------------------------------------------
# Step 6: Run processing
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: Run processing ---"
PROCESS_CMD="python scripts/process_folder.py --config $REPRO_CONFIG --save-cropped --verbose"
if ! $FULL; then
    PROCESS_CMD="$PROCESS_CMD --max-files 1"
fi
echo "  $PROCESS_CMD"
if $PROCESS_CMD; then
    step_pass
else
    step_fail "process_folder.py failed"
fi

# ---------------------------------------------------------------------------
# Step 7: Validate outputs
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: Validate outputs ---"
MISSING_OUTPUTS=""

# Check batch-level files
for f in results.csv summary.md; do
    if [ ! -f "$OUTPUT_DIR/$f" ]; then
        MISSING_OUTPUTS="$MISSING_OUTPUTS $f"
    fi
done

# Check that at least one stack.tif exists in a subfolder
STACK_COUNT=$(find "$OUTPUT_DIR" -name "stack.tif" 2>/dev/null | head -20 | wc -l)
if [ "$STACK_COUNT" -eq 0 ]; then
    MISSING_OUTPUTS="$MISSING_OUTPUTS stack.tif(in-subfolder)"
fi

if [ -z "$MISSING_OUTPUTS" ]; then
    echo "  Found results.csv, summary.md, and $STACK_COUNT stack.tif file(s)"
    step_pass
else
    step_fail "Missing outputs:$MISSING_OUTPUTS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "======================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

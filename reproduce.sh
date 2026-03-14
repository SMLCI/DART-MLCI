#!/usr/bin/env bash
# reproduce.sh — Automate the DART experiment reproduction steps from the README.
#
# Usage:
#   bash reproduce.sh            # smoke test (processes 1 stack per folder)
#   bash reproduce.sh --full     # full experiment (all stacks)
#   bash reproduce.sh --map-only # run only map calibration/validation (steps 1,4-6)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
FULL=false
MAP_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --full) FULL=true ;;
        --map-only) MAP_ONLY=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: bash reproduce.sh [--full] [--map-only]"
            exit 1
            ;;
    esac
done

if $MAP_ONLY; then
    echo "=== DART Experiment — map calibration only ==="
elif $FULL; then
    echo "=== DART Experiment — FULL run ==="
else
    echo "=== DART Experiment — smoke test (--max-files 1) ==="
fi

CONFIG="dart_experiment/folder_config_cellpose_sam.json"
OUTPUT_DIR="dart_experiment/output_reproduce"
DATA_DIR="dart_experiment/DART_Experiment"
ZIP_URL="https://fz-juelich.sciebo.de/s/Tq5SW76WG9zqMJi/download"
ZIP_FILE="DART_Experiment.zip"
CONDA_ENV="dmc-reproduce"

# Map calibration data
CAL_DATA_DIR="dart_experiment/calibration_data"
CAL_SUBSET_URL="file://$(pwd)/calibration_zips/calibration_data_subset.zip"
CAL_FULL_URL="file://$(pwd)/calibration_zips/calibration_data_full.zip"
CAL_ZIP_FILE="calibration_data.zip"

PASS=0
FAIL=0

step_pass() { PASS=$((PASS + 1)); echo "  -> PASS"; }
step_fail() { FAIL=$((FAIL + 1)); echo "  -> FAIL: $1"; }

# Helper: run a command inside the conda environment
run() { conda run -n "$CONDA_ENV" --live-stream "$@"; }

# ---------------------------------------------------------------------------
# Step 1: Create conda environment and install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Create conda env and install dependencies ---"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Creating conda environment '$CONDA_ENV' with Python 3.10 ..."
    conda create -y -n "$CONDA_ENV" python=3.10 -q
fi
echo "  Using Python: $(run python --version)"
run pip install --upgrade pip -q
run pip install -e . && run pip install acia cellpose
echo "  -> done"

# ---------------------------------------------------------------------------
# Step 2: Download dataset (skip if already present)
# ---------------------------------------------------------------------------
if ! $MAP_ONLY; then
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
    # The zip contains DMC_Experiment/ — rename to match config's input_dir
    if [ -d "dart_experiment/DMC_Experiment" ] && [ ! -d "$DATA_DIR" ]; then
        mv "dart_experiment/DMC_Experiment" "$DATA_DIR"
        echo "  Renamed DMC_Experiment -> DART_Experiment"
    fi
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
fi  # end !MAP_ONLY (steps 2-3)

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Download calibration data (skip if already present)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Download calibration data ---"
if [ -d "$CAL_DATA_DIR" ]; then
    echo "  Calibration data already exists at $CAL_DATA_DIR — skipping download."
else
    if $FULL; then
        CAL_URL="$CAL_FULL_URL"
        echo "  Using full calibration data..."
    else
        CAL_URL="$CAL_SUBSET_URL"
        echo "  Using subset calibration data..."
    fi
    if [[ "$CAL_URL" == file://* ]]; then
        cp "${CAL_URL#file://}" "$CAL_ZIP_FILE"
    else
        wget -q --show-progress -O "$CAL_ZIP_FILE" "$CAL_URL"
    fi
    echo "  Unzipping..."
    unzip -q "$CAL_ZIP_FILE" -d dart_experiment/
    rm -f "$CAL_ZIP_FILE"
    echo "  -> done"
fi

# ---------------------------------------------------------------------------
# Step 5: Run map calibration
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Run map calibration ---"
CALIBRATED_MAP="$OUTPUT_DIR/calibrated_map.csv"
if run python scripts/calibrate_map.py \
    --config "$CAL_DATA_DIR/calibration_config.json" \
    --output "$CALIBRATED_MAP" \
    --output-dir "$OUTPUT_DIR/calibration" \
    --verbose; then
    if [ -f "$CALIBRATED_MAP" ]; then
        step_pass
    else
        step_fail "calibrated_map.csv not created"
    fi
else
    step_fail "calibrate_map.py failed"
fi

# ---------------------------------------------------------------------------
# Step 6: Validate calibrated map
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: Validate calibrated map ---"
VALIDATION_DIR="$OUTPUT_DIR/validation"
# Patch validation config to point to the calibrated map we just created
run python -c "
import json, pathlib
cfg = json.loads(pathlib.Path('$CAL_DATA_DIR/validation_config.json').read_text())
cfg['calibrated_map_path'] = '$CALIBRATED_MAP'
cfg['meta_csv_path'] = '$CAL_DATA_DIR/meta.csv'
cfg['images_dir'] = '$CAL_DATA_DIR/images'
pathlib.Path('$OUTPUT_DIR/validation_config.json').write_text(json.dumps(cfg, indent=2))
print('  Wrote patched validation config')
"
if run python scripts/validate_map.py \
    --config "$OUTPUT_DIR/validation_config.json" \
    --output-dir "$VALIDATION_DIR" \
    --verbose; then
    MISSING_VAL=""
    [ ! -f "$VALIDATION_DIR/validation_results.csv" ] && MISSING_VAL="$MISSING_VAL validation_results.csv"
    [ ! -f "$VALIDATION_DIR/error_histogram.png" ] && MISSING_VAL="$MISSING_VAL error_histogram.png"
    if [ -z "$MISSING_VAL" ]; then
        step_pass
    else
        step_fail "Missing validation outputs:$MISSING_VAL"
    fi
else
    step_fail "validate_map.py failed"
fi

# ---------------------------------------------------------------------------
# Step 7: Pixel calibration
# ---------------------------------------------------------------------------
if ! $MAP_ONLY; then
echo ""
echo "--- Step 7: Pixel calibration ---"
if run python scripts/calibrate_pixel_scale.py --config "$CONFIG"; then
    step_pass
else
    step_fail "calibrate_pixel_scale.py failed"
fi

# ---------------------------------------------------------------------------
# Step 8: Prepare output config
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 8: Prepare output config ---"
REPRO_CONFIG="dart_experiment/folder_config_reproduce.json"
run python -c "
import json, pathlib
cfg = json.loads(pathlib.Path('$CONFIG').read_text())
cfg['output_dir'] = '$OUTPUT_DIR'
pathlib.Path('$REPRO_CONFIG').write_text(json.dumps(cfg, indent=2))
print('  Wrote', '$REPRO_CONFIG', 'with output_dir =', '$OUTPUT_DIR')
"

# ---------------------------------------------------------------------------
# Step 9: Run processing
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 9: Run processing ---"
PROCESS_ARGS="--config $REPRO_CONFIG --save-cropped --verbose"
if ! $FULL; then
    PROCESS_ARGS="$PROCESS_ARGS --max-files 1"
fi
echo "  python scripts/process_folder.py $PROCESS_ARGS"
if run python scripts/process_folder.py $PROCESS_ARGS; then
    step_pass
else
    step_fail "process_folder.py failed"
fi

# ---------------------------------------------------------------------------
# Step 10: Validate outputs
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 10: Validate outputs ---"
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
fi  # end !MAP_ONLY (steps 7-10)

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

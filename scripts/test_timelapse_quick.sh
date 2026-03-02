#!/bin/bash
# Quick test of time-lapse stacking pipeline (1 ROI only)
# Usage: ./scripts/test_timelapse_quick.sh [dataset_dir] [output_dir]

set -e

DATASET_DIR="${1:-/data/EMSIG/jedi2rt/nodes/jupyter_notebook_new/2025_02_21/mapping/data/output-2025-02-25_16:10:00}"
OUTPUT_DIR="${2:-./test_quick}"

echo "=== QUICK TEST MODE (1 ROI) ==="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dmc-masking-claude

# Run with 1 ROI for quick testing
python scripts/process_experiment.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --enable-stacking \
  --enable-registration \
  --max-rois 1 \
  --save-cropped \
  --render-stacks \
  --verbose

echo ""
echo "Quick test complete! Check: $OUTPUT_DIR/roi_0000/"

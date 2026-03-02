#!/bin/bash
# Time-lapse stacking pipeline for DMC masking
# This script processes time-lapse experiments with optional registration

# Exit on error
set -e

# Default parameters (modify these as needed)
DATASET_DIR="${1:-/data/EMSIG/artifacts/lukas_data/output-2025-01-07_10:15:37}"
OUTPUT_DIR="${2:-/mnt/nvme_storage/experiments/DMC/2025-01-07}"
MAX_ROIS="${3:-}"  # Empty by default = process ALL ROIs. Set number to limit (e.g., 5)

# Activate conda environment
echo "Activating conda environment: dmc-masking-claude"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dmc-masking-claude

# Build command
CMD="python scripts/process_experiment.py \
  --dataset-dir \"$DATASET_DIR\" \
  --output-dir \"$OUTPUT_DIR\" \
  --enable-stacking \
  --enable-registration \
  --save-cropped \
  --render-stacks \
  --skip-existing \
  --verbose"

# Add max-rois if specified
if [ -n "$MAX_ROIS" ]; then
  CMD="$CMD --max-rois $MAX_ROIS"
  echo "Processing first $MAX_ROIS ROI(s) for testing"
else
  echo "Processing ALL ROIs"
fi

# Display command
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Run the pipeline
eval $CMD

# Report completion
echo ""
echo "========================================"
echo "Processing complete!"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Check summary: cat $OUTPUT_DIR/summary.md"
echo "  2. View results: cat $OUTPUT_DIR/results.csv"
echo "  3. Open stacks in ImageJ/Fiji"

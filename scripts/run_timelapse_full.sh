#!/bin/bash
# Full time-lapse stacking pipeline (ALL ROIs)
# Usage: ./scripts/run_timelapse_full.sh <dataset_dir> <output_dir>

set -e

# Require both arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset_dir> <output_dir>"
  echo ""
  echo "Example:"
  echo "  $0 /path/to/experiment ./final_output"
  exit 1
fi

DATASET_DIR="$1"
OUTPUT_DIR="$2"

# Verify dataset exists
if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Dataset directory not found: $DATASET_DIR"
  exit 1
fi

echo "=== FULL PROCESSING MODE (ALL ROIs) ==="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo ""
echo "WARNING: This will process ALL ROIs in the dataset."
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dmc-masking-claude

# Record start time
START_TIME=$(date +%s)

# Run full pipeline
python scripts/process_experiment.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --enable-stacking \
  --enable-registration \
  --save-cropped \
  --skip-existing

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "Full processing complete!"
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Results:"
echo "  - Summary: $OUTPUT_DIR/summary.md"
echo "  - Results CSV: $OUTPUT_DIR/results.csv"
echo "  - ROI stacks: $OUTPUT_DIR/roi_*/stack.tif"

#!/usr/bin/env bash
# Download model weights and test images required for development/testing.
# These files are hosted on Sciebo (FZ Jülich cloud storage) to avoid
# Git LFS bandwidth costs on GitHub.
set -euo pipefail

ARTIFACTS_URL="${DART_ARTIFACTS_URL:-https://fz-juelich.sciebo.de/s/S4bYt6C9rtR3sF2/download}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACTS_DIR="$REPO_DIR/artifacts"

if [ -d "$ARTIFACTS_DIR/models" ] && [ -d "$ARTIFACTS_DIR/images" ]; then
    echo "Artifacts already present — skipping download."
    exit 0
fi

echo "Downloading artifacts from Sciebo..."
wget -q --show-progress -O /tmp/dart_artifacts.zip "$ARTIFACTS_URL"
echo "Extracting to $ARTIFACTS_DIR..."
unzip -qo /tmp/dart_artifacts.zip -d "$REPO_DIR"
rm /tmp/dart_artifacts.zip
echo "Done. Artifacts at $ARTIFACTS_DIR"

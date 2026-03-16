FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY README.md .
COPY dart_mlci/ dart_mlci/
COPY scripts/ scripts/

# Install the package with API and segmentation dependencies
RUN pip install --no-cache-dir -e ".[api,segmentation]"

# Copy artifacts (model, structure library, blueprint map)
COPY artifacts/ /app/artifacts/

# Set environment variables for default paths
ENV DART_MODEL_PATH=/app/artifacts/models/v26_detect_s_imgsz1280.pt
ENV DART_STRUCTURE_LIBRARY_PATH=/app/artifacts/chamber_structure.json
ENV DART_BLUEPRINT_MAP_PATH=/app/artifacts/sak_blueprint_map.csv
ENV DART_CHIP_CONFIGS_DIR=/app/artifacts/chips/
ENV DART_PIXEL_SIZE=0.065789
ENV DART_SEGMENTER=cellpose-sam

# Expose port
EXPOSE 8000

# Run FastAPI server
# GPU is auto-detected - works with or without CUDA
CMD ["uvicorn", "dart_mlci.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

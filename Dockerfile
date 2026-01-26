FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY dmc_masking/ dmc_masking/
COPY scripts/ scripts/

# Install the package with API dependencies
RUN pip install --no-cache-dir -e ".[api]"

# Copy artifacts (model, structure library, blueprint map)
COPY artifacts/ /app/artifacts/

# Set environment variables for default paths
ENV DMC_MODEL_PATH=/app/artifacts/models/v8_detect_s_imgsz640.pt
ENV DMC_STRUCTURE_LIBRARY_PATH=/app/artifacts/chamber_structure.json
ENV DMC_BLUEPRINT_MAP_PATH=/app/artifacts/sak_blueprint_map.csv
ENV DMC_PIXEL_SIZE=0.065789

# Expose port
EXPOSE 8000

# Run FastAPI server
# GPU is auto-detected - works with or without CUDA
CMD ["uvicorn", "dmc_masking.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

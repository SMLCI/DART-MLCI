FROM python:3.12-slim

# Install system dependencies for OpenCV and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package from git (requires deploy token)
ARG DEPLOY_TOKEN_KEY
ARG DEPLOY_TOKEN_NAME=gitlab+deploy-token-440
RUN pip install --no-cache-dir git+https://${DEPLOY_TOKEN_NAME}:${DEPLOY_TOKEN_KEY}@jugit.fz-juelich.de/emsig/dmc-masking.git

CMD ["python"]

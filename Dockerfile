# syntax=docker/dockerfile:1.4
# Allow overriding the base image to a GPU-ready image (e.g., PyTorch CUDA runtime)
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for Pillow (zlib, libjpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
COPY api/requirements-ml.txt /app/api/requirements-ml.txt

ARG INSTALL_ML_DEPS=true
ARG DOWNLOAD_MODELS=true
ARG STABLEVITON_SHAREPOINT_URL
ARG CONTROLNET_OPENPOSE_URL
ARG SCHP_DRIVE_URL
ARG INSTANTID_ANTELOPE_URL
ARG HUGGINGFACE_HUB_TOKEN
ENV INSTALL_ML_DEPS=${INSTALL_ML_DEPS}
ENV DOWNLOAD_MODELS=${DOWNLOAD_MODELS}
ENV STABLEVITON_SHAREPOINT_URL=${STABLEVITON_SHAREPOINT_URL}
ENV CONTROLNET_OPENPOSE_URL=${CONTROLNET_OPENPOSE_URL}
ENV SCHP_DRIVE_URL=${SCHP_DRIVE_URL}
ENV INSTANTID_ANTELOPE_URL=${INSTANTID_ANTELOPE_URL}
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
# Use BuildKit cache for pip to avoid re-downloading ML deps across rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/api/requirements.txt && \
    if [ "$INSTALL_ML_DEPS" = "true" ]; then \
        pip install -r /app/api/requirements-ml.txt ; \
    fi

COPY api /app/api
COPY scripts /app/scripts
RUN chmod +x /app/scripts/entrypoint.sh

RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        python /app/scripts/download_models.py || echo "Model download script ended with non-zero status" ; \
    fi

EXPOSE 8008 8000

WORKDIR /app/api

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]

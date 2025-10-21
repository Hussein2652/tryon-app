FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for Pillow (zlib, libjpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
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
RUN pip install --no-cache-dir -r /app/api/requirements.txt && \
    if [ "$INSTALL_ML_DEPS" = "true" ]; then \
        pip install --no-cache-dir -r /app/api/requirements-ml.txt ; \
    fi

COPY api /app/api
COPY scripts /app/scripts

RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        python /app/scripts/download_models.py || echo "Model download script ended with non-zero status" ; \
    fi

EXPOSE 8008

WORKDIR /app/api

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]

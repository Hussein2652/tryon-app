#!/usr/bin/env bash
set -euo pipefail

if [[ "${DOWNLOAD_MODELS_ON_START:-0}" = "1" || "${DOWNLOAD_MODELS_ON_START:-false}" = "true" ]]; then
  echo "[entrypoint] Checking/downloading models into ${TRYON_MODELS_DIR:-/models}..."
  python /app/scripts/download_models.py || echo "[entrypoint] Model download script failed; continuing"
fi

exec "$@"


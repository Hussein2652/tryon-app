#!/usr/bin/env bash
set -euo pipefail

if [[ "${DOWNLOAD_MODELS_ON_START:-0}" = "1" || "${DOWNLOAD_MODELS_ON_START:-false}" = "true" ]]; then
  echo "[entrypoint] Checking/downloading models into ${TRYON_MODELS_DIR:-/models}..."
  python /app/scripts/download_models.py || echo "[entrypoint] Model download script failed; continuing"
fi

# Ensure IDM-VTON venv is ready. Do this synchronously the first time so that
# /api/v2/tryon works on the very first request. Subsequent starts are instant
# thanks to the stamp and persisted /third_party volume.
python /app/scripts/bootstrap_idm_vton.py || echo "[entrypoint] IDM-VTON venv bootstrap failed; using fallback"

exec "$@"

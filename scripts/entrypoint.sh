#!/usr/bin/env bash
set -euo pipefail

if [[ "${DOWNLOAD_MODELS_ON_START:-0}" = "1" || "${DOWNLOAD_MODELS_ON_START:-false}" = "true" ]]; then
  MODELS_DIR="${TRYON_MODELS_DIR:-/models}"
  LOG_FILE="${MODELS_DIR}/_downloads.log"
  PID_FILE="${MODELS_DIR}/_downloads.pid"
  echo "[entrypoint] Spawning non-blocking model sync -> ${LOG_FILE}"
  mkdir -p "${MODELS_DIR}"
  # Truncate previous log and start in background; keep PID for status endpoints
  : > "${LOG_FILE}"
  ( nohup python /app/scripts/download_models.py >> "${LOG_FILE}" 2>&1 & echo $! > "${PID_FILE}" ) || echo "[entrypoint] Model download script spawn failed; continuing"
fi

# Optionally prepare IDM-VTON venv; can be disabled to use image deps only.
if [[ "${IDM_VTON_DISABLE_VENV:-0}" != "1" && "${IDM_VTON_DISABLE_VENV:-false}" != "true" ]]; then
  # Ensure IDM-VTON venv is ready. Do this synchronously the first time so that
  # /api/v2/tryon works on the very first request. Subsequent starts are instant
  # thanks to the stamp and persisted /third_party volume.
  python /app/scripts/bootstrap_idm_vton.py || echo "[entrypoint] IDM-VTON venv bootstrap failed; using fallback"
else
  echo "[entrypoint] IDM-VTON venv disabled (using image dependencies)"
  # Best-effort cleanup to avoid confusion
  rm -rf /third_party/.venv_idm_iso /third_party/.venv_idm 2>/dev/null || true
fi

exec "$@"

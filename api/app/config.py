import os
from pathlib import Path

# Base path for the API package
BASE_PATH = Path(__file__).resolve().parent

MODEL_VERSION = os.environ.get("TRYON_MODEL_VERSION", "v0")

ENGINE_MODE = os.environ.get("TRYON_ENGINE", "placeholder").lower()

STABLEVITON_CKPT_DIR = Path(
    os.environ.get("STABLEVITON_CKPT_DIR", "/models/stableviton")
)
CONTROLNET_OPENPOSE_DIR = Path(
    os.environ.get("CONTROLNET_OPENPOSE_DIR", "/models/controlnet/openpose")
)
INSTANTID_DIR = Path(os.environ.get("INSTANTID_DIR", "/models/instantid"))
SCHP_WEIGHTS = Path(os.environ.get("SCHP_WEIGHTS", "/models/schp/lip.pth"))
USE_FP16 = os.environ.get("TRYON_USE_FP16", "1") == "1"
MAX_RENDER_RES = int(os.environ.get("TRYON_MAX_RES", "768"))

_DEFAULT_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

_cors_env = os.environ.get("TRYON_CORS_ORIGINS")
if _cors_env:
    CORS_ALLOW_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
else:
    CORS_ALLOW_ORIGINS = _DEFAULT_CORS_ORIGINS

CORS_ALLOW_ORIGIN_REGEX = os.environ.get(
    "TRYON_CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
)

# Directory where generated artifacts (e.g., try-on previews) are written.
OUTPUTS_DIR = Path(
    os.environ.get("TRYON_OUTPUTS_DIR", BASE_PATH.parent / "outputs")
)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Directory where temporary uploads can be staged if needed.
UPLOADS_DIR = Path(
    os.environ.get("TRYON_UPLOADS_DIR", BASE_PATH.parent / "uploads")
)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# Cache directory (metadata, serialized preprocessing artifacts, etc.)
CACHE_DIR = Path(os.environ.get("TRYON_CACHE_DIR", BASE_PATH.parent / "cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Simple log directory for session metadata (optional)
LOG_DIR = Path(os.environ.get("TRYON_LOG_DIR", BASE_PATH.parent / "logs"))
LOG_DIR.mkdir(exist_ok=True, parents=True)

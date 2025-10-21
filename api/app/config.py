import os
from pathlib import Path

# Base path for the API package
BASE_PATH = Path(__file__).resolve().parent

MODEL_VERSION = os.environ.get("TRYON_MODEL_VERSION", "v0")

ENGINE_MODE = os.environ.get("TRYON_ENGINE", "placeholder").lower()

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

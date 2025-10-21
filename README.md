# TryOn App (P0)

Local FastAPI implementation that exposes the `/size/recommend` and `/tryon/preview` endpoints described in the product spec. A `/healthz` probe is available for readiness/liveness checks.

## Layout

- `api/app/main.py` – FastAPI application definition and routes.
- `api/app/sizing.py` – Deterministic sizing rule engine + Pydantic schemas.
- `api/app/tryon_pipeline.py` – Engine-aware try-on pipeline (placeholder or StableVITON adapter).
- `api/app/config.py` – Local directories for generated artifacts.
- `api/app/engines/stableviton_adapter.py` – StableVITON adapter with a deterministic compositor baseline (non‑placeholder). Swap in the real engine when available.

## Getting started

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional ML stack (StableVITON, SCHP, etc.)
# pip install -r requirements-ml.txt
uvicorn app.main:app --reload --port 8008
```

## Run with Docker

```bash
docker build -t tryon-api .
# Install ML dependencies during build (optional):
# docker build -t tryon-api --build-arg INSTALL_ML_DEPS=true .
docker run --rm -p 8008:8008 \
  -e TRYON_OUTPUTS_DIR=/tmp/tryon/outputs \
  -e TRYON_UPLOADS_DIR=/tmp/tryon/uploads \
  -v $(pwd)/outputs:/tmp/tryon/outputs \
  -v $(pwd)/uploads:/tmp/tryon/uploads \
  tryon-api
```

The container exposes the API under `http://localhost:8008`. Mounting host directories is optional but keeps generated frames accessible on the host.

### Full stack (API + web harness)

```bash
docker compose up --build
```

The API will listen on `localhost:8008` and the React harness on `localhost:5173` (proxying API calls).

### Sample requests

Sample sizing payloads live under `samples/`.

```bash
curl -X POST http://localhost:8008/size/recommend \
  -H "Content-Type: application/json" \
  -d @samples/size_top.json
```

```bash
curl -X POST http://localhost:8008/tryon/preview \\
  -F "user_photo=@/path/to/photo.jpg" \\
  -F "garment_front=@/path/to/garment.png" \\
  -F "sku=SKU123" \\
  -F "size=M"
```

Generated frames are written under `api/outputs/` and served at `/outputs/...`.

Set `TRYON_ENGINE=stableviton` to route `/tryon/preview` through the StableVITON adapter. If the full ML stack is not present, the adapter produces non‑placeholder composites (garment over user photo). When the official StableVITON pipeline is installed, you can wire it into the adapter’s `_infer_real` path.

### Tests

```bash
cd api
pytest
```

### Frontend harness (local)

```bash
cd sdk/web
npm install
npm run dev
```

The harness proxies API calls to `http://localhost:8008` by default.

### Engine configuration

Environment knobs (defaults in `api/app/config.py`):

- `STABLEVITON_CKPT_DIR` – base directory containing StableVITON weights/checkpoints.
- `CONTROLNET_OPENPOSE_DIR` – ControlNet OpenPose checkpoint path.
- `INSTANTID_DIR` – InstantID assets for identity preservation.
- `SCHP_WEIGHTS` – Semantic human parsing weights (LIP/CIHP).
- `TRYON_USE_FP16` – `"1"` to enable fp16 inference when available.
- `TRYON_MAX_RES` – Max render resolution (e.g., 768 or 1024).
- `TRYON_CACHE_DIR`, `TRYON_LOG_DIR` – content cache + structured logs.
- `STABLEVITON_SHAREPOINT_URL` – optional direct download link for the official StableVITON checkpoint (SharePoint).
- `SCHP_DRIVE_URL` – Google Drive URL for SCHP weights (defaults to the LIP checkpoint).
- `INSTANTID_ANTELOPE_URL` – optional override for the InsightFace `antelopev2` bundle.

### Auto-downloading model assets

Set the Docker build arguments `INSTALL_ML_DEPS=true` and `DOWNLOAD_MODELS=true` to pull the ML stack and attempt model downloads during the image build. Example Compose snippet:

```yaml
services:
  api:
    build:
      context: .
      args:
        INSTALL_ML_DEPS: "true"
        DOWNLOAD_MODELS: "true"
    environment:
      - STABLEVITON_SHAREPOINT_URL=<your SharePoint link>
```

Assets hosted behind authentication (e.g., StableVITON SharePoint, SMPL-X) still require a valid link or a mounted volume; the script logs a warning when it cannot download them.

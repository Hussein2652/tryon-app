# TryOn App (P0)

Local FastAPI implementation that exposes the `/size/recommend` and `/tryon/preview` endpoints described in the product spec. A `/healthz` probe is available for readiness/liveness checks.

## Layout

- `api/app/main.py` – FastAPI application definition and routes.
- `api/app/sizing.py` – Deterministic sizing rule engine + Pydantic schemas.
- `api/app/tryon_pipeline.py` – Engine-aware try-on pipeline (placeholder or StableVITON adapter).
- `api/app/config.py` – Local directories for generated artifacts.
- `api/app/engines/stableviton_adapter.py` – Lazy-loaded StableVITON adapter stub (swap in the real engine when available).

## Getting started

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8008
```

## Run with Docker

```bash
docker build -t tryon-api .
docker run --rm -p 8008:8008 \
  -e TRYON_OUTPUTS_DIR=/tmp/tryon/outputs \
  -e TRYON_UPLOADS_DIR=/tmp/tryon/uploads \
  -v $(pwd)/outputs:/tmp/tryon/outputs \
  -v $(pwd)/uploads:/tmp/tryon/uploads \
  tryon-api
```

The container exposes the API under `http://localhost:8008`. Mounting host directories is optional but keeps generated frames accessible on the host.

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

Set `TRYON_ENGINE=stableviton` to route `/tryon/preview` through the StableVITON adapter once the real engine dependencies are installed. Otherwise, the placeholder renderer remains active.

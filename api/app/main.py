from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    CORS_ALLOW_ORIGIN_REGEX,
    CORS_ALLOW_ORIGINS,
    ENGINE_MODE,
    MODEL_VERSION,
    OUTPUTS_DIR,
)
from .cache import delete_cache, read_cache
from .metrics import increment, observe_latency, snapshot
from .moderation import evaluate_user_photo
from .sizing import (
    SizeRecommendRequest,
    SizingEngine,
    build_response,
)
from .tryon_pipeline import TryOnPipeline


logger = logging.getLogger(__name__)


def get_sizing_engine() -> SizingEngine:
    return SizingEngine()


def get_tryon_pipeline() -> TryOnPipeline:
    return TryOnPipeline(model_version=MODEL_VERSION, engine_mode=ENGINE_MODE)


app = FastAPI(
    title="TryOn API",
    version="0.1.0",
    description="API surface for sizing recommendations and virtual try-on previews.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    label = f"{request.method} {request.url.path}"
    increment("requests_total", label)
    observe_latency("http_request_seconds", label, duration)
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return JSONResponse(status_code=status.HTTP_200_OK, content=snapshot())


@app.post("/size/recommend")
async def recommend_size(
    request: SizeRecommendRequest,
    engine: SizingEngine = Depends(get_sizing_engine),
):
    try:
        recommendation = engine.recommend(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    response = build_response(request, recommendation)
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.model_dump())


@app.post("/tryon/preview")
async def tryon_preview(
    user_photo: UploadFile = File(...),
    garment_front: UploadFile = File(...),
    garment_mask: Optional[UploadFile] = File(default=None),
    sku: Optional[str] = Form(default=None),
    size: Optional[str] = Form(default=None),
    pose_set: Optional[str] = Form(default=None),
    pipeline: TryOnPipeline = Depends(get_tryon_pipeline),
):
    user_photo_bytes = await user_photo.read()
    garment_front_bytes = await garment_front.read()
    garment_mask_bytes = await garment_mask.read() if garment_mask else None

    if not user_photo_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_photo is empty.")
    if not garment_front_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="garment_front is empty.",
        )

    qa_issues = evaluate_user_photo(user_photo_bytes)
    if qa_issues:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[issue.__dict__ for issue in qa_issues],
        )

    result = pipeline.run(
        user_photo=user_photo_bytes,
        garment_front=garment_front_bytes,
        garment_mask=garment_mask_bytes,
        sku=sku,
        size=size,
        pose_set=pose_set,
    )

    images_payload = [
        f"/outputs/{path.name}"
        for path in result.image_paths
    ]

    payload = {
        "ok": True,
        "cache_key": result.cache_key,
        "images": images_payload,
        "count": len(result.image_paths),
        "frame_scores": result.frame_scores,
        "confidence_avg": round(result.confidence_avg, 2),
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


def _safe_output_path(path_str: str) -> Optional[Path]:
    path = Path(path_str)
    try:
        path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        logger.warning("Attempted to remove file outside outputs directory: %s", path)
        return None
    return path


@app.delete("/tryon/{cache_key}")
async def delete_tryon_session(cache_key: str):
    metadata = read_cache(cache_key, "tryon")
    deleted_meta = delete_cache(cache_key, "tryon")
    removed_files = []
    if metadata:
        for path_str in metadata.get("image_paths", []):
            candidate = _safe_output_path(path_str)
            if candidate and candidate.exists():
                candidate.unlink()
                removed_files.append(candidate.name)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "ok": True,
            "cache_key": cache_key,
            "metadata_removed": deleted_meta,
            "files_removed": removed_files,
        },
    )

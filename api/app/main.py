from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
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
from .sizing import (
    SizeRecommendRequest,
    SizingEngine,
    build_response,
)
from .tryon_pipeline import TryOnPipeline


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


@app.get("/healthz")
async def healthz():
    return {"ok": True}


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

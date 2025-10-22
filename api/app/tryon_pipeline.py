from __future__ import annotations

import io
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from PIL import Image, ImageDraw, ImageFont

from .cache import write_cache
from .config import ENGINE_MODE, OUTPUTS_DIR, NUM_TRYON_FRAMES
from . import config as app_config
from .metrics import observe_latency
from .cv import PreprocessManager
from .utils import stable_content_hash


logger = logging.getLogger(__name__)

DEFAULT_POSE_SET = "SET_A"
POSE_COUNT = NUM_TRYON_FRAMES

ENGINE_PLACEHOLDER = "placeholder"
ENGINE_STABLEVITON = "stableviton"


@dataclass
class TryOnResult:
    cache_key: str
    image_paths: List[Path]
    frame_scores: List[float]
    confidence_avg: float
    debug_paths: Optional[dict]


class TryOnPipeline:
    """Try-on pipeline with pluggable engines."""

    def __init__(
        self,
        *,
        model_version: str = "v0",
        engine_mode: Optional[str] = None,
    ):
        self.model_version = model_version
        self.engine_mode = (engine_mode or ENGINE_MODE or ENGINE_PLACEHOLDER).lower()
        if self.engine_mode not in {ENGINE_PLACEHOLDER, ENGINE_STABLEVITON}:
            logger.warning("Unsupported TRYON_ENGINE %s; falling back to placeholder.", self.engine_mode)
            self.engine_mode = ENGINE_PLACEHOLDER
        self.preprocessor = PreprocessManager()

    def run(
        self,
        *,
        user_photo: bytes,
        garment_front: bytes,
        garment_mask: Optional[bytes],
        sku: Optional[str],
        size: Optional[str],
        pose_set: Optional[str],
        diffusion_params: Optional[Dict[str, Any]] = None,
        debug: Optional[bool] = None,
    ) -> TryOnResult:
        pose_set_key = pose_set or DEFAULT_POSE_SET
        cache_key = self._build_cache_key(
            user_photo=user_photo,
            garment_front=garment_front,
            garment_mask=garment_mask,
            sku=sku,
            size=size,
            pose_set=pose_set_key,
            diffusion_params=diffusion_params,
        )

        artifacts = None
        pose_map_bytes = None
        if self.engine_mode == ENGINE_STABLEVITON:
            artifacts = self.preprocessor.run(user_photo)
            if artifacts.pose.pose_map is not None:
                pose_map_bytes = self._encode_image(artifacts.pose.pose_map)

        start_time = time.perf_counter()

        debug_paths: Optional[dict] = None
        if self.engine_mode == ENGINE_STABLEVITON:
            image_paths = self._ensure_stableviton_outputs(
                cache_key=cache_key,
                user_photo=user_photo,
                garment_front=garment_front,
                garment_mask=garment_mask,
                pose_set=pose_set_key,
                sku=sku,
                size=size,
                pose_map=pose_map_bytes,
                artifacts=artifacts,
                diffusion_params=diffusion_params,
            )
            if (debug or False) and artifacts is not None:
                debug_paths = {}
                if artifacts.pose.pose_map is not None:
                    pose_path = OUTPUTS_DIR / f"tryon_{cache_key}_pose.png"
                    if not pose_path.exists():
                        artifacts.pose.pose_map.save(pose_path)
                    debug_paths["pose"] = pose_path
                if artifacts.segmentation.alpha_mask is not None:
                    alpha_path = OUTPUTS_DIR / f"tryon_{cache_key}_alpha.png"
                    if not alpha_path.exists():
                        artifacts.segmentation.alpha_mask.save(alpha_path)
                    debug_paths["alpha"] = alpha_path
        else:
            image_paths = self._ensure_placeholder_outputs(
                cache_key=cache_key,
                pose_set=pose_set_key,
                sku=sku,
                size=size,
            )

        frame_scores = self._generate_frame_scores(len(image_paths))
        confidence_avg = sum(frame_scores) / len(frame_scores) if frame_scores else 0.0

        observe_latency(
            "tryon_pipeline_seconds",
            f"engine={self.engine_mode}",
            time.perf_counter() - start_time,
        )

        write_cache(
            cache_key,
            namespace="tryon",
            payload={
                "engine": self.engine_mode,
                "model_version": self.model_version,
                "sku": sku,
                "size": size,
                "pose_set": pose_set_key,
                "image_paths": [str(path) for path in image_paths],
                "frame_scores": frame_scores,
            },
        )

        return TryOnResult(
            cache_key=cache_key,
            image_paths=image_paths,
            frame_scores=frame_scores,
            confidence_avg=round(confidence_avg, 2),
            debug_paths=debug_paths,
        )

    def _build_cache_key(
        self,
        *,
        user_photo: bytes,
        garment_front: bytes,
        garment_mask: Optional[bytes],
        sku: Optional[str],
        size: Optional[str],
        pose_set: str,
        diffusion_params: Optional[Dict[str, Any]],
    ) -> str:
        parts = [
            f"v={self.model_version}".encode("utf-8"),
            user_photo,
            garment_front,
        ]
        if garment_mask:
            parts.append(garment_mask)
        if sku:
            parts.append(f"sku={sku}".encode("utf-8"))
        if size:
            parts.append(f"size={size}".encode("utf-8"))
        parts.append(f"pose_set={pose_set}".encode("utf-8"))
        # Include engine mode and readiness in the signature so that when
        # diffusion assets become available, we produce a new cache key and
        # regenerate frames instead of reusing compositor outputs.
        parts.append(f"engine={self.engine_mode}".encode("utf-8"))
        parts.append(self._engine_readiness_sig().encode("utf-8"))
        parts.extend(self._diffusion_sig_parts(diffusion_params))
        return stable_content_hash(parts, prefix="tryon:")

    @staticmethod
    def _diffusion_sig_parts(params: Optional[Dict[str, Any]]) -> List[bytes]:
        sig: List[bytes] = []
        if not params:
            # Include environment defaults so cache updates when env tuning changes
            sig.append(f"steps={app_config.DIFFUSION_STEPS}".encode("utf-8"))
            sig.append(f"guidance={app_config.DIFFUSION_GUIDANCE}".encode("utf-8"))
            sig.append(f"strength={app_config.DIFFUSION_STRENGTH}".encode("utf-8"))
            return sig
        for key in ("steps", "guidance", "strength"):
            if key in params and params[key] is not None:
                sig.append(f"{key}={params[key]}".encode("utf-8"))
        return sig

    def _engine_readiness_sig(self) -> str:
        if self.engine_mode == ENGINE_STABLEVITON:
            sd_ok = (app_config.SD15_MODEL_DIR / ".complete").exists()
            cn_ok = (app_config.CONTROLNET_OPENPOSE_DIR / "config.json").exists()
            return f"sv_ready={int(sd_ok and cn_ok)}"
        return "sv_ready=0"

    def _ensure_placeholder_outputs(
        self,
        *,
        cache_key: str,
        pose_set: str,
        sku: Optional[str],
        size: Optional[str],
    ) -> List[Path]:
        image_paths: List[Path] = []
        for pose_idx in range(POSE_COUNT):
            output_path = OUTPUTS_DIR / f"tryon_{cache_key}_{pose_idx + 1}.png"
            if not output_path.exists():
                self._write_placeholder_image(
                    output_path=output_path,
                    pose_index=pose_idx,
                    cache_key=cache_key,
                    sku=sku,
                    size=size,
                    pose_set=pose_set,
                )
            image_paths.append(output_path)
        return image_paths

    def _ensure_stableviton_outputs(
        self,
        *,
        cache_key: str,
        user_photo: bytes,
        garment_front: bytes,
        garment_mask: Optional[bytes],
        pose_set: str,
        sku: Optional[str],
        size: Optional[str],
        pose_map: Optional[bytes],
        artifacts: Optional[object],
        diffusion_params: Optional[Dict[str, Any]],
    ) -> List[Path]:
        try:
            from .engines.stableviton_adapter import run_stableviton
        except ImportError as exc:
            raise RuntimeError(
                "StableVITON adapter is not available. Ensure dependencies are installed."
            ) from exc

        masks = {"garment": garment_mask} if garment_mask else None
        if artifacts is not None:
            if artifacts.segmentation.alpha_mask is not None:
                encoded = self._encode_image(artifacts.segmentation.alpha_mask)
                masks = masks or {}
                masks["user_alpha"] = encoded
            if getattr(artifacts.segmentation, "face_mask", None) is not None:
                encoded = self._encode_image(artifacts.segmentation.face_mask)
                masks = masks or {}
                masks["face"] = encoded

        frames = run_stableviton(
            user_png=user_photo,
            garment_png=garment_front,
            pose_map=pose_map,
            masks=masks,
            pose_set=pose_set,
            sku=sku,
            size=size,
            artifacts=artifacts,
            diffusion_params=diffusion_params,
        )
        if not frames:
            raise RuntimeError("StableVITON adapter returned no frames.")

        image_paths: List[Path] = []
        for idx, frame_bytes in enumerate(frames, start=1):
            output_path = OUTPUTS_DIR / f"tryon_{cache_key}_{idx}.png"
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(frame_bytes)
            image_paths.append(output_path)
        return image_paths

    @staticmethod
    def _write_placeholder_image(
        *,
        output_path: Path,
        pose_index: int,
        cache_key: str,
        sku: Optional[str],
        size: Optional[str],
        pose_set: str,
    ) -> None:
        colors = [
            (66, 135, 245),
            (245, 163, 66),
            (126, 217, 87),
            (255, 99, 146),
            (148, 112, 255),
        ]
        background = colors[pose_index % len(colors)]
        image = Image.new("RGB", (768, 1024), background)
        draw = ImageDraw.Draw(image)

        lines = [
            "Virtual Try-On Preview",
            f"Pose {pose_index + 1} / {POSE_COUNT}",
            f"Cache {cache_key[:8]}…",
        ]
        if sku:
            lines.append(f"SKU {sku}")
        if size:
            lines.append(f"Size {size}")
        if pose_set:
            lines.append(f"Pose Set {pose_set}")
        lines.append(f"Engine {ENGINE_PLACEHOLDER}")

        y = 150
        font = ImageFont.load_default()
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                ((image.width - text_width) / 2, y),
                line,
                fill=(255, 255, 255),
                font=font,
            )
            y += text_height + 20

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")

    @staticmethod
    def _generate_frame_scores(count: int) -> List[float]:
        rng = random.Random(1234)
        base = 0.82
        return [round(base + rng.random() * 0.06, 2) for _ in range(count)]

    @staticmethod
    def _encode_image(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

from __future__ import annotations

import io
import logging
import math
import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

from .. import config

if TYPE_CHECKING:  # pragma: no cover - typing hint only
    from ..cv import PreprocessArtifacts


logger = logging.getLogger(__name__)

FALLBACK_FRAME_COUNT = 5


class StableVITONNotReady(RuntimeError):
    """Raised when StableVITON cannot run because dependencies are missing."""


@dataclass(frozen=True)
class EngineConfig:
    ckpt_dir: str
    controlnet_dir: str
    instantid_dir: str
    use_fp16: bool
    max_res: int

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.ckpt_dir,
            self.controlnet_dir,
            self.instantid_dir,
            self.use_fp16,
            self.max_res,
        )


@dataclass
class EngineInputs:
    user_png: bytes
    garment_png: bytes
    pose_map: Optional[bytes]
    masks: Optional[Dict[str, bytes]]
    pose_set: str
    sku: Optional[str]
    size: Optional[str]
    artifacts: Optional["PreprocessArtifacts"] = None


class StableVITONEngine:
    """Lazy-loaded StableVITON runner with graceful fallbacks.

    If the full ML stack is available, this can be extended to invoke the
    official StableVITON inference. When unavailable, a deterministic
    compositor overlays the garment over the user photo to produce
    non-placeholder previews.
    """

    def __init__(self, engine_cfg: EngineConfig) -> None:
        self.cfg = engine_cfg
        self._loaded = False
        self._load_lock = threading.Lock()
        self._model = None  # Placeholder for framework-specific objects.

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                # Prefer real stack if present; otherwise proceed without it.
                import torch  # noqa: F401  # type: ignore
                import diffusers  # noqa: F401  # type: ignore
                logger.info(
                    "StableVITON stack available (ckpt=%s, fp16=%s).",
                    self.cfg.ckpt_dir,
                    self.cfg.use_fp16,
                )
                # TODO: Wire actual StableVITON pipeline here when available.
                self._model = object()
                self._loaded = True
            except ImportError:
                logger.info(
                    "StableVITON ML deps not present; using compositor baseline."
                )
                # Use compositor path (still non-placeholder output)
                self._model = None
                self._loaded = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("StableVITON initialization failed: %s", exc)
                raise StableVITONNotReady(str(exc))

    def run(self, inputs: EngineInputs) -> List[bytes]:
        self.ensure_loaded()
        try:
            if self._model is not None:
                return self._infer_real(inputs)
        except StableVITONNotReady:
            # Fall back to compositor if real pipeline signals not ready.
            logger.warning("StableVITON not ready; falling back to compositor.")
        return self._infer_compositor(inputs)

    def _infer_real(self, inputs: EngineInputs) -> List[bytes]:
        # Attempt a ControlNet OpenPose + SD1.5 img2img render using local assets.
        try:
            import torch  # type: ignore
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
                UniPCMultistepScheduler,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise StableVITONNotReady(f"Diffusers/Torch not available: {exc}") from exc

        # Validate assets
        from pathlib import Path

        controlnet_path = Path(self.cfg.controlnet_dir) / "control_v11p_sd15_openpose.safetensors"
        sd15_dir = config.SD15_MODEL_DIR
        if not controlnet_path.exists():
            raise StableVITONNotReady(f"Missing ControlNet OpenPose at {controlnet_path}")
        if not (sd15_dir.exists() and (sd15_dir / "model_index.json").exists()):
            raise StableVITONNotReady(
                f"Missing SD1.5 model in diffusers format at {sd15_dir}."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (self.cfg.use_fp16 and device == "cuda") else torch.float32

        controlnet = ControlNetModel.from_single_file(str(controlnet_path), torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(sd15_dir),
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        try:
            pipe.disable_progress_bar()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        user_img = Image.open(io.BytesIO(inputs.user_png)).convert("RGB")
        if inputs.pose_map:
            control_image = Image.open(io.BytesIO(inputs.pose_map)).convert("RGB")
        elif inputs.artifacts and getattr(inputs.artifacts.pose, "pose_map", None) is not None:
            control_image = inputs.artifacts.pose.pose_map.convert("RGB")
        else:
            control_image = Image.new("RGB", user_img.size, (0, 0, 0))
        control_image = control_image.resize(user_img.size)

        prompt_bits = [
            "photo of a person wearing a shirt",
            "realistic fabric, natural drape, clean background",
        ]
        if inputs.size:
            prompt_bits.append(f"size {inputs.size}")
        prompt = ", ".join(prompt_bits)
        negative = "blurry, lowres, deformed, extra limbs, bad hands, text"

        frames: List[bytes] = []
        count = max(1, FALLBACK_FRAME_COUNT)
        base_seed = 24681357
        for i in range(count):
            g = torch.Generator(device=device).manual_seed(base_seed + i)
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=user_img,
                    control_image=control_image,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    strength=0.35,
                    generator=g,
                )
            except Exception as exc:  # pylint: disable=broad-except
                raise StableVITONNotReady(f"Diffusion inference failed: {exc}") from exc
            image = result.images[0]
            image = self._clamp_resolution(image)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            frames.append(buf.getvalue())

        return frames

    def _infer_compositor(self, inputs: EngineInputs) -> List[bytes]:
        """Produce non-placeholder previews by compositing garment onto user.

        Notes:
        - Uses provided garment mask if available; otherwise uses the garment
          alpha channel as mask when present.
        - Places the garment near the upper-torso region with light jitter to
          simulate pose variation.
        - Intentionally conservative to avoid unrealistic artifacts.
        """
        user_img = Image.open(io.BytesIO(inputs.user_png)).convert("RGB")
        garment_img = Image.open(io.BytesIO(inputs.garment_png))
        if garment_img.mode not in ("RGBA", "LA"):
            garment_img = garment_img.convert("RGBA")

        # Derive mask: prefer explicit garment mask; otherwise alpha channel.
        mask_img: Optional[Image.Image] = None
        if inputs.masks and "garment" in inputs.masks and inputs.masks["garment"]:
            mask_img = Image.open(io.BytesIO(inputs.masks["garment"])).convert("L")
        elif garment_img.mode == "RGBA":
            mask_img = garment_img.getchannel("A")
        else:
            # Luminance-based heuristic mask
            mask_img = garment_img.convert("L")
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1)) if mask_img else None

        # Determine target size/position: center-top ~65% of user width
        u_w, u_h = user_img.size
        target_w = int(min(self.cfg.max_res, u_w) * 0.65)
        scale = target_w / max(1, garment_img.width)
        target_h = int(garment_img.height * scale)
        garment_resized = garment_img.resize((target_w, target_h), Image.LANCZOS)
        mask_resized = (
            mask_img.resize((target_w, target_h), Image.LANCZOS) if mask_img else None
        )

        # Anchor near upper third; add small jitter per frame
        base_x = (u_w - target_w) // 2
        base_y = max(0, int(u_h * 0.25) - target_h // 8)

        rng = random.Random(123)
        frames: List[bytes] = []
        count = max(1, FALLBACK_FRAME_COUNT)
        for _ in range(count):
            dx = int(rng.uniform(-u_w * 0.02, u_w * 0.02))
            dy = int(rng.uniform(-u_h * 0.015, u_h * 0.02))
            angle = rng.uniform(-2.0, 2.0)

            # Rotate garment slightly around center
            rotated = garment_resized.rotate(angle, resample=Image.BICUBIC, expand=False)
            rotated_mask = (
                mask_resized.rotate(angle, resample=Image.BICUBIC, expand=False)
                if mask_resized
                else None
            )

            composite = user_img.copy()
            paste_xy = (max(0, base_x + dx), max(0, base_y + dy))
            if rotated_mask is not None:
                composite.paste(rotated, paste_xy, rotated_mask)
            else:
                composite.paste(rotated, paste_xy)

            # Clamp to max resolution if needed
            composite = self._clamp_resolution(composite)

            buf = io.BytesIO()
            composite.save(buf, format="PNG")
            frames.append(buf.getvalue())

        return frames

    def _clamp_resolution(self, image: Image.Image) -> Image.Image:
        max_side = int(self.cfg.max_res)
        w, h = image.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale >= 0.999:
            return image
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.LANCZOS)


_ENGINE_LOCK = threading.Lock()
_ENGINE_SINGLETON: Optional[StableVITONEngine] = None
_ENGINE_SIGNATURE: Optional[Tuple[Any, ...]] = None


def _get_engine(engine_cfg: EngineConfig) -> StableVITONEngine:
    global _ENGINE_SINGLETON, _ENGINE_SIGNATURE
    signature = engine_cfg.signature()
    with _ENGINE_LOCK:
        if _ENGINE_SINGLETON is None or _ENGINE_SIGNATURE != signature:
            _ENGINE_SINGLETON = StableVITONEngine(engine_cfg)
            _ENGINE_SIGNATURE = signature
    return _ENGINE_SINGLETON


def run_stableviton(
    *,
    user_png: bytes,
    garment_png: bytes,
    pose_map: Optional[bytes] = None,
    masks: Optional[Dict[str, bytes]] = None,
    pose_set: str = "SET_A",
    sku: Optional[str] = None,
    size: Optional[str] = None,
    artifacts: Optional["PreprocessArtifacts"] = None,
) -> List[bytes]:
    """Run StableVITON (or compositor) to generate pose-aligned garment renders."""

    engine_cfg = EngineConfig(
        ckpt_dir=str(config.STABLEVITON_CKPT_DIR),
        controlnet_dir=str(config.CONTROLNET_OPENPOSE_DIR),
        instantid_dir=str(config.INSTANTID_DIR),
        use_fp16=config.USE_FP16,
        max_res=config.MAX_RENDER_RES,
    )
    engine = _get_engine(engine_cfg)
    inputs = EngineInputs(
        user_png=user_png,
        garment_png=garment_png,
        pose_map=pose_map,
        masks=masks,
        pose_set=pose_set,
        sku=sku,
        size=size,
        artifacts=artifacts,
    )
    return engine.run(inputs)

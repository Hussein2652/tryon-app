from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TryOnV2Config:
    models_dir: Path
    enable_instantid: bool = False
    device_preference: str = "auto"  # "cuda" | "cpu" | "auto"


class TryOnV2Engine:
    """V2 try-on engine.

    Primary path: IDM-VTON wrapper (no-training). If unavailable, fallback to
    SDXL inpainting + ControlNet + optional IP-Adapter Face.
    """

    def __init__(self, cfg: TryOnV2Config) -> None:
        self.cfg = cfg
        self._idm_ready = False
        self._sdxl_ready = False
        self._init_once()

    def _init_once(self) -> None:
        # Detect IDM-VTON assets
        idm_ckpt = self.cfg.models_dir / "idm_vton" / "ckpt"
        if idm_ckpt.exists():
            self._idm_ready = True
        # Detect SDXL assets
        sdxl_inp = self.cfg.models_dir / "sdxl-inpaint" / "model_index.json"
        cn_openpose = self.cfg.models_dir / "controlnets" / "openpose-sdxl" / "model_index.json"
        if sdxl_inp.exists() and cn_openpose.exists():
            self._sdxl_ready = True
        logger.info("tryon_v2 init: idm_ready=%s, sdxl_ready=%s", self._idm_ready, self._sdxl_ready)

    def run(
        self,
        *,
        person: Image.Image,
        cloth: Image.Image,
        category: str = "upper_body",
        steps: int = 30,
        guidance: float = 5.0,
        seed: int = 42,
    ) -> Image.Image:
        # Try IDM-VTON first
        if self._idm_ready:
            try:
                return self._run_idm_vton(person=person, cloth=cloth, category=category, steps=steps, guidance=guidance, seed=seed)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("IDM-VTON path failed: %s; falling back to SDXL.", exc)

        # Fallback SDXL
        if self._sdxl_ready:
            try:
                return self._run_sdxl_inpaint(person=person, cloth=cloth, steps=steps, guidance=guidance, seed=seed)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("SDXL fallback failed: %s; falling back to compositor.", exc)

        # Last resort: simple compositor (no training)
        return self._composite(person, cloth)

    # ----------------------
    # Primary: IDM-VTON path
    # ----------------------
    def _run_idm_vton(
        self,
        *,
        person: Image.Image,
        cloth: Image.Image,
        category: str,
        steps: int,
        guidance: float,
        seed: int,
    ) -> Image.Image:
        third_party = Path("/app/third_party/idm_vton")
        if not third_party.exists():
            raise RuntimeError("IDM-VTON repo not present.")
        # Lazy import to avoid hard dependency when unused
        try:
            import sys
            sys.path.insert(0, str(third_party))
            # IDM-VTON demo packs a gradio app; import minimal callable if available
            # Placeholder hook: implement exact import mapping when repo present.
            # For now, raise to trigger fallback if not wired.
            raise RuntimeError("IDM-VTON callable not wired yet.")
        finally:
            if str(third_party) in sys.path:
                sys.path.remove(str(third_party))

    # ------------------------------
    # Fallback: SDXL inpaint ControlNet
    # ------------------------------
    def _run_sdxl_inpaint(
        self,
        *,
        person: Image.Image,
        cloth: Image.Image,
        steps: int,
        guidance: float,
        seed: int,
    ) -> Image.Image:
        from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        base_dir = self.cfg.models_dir
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            str(base_dir / "sdxl-inpaint"), torch_dtype=dtype, variant="fp16" if dtype==torch.float16 else None
        )
        cn_openpose = ControlNetModel.from_pretrained(str(base_dir / "controlnets/openpose-sdxl"), torch_dtype=dtype)

        try:
            pipe.controlnet = [cn_openpose]
        except Exception:
            pipe.controlnet = cn_openpose

        pipe = pipe.to(device)

        # Build a conservative inpaint mask over torso region (exclude face)
        w, h = person.size
        import numpy as np
        mask = Image.new("L", (w, h), 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(mask)
        y_top = int(h * 0.22)
        y_bot = int(h * 0.85)
        draw.rectangle([(int(w*0.2), y_top), (int(w*0.8), y_bot)], fill=255)

        # Pose map placeholder: a black image (ControlNet still loads)
        pose = Image.new("RGB", (w, h), (0, 0, 0))

        # Simple composited init (cloth pasted on torso) to help inpainting
        init = self._composite(person, cloth)

        g = torch.Generator(device=device).manual_seed(int(seed))
        result = pipe(
            prompt="photo of a person wearing a shirt, realistic fabric, natural drape",
            negative_prompt="blurry, lowres, deformed, extra limbs, bad hands, text",
            image=init,
            mask_image=mask,
            control_image=pose,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=0.45,
            generator=g,
        )
        return result.images[0]

    # ----------------------
    # Utility: simple paste
    # ----------------------
    def _composite(self, person: Image.Image, cloth: Image.Image) -> Image.Image:
        person = person.convert("RGB")
        cloth = cloth.convert("RGBA") if cloth.mode != "RGBA" else cloth
        w, h = person.size
        scale = int(w * 0.5) / max(1, cloth.width)
        target = cloth.resize((int(cloth.width*scale), int(cloth.height*scale)), Image.LANCZOS)
        x = (w - target.width) // 2
        y = max(0, int(h * 0.30) - target.height // 8)
        out = person.copy()
        out.paste(target, (x, y), target.split()[-1])
        return out


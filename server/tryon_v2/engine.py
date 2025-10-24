from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
        third_party = Path("/third_party/idm_vton") if Path("/third_party/idm_vton").exists() else Path("/app/third_party/idm_vton")
        if not third_party.exists():
            raise RuntimeError("IDM-VTON repo not present.")
        # Try to import the exact Gradio demo entrypoint: gradio_demo/app.py
        import sys, importlib.util, os
        sys.path.insert(0, str(third_party))
        try:
            module_path = third_party / "gradio_demo" / "app.py"
            if not module_path.exists():
                raise RuntimeError("gradio_demo/app.py not found in IDM-VTON repo")

            spec = importlib.util.spec_from_file_location("idm_vton_gradio_app", str(module_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load IDM-VTON gradio module")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]

            # Export ckpt dir env for the demo (common pattern)
            os.environ.setdefault("IDM_VTON_CKPT_DIR", str(ckpt))

            # Preferred callable order: process, predict
            fn = None
            for name in ("process", "predict"):
                if hasattr(mod, name):
                    fn = getattr(mod, name)
                    break
            if fn is None:
                raise RuntimeError("IDM-VTON gradio callable not found (process/predict)")

            # Attempt multiple kwarg schemes to match various commits
            variants = [
                dict(person_image=person, cloth_image=cloth, category=category, steps=steps, guidance=guidance, seed=seed),
                dict(human_img=person, cloth_img=cloth, category=category, steps=steps, cfg=guidance, seed=seed),
                dict(im=person, c=cloth, cate=category, num_steps=steps, cfg_scale=guidance, seed=seed),
            ]
            for kwargs in variants:
                try:
                    allowed = {k: v for k, v in kwargs.items() if k in getattr(fn, "__code__").co_varnames}
                    out = fn(**allowed)
                    if isinstance(out, Image.Image):
                        return out
                    if isinstance(out, (list, tuple)) and out and isinstance(out[0], Image.Image):
                        return out[0]
                except Exception as exc:
                    logger.debug("IDM-VTON call variant failed: %s", exc)
                    continue
            raise RuntimeError("IDM-VTON callable invocation failed for all variants")
        finally:
            try:
                sys.path.remove(str(third_party))
            except ValueError:
                pass

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
        cn_softedge = None
        softedge_path = base_dir / "controlnets/softedge-sdxl"
        if (softedge_path / "model_index.json").exists():
            try:
                cn_softedge = ControlNetModel.from_pretrained(str(softedge_path), torch_dtype=dtype)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("SoftEdge ControlNet failed to load: %s", exc)
        pipe = pipe.to(device)

        pipe = pipe.to(device)

        # Build a conservative inpaint mask over torso region (exclude face)
        w, h = person.size
        import numpy as np
        mask = Image.new("L", (w, h), 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(mask)
        y_top = int(h * 0.25)
        y_bot = int(h * 0.88)
        draw.rectangle([(int(w*0.22), y_top), (int(w*0.78), y_bot)], fill=255)

        # Pose/edge placeholders (black); ControlNets still load. You can replace
        # with real maps from DW-Pose/SoftEdge if desired.
        pose = Image.new("RGB", (w, h), (0, 0, 0))
        edge = Image.new("RGB", (w, h), (0, 0, 0))

        # Simple composited init (cloth pasted on torso) to help inpainting
        init = self._composite(person, cloth)

        g = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = {}
        # Attempt to apply IP-Adapter Face for identity if available
        try:
            ip_dir = base_dir / "ip_adapter" / "sdxl_models"
            weight = ip_dir / "ip-adapter_sdxl.bin"
            if weight.exists():
                pipe.load_ip_adapter(str(ip_dir.parent), subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                pipe.set_ip_adapter_scale(0.5)
                # Crude face crop: top center square
                face_crop = person.crop((int(w*0.3), int(h*0.05), int(w*0.7), int(h*0.45)))
                kwargs["ip_adapter_image"] = face_crop
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("IP-Adapter load failed: %s", exc)

        # ControlNet list and scales
        control_images: List[Image.Image] = [pose]
        scales: List[float] = [0.8]
        if cn_softedge is not None:
            control_images.append(edge)
            scales.append(0.6)
        try:
            pipe.controlnet = [cn for cn in [cn_openpose, cn_softedge] if cn is not None]
        except Exception:
            pipe.controlnet = cn_openpose

        result = pipe(
            prompt="photo of a person wearing a shirt, realistic fabric, natural drape",
            negative_prompt="blurry, lowres, deformed, extra limbs, bad hands, text",
            image=init,
            mask_image=mask,
            control_image=control_images if len(control_images) > 1 else control_images[0],
            controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=0.45,
            generator=g,
            **kwargs,
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

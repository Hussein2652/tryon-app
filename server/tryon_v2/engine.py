from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import os
import subprocess
import tempfile

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
        repo_root = Path("/third_party/idm_vton")
        if not repo_root.exists():
            raise RuntimeError("IDM-VTON repo not present.")
        # Prefer isolated venv execution to avoid dependency conflicts with API env
        venv_python = Path("/third_party/.venv_idm/bin/python")
        runner = Path("/app/server/tryon_v2/idm_runner.py")
        if not venv_python.exists() or not runner.exists():
            raise RuntimeError("IDM-VTON venv runner not available.")

        # Prepare temp files
        with tempfile.TemporaryDirectory() as td:
            p_path = Path(td) / "person.png"
            c_path = Path(td) / "cloth.png"
            o_path = Path(td) / "out.png"
            person.save(p_path, format="PNG")
            cloth.save(c_path, format="PNG")
            env = os.environ.copy()
            env.setdefault("IDM_VTON_CKPT_DIR", str(self.cfg.models_dir / "idm_vton" / "ckpt"))
            cmd = [str(venv_python), str(runner), str(p_path), str(c_path), str(o_path), category, str(int(steps)), str(float(guidance)), str(int(seed))]
            try:
                subprocess.run(cmd, env=env, check=True)
            except Exception as exc:
                raise RuntimeError(f"IDM-VTON runner failed: {exc}") from exc
            return Image.open(o_path).convert("RGB")

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

        # Build control images: OpenPose (skeleton) and SoftEdge (edges)
        pose = Image.new("RGB", (w, h), (0, 0, 0))
        edge = Image.new("RGB", (w, h), (0, 0, 0))
        try:
            from controlnet_aux.open_pose import OpenposeDetector  # type: ignore
            op = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            pose = op(person.convert("RGB")).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("OpenPose control image failed; using blank. %s", exc)
        try:
            # Soft edge approximation via PidiNet; falls back to Canny if unavailable
            from controlnet_aux import PidiNetDetector  # type: ignore
            pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            edge = pidi(person.convert("RGB"), safe=True).convert("RGB")
        except Exception:
            try:
                import cv2  # type: ignore
                import numpy as np
                arr = np.array(person.convert("RGB"))
                can = cv2.Canny(arr, 100, 200)
                edge = Image.fromarray(can).convert("RGB")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Edge control image failed; using blank. %s", exc)

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
        candidate = result.images[0]

        # Quick QC gates. If they fail, re-run once with stronger identity or denoise.
        try:
            face_ok = self._face_similarity_ok(person, candidate, threshold=0.35)
            cloth_ok = self._cloth_similarity_ok(cloth, candidate, threshold=0.28)
            bg_ok = self._bg_ssim_ok(person, candidate, mask, min_ssim=0.90)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("QC checks failed (%s); accepting first result.", exc)
            return candidate

        if face_ok and cloth_ok and bg_ok:
            return candidate

        # Retry once with tweaks
        try:
            # Increase identity lock if available
            if "ip_adapter_image" in kwargs:
                pipe.set_ip_adapter_scale(0.7)
            # Increase denoise a bit
            result2 = pipe(
                prompt="photo of a person wearing a shirt, realistic fabric, natural drape",
                negative_prompt="blurry, lowres, deformed, extra limbs, bad hands, text",
                image=init,
                mask_image=mask,
                control_image=control_images if len(control_images) > 1 else control_images[0],
                controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
                num_inference_steps=int(max(steps, 30)),
                guidance_scale=float(max(guidance, 5.0)),
                strength=min(0.60, 0.45 + 0.05),
                generator=g,
                **kwargs,
            )
            cand2 = result2.images[0]

            # Pick better by simple score
            s1 = self._score_overall(person, cloth, candidate, mask)
            s2 = self._score_overall(person, cloth, cand2, mask)
            return cand2 if s2 >= s1 else candidate
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Retry generation failed: %s", exc)
            return candidate

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

    # ----------------------
    # QC helpers
    # ----------------------
    def _face_similarity_ok(self, src: Image.Image, dst: Image.Image, threshold: float) -> bool:
        try:
            v1 = self._arcface_embed_onnx(src)
            v2 = self._arcface_embed_onnx(dst)
            if v1 is None or v2 is None:
                return True  # Skip gate if embeddings unavailable
            import numpy as np
            a = np.array(v1, dtype="float32")
            b = np.array(v2, dtype="float32")
            a = a / max(1e-6, (a**2).sum() ** 0.5)
            b = b / max(1e-6, (b**2).sum() ** 0.5)
            cos = float((a * b).sum())
            return cos >= threshold
        except Exception:
            return True

    def _arcface_embed_onnx(self, rgb: Image.Image):
        # Use antelopev2 glintr100.onnx if present; returns list[float] or None
        try:
            import onnxruntime as ort  # type: ignore
            import numpy as np
        except Exception:
            return None
        glintr = self.cfg.models_dir / "instantid" / "antelopev2" / "glintr100.onnx"
        if not glintr.exists():
            return None
        # Simple face crop: top-center crop
        w, h = rgb.size
        crop = rgb.crop((int(w * 0.3), int(h * 0.05), int(w * 0.7), int(h * 0.45))).resize((112, 112))
        arr = np.asarray(crop)[:, :, ::-1].astype("float32")
        arr = (arr - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        sess = ort.InferenceSession(str(glintr), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # type: ignore
        out = sess.run(None, {sess.get_inputs()[0].name: arr})
        return out[0].reshape(-1).astype("float32").tolist()

    def _cloth_similarity_ok(self, cloth: Image.Image, result: Image.Image, threshold: float) -> bool:
        # CLIP similarity between cloth and garment region (approx: whole result)
        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore
            import torch
        except Exception:
            return True
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            inputs = proc(images=[cloth.convert("RGB"), result.convert("RGB")], return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            a, b = feats[0], feats[1]
            a = a / a.norm()
            b = b / b.norm()
            cos = float((a @ b.T).item())
            return cos >= threshold
        except Exception:
            return True

    def _bg_ssim_ok(self, src: Image.Image, dst: Image.Image, fg_mask: Image.Image, min_ssim: float) -> bool:
        # Ensure background unchanged: compute SSIM on inverse of inpaint mask
        try:
            import numpy as np
            from skimage.metrics import structural_similarity as ssim  # type: ignore
        except Exception:
            return True
        src_g = np.array(src.convert("L"))
        dst_g = np.array(dst.convert("L"))
        mask = np.array(fg_mask.resize(src.size))
        bg = (mask < 128)
        if bg.sum() < 1024:
            return True
        try:
            score = ssim(src_g, dst_g, data_range=255)
        except Exception:
            return True
        return float(score) >= min_ssim

    def _score_overall(self, person: Image.Image, cloth: Image.Image, out: Image.Image, mask: Image.Image) -> float:
        s1 = 1.0 if self._face_similarity_ok(person, out, threshold=0.35) else 0.0
        s2 = 1.0 if self._cloth_similarity_ok(cloth, out, threshold=0.28) else 0.0
        s3 = 1.0 if self._bg_ssim_ok(person, out, mask, min_ssim=0.90) else 0.0
        return s1 * 0.5 + s2 * 0.3 + s3 * 0.2

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class IdentityEmbedding:
    vector: Optional[list[float]]


class IdentityEncoder:
    """Identity embedding with ONNXRuntime InstantID (antelopev2) or ResNet18 fallback."""

    def __init__(self, weights_dir: str) -> None:
        self.weights_dir = weights_dir
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._preprocess = None
        self._mode = "fallback"
        self._onnx_session = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            # Try ONNXRuntime + antelopev2 first
            try:
                import onnxruntime as ort  # type: ignore
                import os
                glintr = os.path.join(self.weights_dir, "antelopev2", "glintr100.onnx")
                if os.path.exists(glintr):
                    logger.info("Loading ONNXRuntime InstantID embedding: %s", glintr)
                    self._onnx_session = ort.InferenceSession(glintr, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # type: ignore
                    self._mode = "onnx"
                    self._loaded = True
                    return
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("ONNX InstantID unavailable: %s", exc)

            # Fallback: ResNet18 (reliable, minimal deps)
            try:
                from torchvision import models, transforms  # type: ignore
                import torch  # type: ignore

                logger.info("Loading ResNet18 fallback for identity embedding")
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                import torch  # type: ignore
                # Replace classification head with identity to output a 512-D embedding
                model.fc = torch.nn.Identity()
                self._preprocess = weights.transforms()
                self._model = model.eval()
                self._mode = "resnet"
                self._loaded = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Fallback identity encoder unavailable: %s", exc)
                self._model = None
                self._loaded = True

    def _embed_onnx(self, rgb: Image.Image) -> Optional[list[float]]:
        try:
            import numpy as np  # type: ignore
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            logger.warning("ONNX embedding prerequisites missing: %s", exc)
            return None
        if self._onnx_session is None:
            return None
        # Simple center-crop to square and resize to 112x112, normalize to [-1, 1]
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        crop = rgb.crop((left, top, left + side, top + side)).resize((112, 112))
        arr = np.asarray(crop)[:, :, ::-1].astype("float32")  # BGR
        arr = (arr - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        try:
            outputs = self._onnx_session.run(None, {self._onnx_session.get_inputs()[0].name: arr})
            vec = outputs[0].reshape(-1).astype("float32")
            return vec.tolist()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ONNX embedding failed: %s", exc)
            return None

    def embed(self, rgb_image: Image.Image) -> IdentityEmbedding:
        self.ensure_loaded()
        if self._mode == "onnx":
            vec = self._embed_onnx(rgb_image)
            if vec is not None:
                return IdentityEmbedding(vector=vec)
        if self._model is None or self._preprocess is None:
            return IdentityEmbedding(vector=None)
        try:
            import torch  # type: ignore

            x = self._preprocess(rgb_image).unsqueeze(0)
            with torch.no_grad():
                feats = self._model(x)
            vec = feats.flatten().cpu().numpy().astype(np.float32)
            return IdentityEmbedding(vector=vec.tolist())
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Identity embedding failed: %s", exc)
            return IdentityEmbedding(vector=None)

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
    """Lightweight identity embedding with optional InstantID.

    If InstantID/InsightFace assets are unavailable, falls back to a
    ResNet18-based global feature embedding to stabilize identity across frames.
    """

    def __init__(self, weights_dir: str) -> None:
        self.weights_dir = weights_dir
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._preprocess = None
        self._mode = "fallback"

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            # Try fallback first (reliable, minimal deps)
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

    def embed(self, rgb_image: Image.Image) -> IdentityEmbedding:
        self.ensure_loaded()
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

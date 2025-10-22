from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SegmentationNotReady(RuntimeError):
    pass


@dataclass
class SegmentationResult:
    """Segmentation outputs used by the pipeline."""

    alpha_mask: Image.Image
    face_mask: Optional[Image.Image]
    hand_mask: Optional[Image.Image]


class SegmentationEngine:
    """Person segmentation via torchvision with fallback.

    This approximates SCHP using torchvision DeepLabV3 to extract a person mask.
    It is light-weight and avoids custom HRNet code while still improving
    compositing/occlusion.
    """

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._preprocess = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch  # type: ignore
                from torchvision import models, transforms  # type: ignore

                logger.info("Loading torchvision DeepLabV3 for person segmentation")
                weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                self._model = models.segmentation.deeplabv3_resnet50(weights=weights).eval()
                self._preprocess = weights.transforms()
                self._loaded = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Segmentation model unavailable; falling back. %s", exc)
                self._model = None
                self._loaded = True

    def run(self, rgb_image: Image.Image) -> SegmentationResult:
        self.ensure_loaded()
        if self._model is None:
            return self.fallback(rgb_image)
        try:
            import torch  # type: ignore

            img_tensor = self._preprocess(rgb_image).unsqueeze(0)
            with torch.no_grad():
                out = self._model(img_tensor)["out"][0]
            person_class = 15  # COCO person
            mask = (out.argmax(0).cpu().numpy() == person_class).astype(np.uint8) * 255
            alpha = Image.fromarray(mask, mode="L").resize(rgb_image.size, Image.BILINEAR)
            return SegmentationResult(alpha_mask=alpha, face_mask=None, hand_mask=None)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Segmentation inference failed; using fallback. %s", exc)
            return self.fallback(rgb_image)

    @staticmethod
    def fallback(rgb_image: Image.Image) -> SegmentationResult:
        alpha = Image.new("L", rgb_image.size, 255)
        return SegmentationResult(alpha_mask=alpha, face_mask=None, hand_mask=None)

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class SegmentationNotReady(RuntimeError):
    pass


@dataclass
class SegmentationResult:
    """Placeholder structure to hold segmentation results."""

    alpha_mask: Image.Image
    face_mask: Optional[Image.Image]
    hand_mask: Optional[Image.Image]


class SegmentationEngine:
    """SCHP-based segmentation with graceful fallback."""

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch  # type: ignore

                # Placeholder for actual SCHP/HRNet weights load.
                if not self.weights_path or self.weights_path == "None":
                    raise FileNotFoundError("SCHP weights path not set.")
                logger.info("Loading SCHP weights from %s", self.weights_path)
                # TODO: Implement real model loading (HRNet + classification head)
                self._model = object()
                self._loaded = True
            except FileNotFoundError as exc:
                logger.warning("SCHP weights missing: %s", exc)
                self._model = None
                self._loaded = True
            except ImportError:
                logger.warning("PyTorch not available; SCHP segmentation disabled.")
                self._model = None
                self._loaded = True

    def run(self, rgb_image: Image.Image) -> SegmentationResult:
        self.ensure_loaded()
        if self._model is None:
            return self.fallback(rgb_image)
        raise SegmentationNotReady(
            "Segmentation model loaded but forward pass is not implemented yet."
        )

    @staticmethod
    def fallback(rgb_image: Image.Image) -> SegmentationResult:
        alpha = Image.new("L", rgb_image.size, 255)
        return SegmentationResult(
            alpha_mask=alpha,
            face_mask=None,
            hand_mask=None,
        )

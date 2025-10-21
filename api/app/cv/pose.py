from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    keypoints: Optional[List[float]]
    pose_map: Optional[Image.Image]


class PoseEstimator:
    """OpenPose / SMPL-X pose estimator with fallback."""

    def __init__(self, controlnet_dir: str) -> None:
        self.controlnet_dir = controlnet_dir
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
                import controlnet_aux  # type: ignore

                if not self.controlnet_dir or self.controlnet_dir == "None":
                    raise FileNotFoundError("ControlNet OpenPose directory not configured.")
                logger.info("Preparing OpenPose ControlNet assets from %s", self.controlnet_dir)
                # TODO: load actual OpenPose / ControlNet pipeline
                self._model = object()
                self._loaded = True
            except FileNotFoundError as exc:
                logger.warning("ControlNet assets missing: %s", exc)
                self._model = None
                self._loaded = True
            except ImportError:
                logger.warning("controlnet-aux not available; pose estimation disabled.")
                self._model = None
                self._loaded = True

    def run(self, rgb_image: Image.Image) -> PoseResult:
        self.ensure_loaded()
        if self._model is None:
            return self.fallback(rgb_image.size)
        raise RuntimeError("Pose estimation not implemented yet for the loaded model.")

    @staticmethod
    def fallback(size: tuple[int, int]) -> PoseResult:
        width, height = size
        pose = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(pose)
        draw.line([(width * 0.5, height * 0.2), (width * 0.5, height * 0.8)], fill=(255, 255, 255), width=8)
        draw.line([(width * 0.3, height * 0.45), (width * 0.7, height * 0.45)], fill=(255, 255, 255), width=8)
        draw.line([(width * 0.5, height * 0.8), (width * 0.3, height * 0.95)], fill=(255, 255, 255), width=8)
        draw.line([(width * 0.5, height * 0.8), (width * 0.7, height * 0.95)], fill=(255, 255, 255), width=8)
        return PoseResult(keypoints=None, pose_map=pose)

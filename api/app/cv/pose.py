from __future__ import annotations

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
    """OpenPose via controlnet-aux with graceful fallback."""

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
                # controlnet-aux provides an OpenposeDetector that outputs a pose map image
                from controlnet_aux.open_pose import OpenposeDetector  # type: ignore

                logger.info("Loading OpenposeDetector from controlnet-aux")
                # Using default weights shipped by controlnet-aux; this may download on first use
                self._model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                self._loaded = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("controlnet-aux unavailable or failed to load: %s", exc)
                self._model = None
                self._loaded = True

    def run(self, rgb_image: Image.Image) -> PoseResult:
        self.ensure_loaded()
        if self._model is None:
            return self.fallback(rgb_image.size)
        try:
            # Detector accepts PIL.Image and returns a PIL.Image pose map
            pose_map = self._model(rgb_image)
            return PoseResult(keypoints=None, pose_map=pose_map.convert("RGB"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Openpose inference failed (%s); using fallback pose map.", exc)
            return self.fallback(rgb_image.size)

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

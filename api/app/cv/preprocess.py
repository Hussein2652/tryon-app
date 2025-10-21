from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from PIL import Image

from .. import config
from .identity import IdentityEncoder, IdentityEmbedding
from .pose import PoseEstimator, PoseResult
from .segmentation import SegmentationEngine, SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class PreprocessArtifacts:
    segmentation: SegmentationResult
    pose: PoseResult
    identity: IdentityEmbedding


class PreprocessManager:
    """Orchestrates segmentation, pose estimation, and identity encoding."""

    def __init__(self) -> None:
        self.segmentation_engine = SegmentationEngine(str(config.SCHP_WEIGHTS))
        self.pose_estimator = PoseEstimator(str(config.CONTROLNET_OPENPOSE_DIR))
        self.identity_encoder = IdentityEncoder(str(config.INSTANTID_DIR))

    def run(self, user_photo: bytes) -> PreprocessArtifacts:
        image = Image.open(io.BytesIO(user_photo)).convert("RGB")

        segmentation = self._safe_segment(image)
        pose = self._safe_pose(image)
        identity = self._safe_identity(image)

        return PreprocessArtifacts(
            segmentation=segmentation,
            pose=pose,
            identity=identity,
        )

    def _safe_segment(self, image: Image.Image) -> SegmentationResult:
        try:
            return self.segmentation_engine.run(image)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Segmentation failed (%s); falling back.", exc)
            return self.segmentation_engine.fallback(image)

    def _safe_pose(self, image: Image.Image) -> PoseResult:
        try:
            return self.pose_estimator.run(image)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Pose estimation failed (%s); using fallback.", exc)
            return self.pose_estimator.fallback(image.size)

    def _safe_identity(self, image: Image.Image) -> IdentityEmbedding:
        try:
            return self.identity_encoder.embed(image)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Identity encoding failed (%s); clearing embedding.", exc)
            return IdentityEmbedding(vector=None)

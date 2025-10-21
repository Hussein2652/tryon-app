from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PhotoQualityIssue:
    code: str
    message: str


def evaluate_user_photo(photo_bytes: bytes) -> List[PhotoQualityIssue]:
    issues: List[PhotoQualityIssue] = []
    try:
        with Image.open(io.BytesIO(photo_bytes)) as image:
            width, height = image.size
        if width < 256 or height < 256:
            issues.append(
                PhotoQualityIssue(
                    code="low_resolution",
                    message="Upload a larger image (min 256x256).",
                )
            )
        aspect_ratio = height / max(width, 1)
        if aspect_ratio < 1.1:
            issues.append(
                PhotoQualityIssue(
                    code="insufficient_torso",
                    message="Ensure the photo captures the torso (portrait orientation).",
                )
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to parse user photo: %s", exc)
        issues.append(PhotoQualityIssue(code="invalid_image", message="Could not process the uploaded image."))
    return issues

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

_ENGINE_LOCK = threading.Lock()
_ENGINE_SINGLETON: Optional["StableVITONEngine"] = None

FALLBACK_FRAME_COUNT = 5


class StableVITONNotReady(RuntimeError):
    """Raised when StableVITON cannot run because dependencies are missing."""


@dataclass
class EngineInputs:
    user_png: bytes
    garment_png: bytes
    pose_map: Optional[bytes]
    masks: Optional[Dict[str, bytes]]
    pose_set: str
    sku: Optional[str]
    size: Optional[str]


class StableVITONEngine:
    """Lazy-loaded StableVITON runner.

    Replace the `_infer_real` method with the actual pipeline once the
    StableVITON/InstantID/SCHP stack is vendored into the project.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = threading.Lock()
        self._model = None  # Placeholder for framework-specific objects.

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                # Attempt to import heavy dependencies lazily. This keeps startup quick
                # and avoids mandatory installs during placeholder development.
                import stable_viton  # type: ignore

                self._model = stable_viton.load_model()  # type: ignore[attr-defined]
                self._loaded = True
            except ImportError:
                logger.warning(
                    "StableVITON dependencies are not installed; using placeholder frames."
                )
                self._model = None
                self._loaded = True
            except AttributeError as exc:
                raise StableVITONNotReady(
                    "StableVITON library found but load_model is not implemented."
                ) from exc

    def run(self, inputs: EngineInputs) -> List[bytes]:
        self.ensure_loaded()
        if self._model is None:
            return self._infer_placeholder(inputs)
        return self._infer_real(inputs)

    def _infer_real(self, inputs: EngineInputs) -> List[bytes]:
        raise StableVITONNotReady(
            "StableVITON model loading succeeded, but inference is not wired yet."
        )

    def _infer_placeholder(self, inputs: EngineInputs) -> List[bytes]:
        palette = [
            (40, 116, 166),
            (176, 58, 46),
            (48, 113, 107),
            (118, 68, 138),
            (189, 141, 24),
        ]
        frames: List[bytes] = []
        for idx in range(FALLBACK_FRAME_COUNT):
            image = Image.new("RGB", (768, 1024), palette[idx % len(palette)])
            draw = ImageDraw.Draw(image)
            lines = [
                "StableVITON Placeholder",
                f"Pose {idx + 1}/{FALLBACK_FRAME_COUNT}",
            ]
            if inputs.sku:
                lines.append(f"SKU {inputs.sku}")
            if inputs.size:
                lines.append(f"Size {inputs.size}")
            lines.append(f"Pose Set {inputs.pose_set}")
            y = 150
            for line in lines:
                text_width, text_height = draw.textsize(line)
                draw.text(
                    ((image.width - text_width) / 2, y),
                    line,
                    fill=(255, 255, 255),
                )
                y += text_height + 16
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            frames.append(buffer.getvalue())
        return frames


def _get_engine() -> StableVITONEngine:
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        with _ENGINE_LOCK:
            if _ENGINE_SINGLETON is None:
                _ENGINE_SINGLETON = StableVITONEngine()
    return _ENGINE_SINGLETON


def run_stableviton(
    *,
    user_png: bytes,
    garment_png: bytes,
    pose_map: Optional[bytes] = None,
    masks: Optional[Dict[str, bytes]] = None,
    pose_set: str = "SET_A",
    sku: Optional[str] = None,
    size: Optional[str] = None,
) -> List[bytes]:
    """Run StableVITON (or placeholder) to generate pose-aligned garment renders."""
    engine = _get_engine()
    inputs = EngineInputs(
        user_png=user_png,
        garment_png=garment_png,
        pose_map=pose_map,
        masks=masks,
        pose_set=pose_set,
        sku=sku,
        size=size,
    )
    return engine.run(inputs)

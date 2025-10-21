+from __future__ import annotations
+
+import io
+import logging
+import threading
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
+
+from PIL import Image, ImageDraw, ImageFont
+
+from .. import config
+
+if TYPE_CHECKING:  # pragma: no cover - typing hint only
+    from ..cv import PreprocessArtifacts
+
+
+logger = logging.getLogger(__name__)
+
+FALLBACK_FRAME_COUNT = 5
+
+
+class StableVITONNotReady(RuntimeError):
+    """Raised when StableVITON cannot run because dependencies are missing."""
+
+
+@dataclass(frozen=True)
+class EngineConfig:
+    ckpt_dir: str
+    controlnet_dir: str
+    instantid_dir: str
+    use_fp16: bool
+    max_res: int
+
+    def signature(self) -> Tuple[Any, ...]:
+        return (self.ckpt_dir, self.controlnet_dir, self.instantid_dir, self.use_fp16, self.max_res)
+
+
+@dataclass
+class EngineInputs:
+    user_png: bytes
+    garment_png: bytes
+    pose_map: Optional[bytes]
+    masks: Optional[Dict[str, bytes]]
+    pose_set: str
+    sku: Optional[str]
+    size: Optional[str]
+    artifacts: Optional["PreprocessArtifacts"] = None
+
+
+class StableVITONEngine:
+    """Lazy-loaded StableVITON runner with graceful fallbacks."""
+
+    def __init__(self, engine_cfg: EngineConfig) -> None:
+        self.cfg = engine_cfg
+        self._loaded = False
+        self._load_lock = threading.Lock()
+        self._model = None  # Placeholder for framework-specific objects.
+
+    def ensure_loaded(self) -> None:
+        if self._loaded:
+            return
+        with self._load_lock:
+            if self._loaded:
+                return
+            try:
+                import torch  # type: ignore
+                import diffusers  # type: ignore
+
+                logger.info(
+                    "Initializing StableVITON pipeline (ckpt=%s, fp16=%s)",
+                    self.cfg.ckpt_dir,
+                    self.cfg.use_fp16,
+                )
+                # TODO: load StableVITON + ControlNet + InstantID weights.
+                self._model = object()
+                self._loaded = True
+            except ImportError:
+                logger.warning(
+                    "StableVITON dependencies are not installed; using placeholder frames."
+                )
+                self._model = None
+                self._loaded = True
+            except Exception as exc:  # pylint: disable=broad-except
+                logger.exception("StableVITON initialization failed: %s", exc)
+                raise StableVITONNotReady(str(exc))
+
+    def run(self, inputs: EngineInputs) -> List[bytes]:
+        self.ensure_loaded()
+        if self._model is None:
+            return self._infer_placeholder(inputs)
+        return self._infer_real(inputs)
+
+    def _infer_real(self, inputs: EngineInputs) -> List[bytes]:
+        raise StableVITONNotReady(
+            "StableVITON model loading succeeded, but inference is not wired yet."
+        )
+
+    def _infer_placeholder(self, inputs: EngineInputs) -> List[bytes]:
+        palette = [
+            (40, 116, 166),
+            (176, 58, 46),
+            (48, 113, 107),
+            (118, 68, 138),
+            (189, 141, 24),
+        ]
+        font = ImageFont.load_default()
+        width = self.cfg.max_res
+        height = int(self.cfg.max_res * 1.33)
+        frames: List[bytes] = []
+        for idx in range(FALLBACK_FRAME_COUNT):
+            image = Image.new("RGB", (width, height), palette[idx % len(palette)])
+            draw = ImageDraw.Draw(image)
+            lines = [
+                "StableVITON Placeholder",
+                f"Pose {idx + 1}/{FALLBACK_FRAME_COUNT}",
+            ]
+            if inputs.sku:
+                lines.append(f"SKU {inputs.sku}")
+            if inputs.size:
+                lines.append(f"Size {inputs.size}")
+            lines.append(f"Pose Set {inputs.pose_set}")
+            if inputs.artifacts and inputs.artifacts.pose.pose_map is not None:
+                lines.append("Pose guidance: ✓")
+            if inputs.artifacts and inputs.artifacts.identity.vector is not None:
+                lines.append("InstantID: ✓")
+            y = 150
+            for line in lines:
+                bbox = draw.textbbox((0, 0), line, font=font)
+                text_width = bbox[2] - bbox[0]
+                text_height = bbox[3] - bbox[1]
+                draw.text(
+                    ((image.width - text_width) / 2, y),
+                    line,
+                    fill=(255, 255, 255),
+                    font=font,
+                )
+                y += text_height + 16
+            buffer = io.BytesIO()
+            image.save(buffer, format="PNG")
+            frames.append(buffer.getvalue())
+        return frames
+
+
_ENGINE_LOCK = threading.Lock()
_ENGINE_SINGLETON: Optional[StableVITONEngine] = None
_ENGINE_SIGNATURE: Optional[Tuple[Any, ...]] = None


def _get_engine(engine_cfg: EngineConfig) -> StableVITONEngine:
    global _ENGINE_SINGLETON, _ENGINE_SIGNATURE
    signature = engine_cfg.signature()
    with _ENGINE_LOCK:
        if _ENGINE_SINGLETON is None or _ENGINE_SIGNATURE != signature:
            _ENGINE_SINGLETON = StableVITONEngine(engine_cfg)
            _ENGINE_SIGNATURE = signature
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
    artifacts: Optional["PreprocessArtifacts"] = None,
) -> List[bytes]:
    """Run StableVITON (or placeholder) to generate pose-aligned garment renders."""

    engine_cfg = EngineConfig(
        ckpt_dir=str(config.STABLEVITON_CKPT_DIR),
        controlnet_dir=str(config.CONTROLNET_OPENPOSE_DIR),
        instantid_dir=str(config.INSTANTID_DIR),
        use_fp16=config.USE_FP16,
        max_res=config.MAX_RENDER_RES,
    )
    engine = _get_engine(engine_cfg)
    inputs = EngineInputs(
        user_png=user_png,
        garment_png=garment_png,
        pose_map=pose_map,
        masks=masks,
        pose_set=pose_set,
        sku=sku,
        size=size,
        artifacts=artifacts,
    )
    return engine.run(inputs)

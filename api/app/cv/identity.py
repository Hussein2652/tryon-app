from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class IdentityEmbedding:
    vector: Optional[list[float]]


class IdentityEncoder:
    """InstantID encoder placeholder."""

    def __init__(self, weights_dir: str) -> None:
        self.weights_dir = weights_dir
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

                if not self.weights_dir or self.weights_dir == "None":
                    raise FileNotFoundError("InstantID weights directory not configured.")
                logger.info("Loading InstantID assets from %s", self.weights_dir)
                # TODO: implement real InstantID load
                self._model = object()
                self._loaded = True
            except FileNotFoundError as exc:
                logger.warning("InstantID assets missing: %s", exc)
                self._model = None
                self._loaded = True
            except ImportError:
                logger.warning("PyTorch not available; InstantID disabled.")
                self._model = None
                self._loaded = True

    def embed(self, rgb_image: Image.Image) -> IdentityEmbedding:
        self.ensure_loaded()
        if self._model is None:
            return IdentityEmbedding(vector=None)
        raise RuntimeError("InstantID inference not implemented yet.")

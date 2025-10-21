from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from . import config

logger = logging.getLogger(__name__)


def _cache_path(cache_key: str, namespace: str) -> Path:
    return config.CACHE_DIR / namespace / f"{cache_key}.json"


def write_cache(cache_key: str, namespace: str, payload: Dict[str, Any]) -> None:
    path = _cache_path(cache_key, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to write cache metadata for %s: %s", cache_key, exc)


def read_cache(cache_key: str, namespace: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_key, namespace)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to read cache metadata for %s: %s", cache_key, exc)
        return None


def delete_cache(cache_key: str, namespace: str) -> bool:
    path = _cache_path(cache_key, namespace)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to delete cache metadata for %s: %s", cache_key, exc)
        return False

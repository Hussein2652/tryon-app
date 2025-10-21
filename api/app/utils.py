import hashlib
from typing import Iterable


def stable_content_hash(parts: Iterable[bytes], prefix: str = "") -> str:
    """Return a stable SHA256 hex digest for the provided byte chunks."""
    hasher = hashlib.sha256()
    if prefix:
        hasher.update(prefix.encode("utf-8"))
    for chunk in parts:
        hasher.update(chunk)
    return hasher.hexdigest()

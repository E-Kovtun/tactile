"""Stable identifiers shared by downstream evaluation datasets."""

import hashlib
from pathlib import Path
from typing import Union


def _canonical_path(value: Union[str, Path]) -> str:
    """Return a stable textual path without requiring that it exists."""
    return Path(value).expanduser().as_posix().rstrip("/")


def stable_int64_id(*parts: object) -> int:
    """Hash arbitrary identity parts into a non-negative signed int64."""
    encoded = "\x1f".join(_canonical_path(p) if isinstance(p, Path) else str(p) for p in parts)
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

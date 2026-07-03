import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import yaml


def _to_canonical(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _to_canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_to_canonical(v) for v in value]
    return value


def stable_hash(value: Any) -> str:
    canonical = _to_canonical(value)
    payload = yaml.safe_dump(canonical, sort_keys=True, allow_unicode=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def function_fingerprint(fn: Callable) -> dict:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = repr(fn)
    return {
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", repr(fn)),
        "source_hash": stable_hash(source),
    }


def producer_fingerprint(functions: tuple[Callable, ...], constants: Mapping[str, Any]) -> dict:
    return {
        "functions": [function_fingerprint(fn) for fn in functions],
        "constants_hash": stable_hash(constants),
    }

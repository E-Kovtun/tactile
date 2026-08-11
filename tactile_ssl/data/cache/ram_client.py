"""Client helpers for the optional node-local artifact RAM cache daemon."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "/tmp/tactile-xela-ram-cache.sock"
SOCKET_ENV = "TACTILE_RAM_CACHE_SOCKET"
_warned_paths: set[str] = set()


def configured_socket_path() -> Path:
    return Path(os.environ.get(SOCKET_ENV, DEFAULT_SOCKET_PATH))


def _request(
    payload: dict,
    timeout_s: float = 5.0,
    socket_path: Optional[Path] = None,
) -> Optional[dict]:
    socket_path = socket_path or configured_socket_path()
    if not socket_path.exists():
        return None

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_s)
            client.connect(str(socket_path))
            client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            response = bytearray()
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if b"\n" in chunk:
                    break
        if not response:
            return None
        parsed = json.loads(bytes(response).splitlines()[0])
        return parsed if parsed.get("ok") else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warning_key = str(socket_path)
        if warning_key not in _warned_paths:
            log.warning(
                "RAM cache daemon is unavailable at %s; falling back to the regular artifact cache: %s",
                socket_path,
                exc,
            )
            _warned_paths.add(warning_key)
        return None


def load_artifact_from_daemon(npz_path: Path) -> Optional[dict[str, np.ndarray]]:
    """Attach to read-only memfd arrays, or return ``None`` without a daemon."""

    response = _request({"command": "get", "source": str(npz_path.resolve())})
    if response is None:
        return None

    arrays = response.get("arrays")
    if not isinstance(arrays, dict):
        return None
    try:
        return {
            name: np.load(local_path, mmap_mode="r", allow_pickle=False)
            for name, local_path in arrays.items()
        }
    except (OSError, ValueError) as exc:
        log.warning("Unable to attach to RAM-cache artifact %s: %s", npz_path, exc)
        return None


def daemon_status(timeout_s: float = 2.0) -> Optional[dict]:
    return _request({"command": "status"}, timeout_s=timeout_s)

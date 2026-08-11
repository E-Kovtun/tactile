"""Node-local, diskless memfd cache for repeatedly consumed compressed artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import socketserver
import threading
import time
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


def memfd_supported() -> bool:
    if not Path("/proc/self/fd").is_dir():
        return False
    if hasattr(os, "memfd_create"):
        return True
    try:
        return hasattr(ctypes.CDLL(None), "memfd_create")
    except OSError:
        return False


def _memfd_create(name: str) -> int:
    flags = getattr(os, "MFD_CLOEXEC", 0x0001)
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags=flags)

    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(name.encode("utf-8"), flags)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


@dataclass
class _ResidentArray:
    fd: int
    mapping: np.ndarray
    path: str
    nbytes: int


class ArtifactRamStore:
    """Decompress NPZ artifacts into anonymous Linux memfd-backed arrays."""

    def __init__(self, source_root: Path, local_root: Path, warm_workers: int = 2):
        self.source_root = source_root.resolve()
        # Kept only so old CLI invocations remain compatible. The memfd backend
        # never creates or writes this directory.
        self.local_root = local_root.resolve()
        self.warm_workers = max(1, int(warm_workers))
        if not memfd_supported():
            raise RuntimeError("The RAM-cache daemon requires Linux memfd and /proc")
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._resident: dict[str, dict[str, _ResidentArray]] = {}
        self._resident_bytes = 0
        self._state_guard = threading.Lock()
        self.warming = False
        self.ready = False
        self.warm_total = 0
        self.warm_completed = 0
        self.warm_errors: list[str] = []
        self.started_at = time.time()

    def _validate_source(self, source: Path) -> Path:
        source = source.resolve()
        if source.suffix != ".npz" or not source.is_relative_to(self.source_root):
            raise ValueError(f"Artifact is outside source_root or is not NPZ: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        return source

    def _lock_for(self, source: Path) -> threading.Lock:
        key = str(source)
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _create_resident_array(
        self, source: Path, array_name: str, array: np.ndarray
    ) -> _ResidentArray:
        memfd_name = f"xela-{source.stem}-{array_name}"[:200]
        fd = _memfd_create(memfd_name)
        try:
            # Write a standard NPY header and payload into anonymous shmem. The
            # /proc path lets independent downstream processes open the same
            # memfd without copying its contents or creating a disk file.
            with os.fdopen(os.dup(fd), "wb") as output:
                np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
            path = f"/proc/{os.getpid()}/fd/{fd}"
            mapping = np.load(path, mmap_mode="r", allow_pickle=False)
            self._touch_pages(mapping)
            return _ResidentArray(
                fd=fd,
                mapping=mapping,
                path=path,
                nbytes=int(mapping.nbytes),
            )
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _touch_pages(array: np.ndarray) -> None:
        if array.size == 0:
            return
        # Reshape first: NumPy rejects a dtype-size-changing view directly on
        # a 0-D array, while the equivalent one-element 1-D view is valid.
        byte_view = np.asarray(array).reshape(-1).view(np.uint8)
        # One byte per 4 KiB page is enough to populate the node page cache.
        int(byte_view[::4096].sum(dtype=np.uint64))

    def get(self, source_value: str) -> dict[str, str]:
        source = self._validate_source(Path(source_value))
        key = str(source)
        with self._lock_for(source):
            resident = self._resident.get(key)
            if resident is not None:
                return {name: item.path for name, item in resident.items()}

            mapped: dict[str, _ResidentArray] = {}
            try:
                with np.load(source, allow_pickle=False) as archive:
                    for name in archive.files:
                        mapped[name] = self._create_resident_array(
                            source, name, archive[name]
                        )
            except Exception:
                self._close_arrays(mapped)
                raise
            self._resident[key] = mapped
            with self._state_guard:
                self._resident_bytes += sum(item.nbytes for item in mapped.values())
            return {name: item.path for name, item in mapped.items()}

    def warm_all(self) -> None:
        with self._state_guard:
            if self.warming or self.ready:
                return
            self.warming = True
            sources = sorted(
                source
                for source in self.source_root.rglob("*.npz")
                if not any(
                    part.startswith(".")
                    for part in source.relative_to(self.source_root).parts
                )
            )
            self.warm_total = len(sources)

        def warm_one(source: Path) -> Optional[str]:
            try:
                self.get(str(source))
                return None
            except Exception as exc:  # keep warming independent artifacts
                return f"{source}: {exc}"

        with ThreadPoolExecutor(max_workers=self.warm_workers) as pool:
            futures = [pool.submit(warm_one, source) for source in sources]
            for future in as_completed(futures):
                error = future.result()
                with self._state_guard:
                    self.warm_completed += 1
                    if error is not None:
                        self.warm_errors.append(error)

        with self._state_guard:
            self.warming = False
            self.ready = not self.warm_errors

    def start_warm_thread(self) -> None:
        threading.Thread(target=self.warm_all, name="artifact-ram-warm", daemon=True).start()

    def status(self) -> dict:
        with self._state_guard:
            return {
                "source_root": str(self.source_root),
                "storage_backend": "memfd",
                "disk_bytes": 0,
                "warming": self.warming,
                "ready": self.ready,
                "warm_total": self.warm_total,
                "warm_completed": self.warm_completed,
                "warm_errors": self.warm_errors[-10:],
                "resident_artifacts": len(self._resident),
                "resident_bytes": self._resident_bytes,
                "uptime_s": time.time() - self.started_at,
            }

    @staticmethod
    def _close_arrays(arrays: dict[str, _ResidentArray]) -> None:
        for item in arrays.values():
            mmap_obj = getattr(item.mapping, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            try:
                os.close(item.fd)
            except OSError:
                pass

    def close(self) -> None:
        resident = self._resident
        self._resident = {}
        for arrays in resident.values():
            self._close_arrays(arrays)
        self._resident_bytes = 0


class _RamCacheRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline())
            command = request.get("command")
            if command == "get":
                response = {"ok": True, "arrays": self.server.store.get(request["source"])}
            elif command == "status":
                response = {"ok": True, **self.server.store.status()}
            elif command == "warm":
                self.server.store.start_warm_thread()
                response = {"ok": True, **self.server.store.status()}
            elif command == "shutdown":
                response = {"ok": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                response = {"ok": False, "error": f"Unknown command: {command}"}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")


class ArtifactRamServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: Path, store: ArtifactRamStore):
        self.socket_path = socket_path
        self.store = store
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.unlink(missing_ok=True)
        super().__init__(str(socket_path), _RamCacheRequestHandler)
        os.chmod(socket_path, 0o600)

    def server_close(self) -> None:
        try:
            self.store.close()
        finally:
            super().server_close()
            self.socket_path.unlink(missing_ok=True)


def serve(
    *,
    source_root: Path,
    local_root: Path,
    socket_path: Path,
    warm: bool,
    warm_workers: int,
) -> None:
    store = ArtifactRamStore(source_root, local_root, warm_workers=warm_workers)
    server = ArtifactRamServer(socket_path, store)
    if warm:
        store.start_warm_thread()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()

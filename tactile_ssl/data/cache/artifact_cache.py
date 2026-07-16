from pathlib import Path
from typing import Callable, Mapping, Optional
import logging
import os
import shutil
import socket
import threading
import time
import uuid

import numpy as np
import yaml

from tactile_ssl.data.cache.fingerprint import producer_fingerprint, stable_hash
from tactile_ssl.data.cache.spec import CacheSpec


log = logging.getLogger(__name__)


class ArtifactCache:
    def __init__(
        self,
        root: str,
        enabled: bool = True,
        force_recompute: bool = False,
        log_hits: bool = True,
        lock_timeout_s: float = 1800.0,
        stale_lock_s: float = 3600.0,
        lock_log_interval_s: float = 30.0,
        lock_poll_s: float = 0.1,
    ):
        self.root = Path(root)
        self.enabled = enabled
        self.force_recompute = force_recompute
        self.log_hits = log_hits
        self.lock_timeout_s = lock_timeout_s
        self.stale_lock_s = stale_lock_s
        self.lock_log_interval_s = lock_log_interval_s
        self.lock_poll_s = lock_poll_s

    def build_key(self, spec: CacheSpec) -> str:
        payload = {
            "artifact": spec.artifact,
            "schema_version": spec.schema_version,
            "semantic_params": spec.semantic_params,
            "producer": producer_fingerprint(tuple(spec.producer_functions), spec.producer_constants),
            "upstream_keys": spec.upstream_keys,
        }
        return stable_hash(payload)[:24]

    def artifact_paths(self, artifact: str, key: str) -> tuple[Path, Path]:
        artifact_dir = self.root / artifact
        return artifact_dir / f"{key}.npz", artifact_dir / f"{key}.yaml"

    def lock_path(self, artifact: str, key: str) -> Path:
        return self.root / "locks" / artifact / f"{key}.lock"

    def _load_artifact(self, npz_path: Path) -> dict[str, np.ndarray]:
        with np.load(npz_path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def _artifact_exists(self, npz_path: Path, yaml_path: Path) -> bool:
        return npz_path.exists() and yaml_path.exists()

    def _owner_path(self, lock_path: Path) -> Path:
        return lock_path / "owner.yaml"

    def _heartbeat_path(self, lock_path: Path) -> Path:
        return lock_path / "heartbeat"

    def _read_lock_owner(self, lock_path: Path) -> dict:
        try:
            with open(self._owner_path(lock_path)) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            log.warning(f"Unable to read cache lock owner at {lock_path}: {exc}")
            return {}

    def _write_lock_owner(self, lock_path: Path, token: str) -> None:
        owner = {
            "token": token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": time.time(),
        }
        with open(self._owner_path(lock_path), "w") as f:
            yaml.safe_dump(owner, f, sort_keys=True)
        self._touch_lock(lock_path)

    def _touch_lock(self, lock_path: Path) -> None:
        now = time.time()
        heartbeat_path = self._heartbeat_path(lock_path)
        heartbeat_path.touch(exist_ok=True)
        os.utime(heartbeat_path, (now, now))
        os.utime(lock_path, (now, now))

    def _lock_age_s(self, lock_path: Path) -> float:
        candidates = [lock_path, self._heartbeat_path(lock_path), self._owner_path(lock_path)]
        mtimes = []
        for path in candidates:
            try:
                mtimes.append(path.stat().st_mtime)
            except FileNotFoundError:
                pass
        if not mtimes:
            return 0.0
        return max(0.0, time.time() - max(mtimes))

    def _remove_stale_lock(self, lock_path: Path) -> None:
        try:
            if lock_path.is_dir():
                shutil.rmtree(lock_path)
            else:
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # A completed artifact is still usable even if NFS delays or
            # rejects removal of its lock directory. Waiters re-check the
            # artifact while waiting, so a cleanup failure must not deadlock
            # the whole dataset loader.
            log.warning(f"Unable to remove cache lock at {lock_path}: {exc}")

    def _acquire_lock(
        self,
        lock_path: Path,
        artifact_ready: Optional[Callable[[], bool]] = None,
    ) -> Optional[str]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        start_time = time.monotonic()
        last_log_time = start_time
        while True:
            try:
                lock_path.mkdir()
                try:
                    self._write_lock_owner(lock_path, token)
                except BaseException:
                    # Do not strand an ownerless directory if owner metadata
                    # creation fails after the atomic mkdir succeeded.
                    self._remove_stale_lock(lock_path)
                    raise
                return token
            except FileExistsError:
                # Another worker may have completed and atomically published
                # the artifact but failed to remove its lock directory. Once
                # both artifact files exist there is no reason to wait for or
                # mutate that orphaned lock.
                if artifact_ready is not None and artifact_ready():
                    return None

                now = time.monotonic()
                wait_s = now - start_time
                age_s = self._lock_age_s(lock_path)
                owner = self._read_lock_owner(lock_path)

                if self.stale_lock_s is not None and age_s > self.stale_lock_s:
                    log.warning(
                        f"Removing stale cache lock at {lock_path}; "
                        f"age={age_s:.1f}s, owner={owner}"
                    )
                    self._remove_stale_lock(lock_path)
                    continue

                if self.lock_timeout_s is not None and wait_s > self.lock_timeout_s:
                    raise TimeoutError(
                        f"Timed out after {wait_s:.1f}s waiting for cache lock {lock_path}. "
                        f"Lock age is {age_s:.1f}s, owner={owner}. "
                        "If no matching process is alive, remove this lock directory or use a different data.cache.root."
                    )

                if now - last_log_time >= self.lock_log_interval_s:
                    log.warning(
                        f"Waiting for cache lock {lock_path}; "
                        f"waited={wait_s:.1f}s, lock_age={age_s:.1f}s, owner={owner}"
                    )
                    last_log_time = now

                time.sleep(self.lock_poll_s)

    def _start_lock_heartbeat(self, lock_path: Path, token: str) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def run() -> None:
            while not stop_event.wait(10.0):
                owner = self._read_lock_owner(lock_path)
                if owner.get("token") != token:
                    log.warning(f"Stopping cache lock heartbeat for {lock_path}; lock owner changed")
                    return
                try:
                    self._touch_lock(lock_path)
                except FileNotFoundError:
                    log.warning(f"Stopping cache lock heartbeat for {lock_path}; lock disappeared")
                    return
                except Exception as exc:
                    log.warning(f"Unable to update cache lock heartbeat for {lock_path}: {exc}")

        thread = threading.Thread(target=run, name=f"cache-lock-heartbeat-{lock_path.name}", daemon=True)
        thread.start()
        return stop_event, thread

    def _release_lock(self, lock_path: Path, token: str) -> None:
        owner = self._read_lock_owner(lock_path)
        if owner.get("token") != token:
            log.warning(
                f"Not releasing cache lock {lock_path}; "
                f"expected token={token}, current owner={owner}"
            )
            return
        self._remove_stale_lock(lock_path)

    def _write_artifact(
        self,
        spec: CacheSpec,
        key: str,
        arrays: Mapping[str, np.ndarray],
        metadata: Optional[Mapping],
        npz_path: Path,
        yaml_path: Path,
    ) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_prefix = f".{key}.{os.getpid()}"
        tmp_npz_path = npz_path.parent / f"{tmp_prefix}.tmp.npz"
        tmp_yaml_path = yaml_path.parent / f"{tmp_prefix}.tmp.yaml"
        try:
            np.savez_compressed(tmp_npz_path, **arrays)
            manifest = {
                "artifact": spec.artifact,
                "schema_version": spec.schema_version,
                "key": key,
                "semantic_params": spec.semantic_params,
                "upstream_keys": spec.upstream_keys,
                "producer_hash": stable_hash(
                    producer_fingerprint(tuple(spec.producer_functions), spec.producer_constants)
                ),
                "metadata": dict(metadata or {}),
                "outputs": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for name, value in arrays.items()
                },
            }
            with open(tmp_yaml_path, "w") as f:
                yaml.safe_dump(manifest, f, sort_keys=True)
            os.replace(tmp_npz_path, npz_path)
            os.replace(tmp_yaml_path, yaml_path)
        finally:
            tmp_npz_path.unlink(missing_ok=True)
            tmp_yaml_path.unlink(missing_ok=True)

    def get_or_compute(
        self,
        spec: CacheSpec,
        compute_fn: Callable[[], Mapping[str, np.ndarray]],
        metadata: Optional[Mapping] = None,
    ) -> tuple[dict[str, np.ndarray], str]:
        key = self.build_key(spec)
        npz_path, yaml_path = self.artifact_paths(spec.artifact, key)

        if self.enabled and not self.force_recompute and self._artifact_exists(npz_path, yaml_path):
            if self.log_hits:
                log.info(f"Cache hit for {spec.artifact}: {key}")
            return self._load_artifact(npz_path), key

        if not self.enabled:
            arrays = dict(compute_fn())
            return arrays, key

        lock_path = self.lock_path(spec.artifact, key)
        lock_token = self._acquire_lock(
            lock_path,
            artifact_ready=(
                None
                if self.force_recompute
                else lambda: self._artifact_exists(npz_path, yaml_path)
            ),
        )
        if lock_token is None:
            if self.log_hits:
                log.info(f"Cache hit for {spec.artifact}: {key} (artifact published while waiting)")
            return self._load_artifact(npz_path), key

        heartbeat_stop, heartbeat_thread = self._start_lock_heartbeat(lock_path, lock_token)
        try:
            if not self.force_recompute and self._artifact_exists(npz_path, yaml_path):
                if self.log_hits:
                    log.info(f"Cache hit for {spec.artifact}: {key}")
                return self._load_artifact(npz_path), key

            log.info(f"Cache miss for {spec.artifact}: {key}")
            arrays = dict(compute_fn())
            self._write_artifact(spec, key, arrays, metadata, npz_path, yaml_path)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            self._release_lock(lock_path, lock_token)
        return arrays, key

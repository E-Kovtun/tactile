from pathlib import Path
from typing import Callable, Mapping, Optional
import logging
import os
import time

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
    ):
        self.root = Path(root)
        self.enabled = enabled
        self.force_recompute = force_recompute
        self.log_hits = log_hits

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

    def _acquire_lock(self, lock_path: Path) -> None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                lock_path.mkdir()
                return
            except FileExistsError:
                time.sleep(0.1)

    def _release_lock(self, lock_path: Path) -> None:
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass

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
        self._acquire_lock(lock_path)
        try:
            if not self.force_recompute and self._artifact_exists(npz_path, yaml_path):
                if self.log_hits:
                    log.info(f"Cache hit for {spec.artifact}: {key}")
                return self._load_artifact(npz_path), key

            log.info(f"Cache miss for {spec.artifact}: {key}")
            arrays = dict(compute_fn())
            self._write_artifact(spec, key, arrays, metadata, npz_path, yaml_path)
        finally:
            self._release_lock(lock_path)
        return arrays, key

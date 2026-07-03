from pathlib import Path
from typing import Callable, Mapping, Optional
import logging

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

    def get_or_compute(
        self,
        spec: CacheSpec,
        compute_fn: Callable[[], Mapping[str, np.ndarray]],
        metadata: Optional[Mapping] = None,
    ) -> tuple[dict[str, np.ndarray], str]:
        key = self.build_key(spec)
        npz_path, yaml_path = self.artifact_paths(spec.artifact, key)

        if self.enabled and not self.force_recompute and npz_path.exists() and yaml_path.exists():
            if self.log_hits:
                log.info(f"Cache hit for {spec.artifact}: {key}")
            with np.load(npz_path, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}, key

        if self.enabled:
            log.info(f"Cache miss for {spec.artifact}: {key}")

        arrays = dict(compute_fn())
        if not self.enabled:
            return arrays, key

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
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
        with open(yaml_path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=True)
        return arrays, key

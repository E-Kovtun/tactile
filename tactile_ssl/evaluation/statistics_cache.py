"""Persistent cache for expensive downstream bootstrap and permutation statistics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .artifacts import EvaluationArtifact
from .statistics import MetricResult, analyze_pair


STATISTICS_CACHE_VERSION = 1


def artifact_content_fingerprint(artifact: EvaluationArtifact) -> str:
    """Hash every value that can affect a downstream metric."""
    digest = hashlib.sha256()
    for name in ("sample_id", "group_id", "y_true", "y_pred"):
        array = np.ascontiguousarray(getattr(artifact, name))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(str(artifact.manifest.get("metric_version")).encode("utf-8"))
    return digest.hexdigest()


def _cache_identity(
    task: str,
    baseline: EvaluationArtifact,
    candidate: Optional[EvaluationArtifact],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> Tuple[str, Dict[str, Any]]:
    identity = {
        "cache_version": STATISTICS_CACHE_VERSION,
        "task": task,
        "baseline_fingerprint": artifact_content_fingerprint(baseline),
        "candidate_fingerprint": (
            artifact_content_fingerprint(candidate) if candidate is not None else None
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "permutation_samples": int(permutation_samples),
        "seed": int(seed),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def _load_cache(path: Path, identity: Dict[str, Any]) -> Optional[List[MetricResult]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("identity") != identity:
            return None
        return [MetricResult(**result) for result in payload["results"]]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_cache(path: Path, identity: Dict[str, Any], results: List[MetricResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "results": [asdict(result) for result in results],
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def analyze_pair_cached(
    cache_dir: Path,
    task: str,
    baseline: EvaluationArtifact,
    candidate: Optional[EvaluationArtifact],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> Tuple[List[MetricResult], str, bool]:
    """Return statistics, cache key and whether the result came from cache."""
    cache_key, identity = _cache_identity(
        task,
        baseline,
        candidate,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
    )
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    cached = _load_cache(cache_path, identity)
    if cached is not None:
        return cached, cache_key, True

    results = analyze_pair(
        task,
        baseline,
        candidate,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
    )
    _save_cache(cache_path, identity, results)
    return results, cache_key, False

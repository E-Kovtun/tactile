"""Atomic storage for downstream test predictions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


ARTIFACT_SCHEMA_VERSION = 1
METRIC_VERSION = "downstream-test-v1"
GROUPING_VERSION_BY_TASK = {
    "force": "force-contact-episodes-v1",
    "pose": "pose-recordings-v1",
    "object_classification": "object-recordings-v1",
}


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def load_run_metadata(run_root: Path) -> Dict[str, Any]:
    """Read seed and spatial-coordinate mode from a saved Hydra config."""
    from omegaconf import OmegaConf

    run_root = Path(run_root)
    candidates = (
        run_root / "resolved_config.yaml",
        run_root / "config.yaml",
        run_root / ".hydra" / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            config = OmegaConf.load(path)
            raw = OmegaConf.to_container(config, resolve=False)
            seed = OmegaConf.select(config, "seed", default=None)
            spatial = None
            for key in (
                "data.dataset.config.features.use_spatial_coords",
                "data.dataset.features.use_spatial_coords",
                "data.features.use_spatial_coords",
            ):
                spatial = OmegaConf.select(config, key, default=None)
                if spatial is not None:
                    break
            if spatial is None:
                spatial = _find_key(raw, "use_spatial_coords")
            if isinstance(spatial, str):
                normalized = spatial.strip().lower()
                if normalized in {"true", "1", "yes"}:
                    spatial = True
                elif normalized in {"false", "0", "no"}:
                    spatial = False
            return {
                "seed": int(seed) if seed is not None else None,
                "use_spatial_coords": bool(spatial) if spatial is not None else None,
                "config_path": str(path),
            }
        except Exception:
            continue
    return {"seed": None, "use_spatial_coords": None, "config_path": None}


@dataclass(frozen=True)
class EvaluationArtifact:
    sample_id: np.ndarray
    group_id: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    manifest: Dict[str, Any]
    path: Path


def dataset_fingerprint(sample_id: np.ndarray, group_id: np.ndarray) -> str:
    """Fingerprint the exact evaluated sequence, including DDP duplicates."""
    digest = hashlib.sha256()
    digest.update(np.asarray(sample_id, dtype=np.int64).tobytes())
    digest.update(np.asarray(group_id, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def save_evaluation_artifact(
    run_root: Path,
    *,
    task: str,
    sample_id: np.ndarray,
    group_id: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    checkpoint: Optional[str],
    seed: Optional[int],
    use_spatial_coords: Optional[bool],
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically write predictions and their manifest under ``run_root``."""
    sample_id = np.asarray(sample_id, dtype=np.int64).reshape(-1)
    group_id = np.asarray(group_id, dtype=np.int64).reshape(-1)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    size = len(sample_id)
    if not (len(group_id) == len(y_true) == len(y_pred) == size):
        raise ValueError("sample_id, group_id, y_true and y_pred must have the same leading dimension")

    evaluation_dir = Path(run_root) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = evaluation_dir / "test_predictions.npz"
    with tempfile.NamedTemporaryFile("wb", dir=evaluation_dir, suffix=".npz", delete=False) as handle:
        np.savez_compressed(
            handle,
            sample_id=sample_id,
            group_id=group_id,
            y_true=y_true,
            y_pred=y_pred,
        )
        tmp_predictions = Path(handle.name)
    os.replace(tmp_predictions, predictions_path)

    manifest: Dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "task": task,
        "grouping_version": GROUPING_VERSION_BY_TASK.get(task),
        "seed": seed,
        "checkpoint": checkpoint,
        "use_spatial_coords": use_spatial_coords,
        "dataset_fingerprint": dataset_fingerprint(sample_id, group_id),
        "num_examples": size,
        "num_groups": int(np.unique(group_id).size),
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    _atomic_json(evaluation_dir / "manifest.json", manifest)
    return predictions_path


def load_evaluation_artifact(run_root: Path) -> EvaluationArtifact:
    evaluation_dir = Path(run_root) / "evaluation"
    predictions_path = evaluation_dir / "test_predictions.npz"
    manifest_path = evaluation_dir / "manifest.json"
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete evaluation artifact in {evaluation_dir}")
    with np.load(predictions_path, allow_pickle=False) as values:
        required = {"sample_id", "group_id", "y_true", "y_pred"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(f"Missing arrays in {predictions_path}: {sorted(missing)}")
        arrays = {name: np.array(values[name]) for name in required}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return EvaluationArtifact(
        sample_id=arrays["sample_id"],
        group_id=arrays["group_id"],
        y_true=arrays["y_true"],
        y_pred=arrays["y_pred"],
        manifest=manifest,
        path=predictions_path,
    )

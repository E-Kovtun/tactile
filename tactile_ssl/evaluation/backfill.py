"""Backfill missing test artifacts without starting a training loop."""

from __future__ import annotations

import importlib
import re
import shutil
from pathlib import Path
from typing import Any, Tuple

import hydra
import numpy as np
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from tactile_ssl.trainer import Trainer

from .artifacts import (
    GROUPING_VERSION_BY_TASK,
    EvaluationArtifact,
    load_evaluation_artifact,
    save_evaluation_artifact,
)


_FORCE_GROUP_LOOKUPS: list[dict[int, int]] = []


class NullSummaryWriter:
    """TensorBoard-compatible sink used during inference-only backfill."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_image(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def find_run_config(run_root: Path) -> Path:
    candidates = (
        run_root / "config.yaml",
        run_root / "resolved_config.yaml",
        run_root / ".hydra" / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No saved run config found in {run_root}; tried " + ", ".join(str(path) for path in candidates)
    )


def select_checkpoint(run_root: Path) -> Path:
    checkpoint_dir = run_root / "checkpoints"
    for name in ("best.ckpt", "last.ckpt"):
        path = checkpoint_dir / name
        if path.is_file():
            return path
    epoch_pattern = re.compile(r"epoch[-_=]?(\d+)", re.IGNORECASE)
    candidates = []
    for path in checkpoint_dir.iterdir() if checkpoint_dir.is_dir() else ():
        if path.suffix not in {".ckpt", ".pth", ".pt"}:
            continue
        match = epoch_pattern.search(path.stem)
        if match:
            candidates.append((int(match.group(1)), path.suffix == ".ckpt", path))
    if not candidates:
        raise FileNotFoundError(f"No best, last, or epoch checkpoint found in {checkpoint_dir}")
    return max(candidates, key=lambda value: (value[0], value[1], value[2].name))[2]


def _loaders_for_task(task: str, cfg: DictConfig):
    modules = {
        "force": "train_task_force",
        "pose": "train_task_pose_estimation",
        "object_classification": "train_task_object",
    }
    if task not in modules:
        raise ValueError(f"Unsupported downstream task: {task}")
    # The legacy entrypoints register some resolver names at import time.
    # Their definitions are equivalent, but OmegaConf rejects duplicate registration.
    for resolver in (
        "int_multiply",
        "int_divide",
        "d360_expt_name",
        "d360_modal_tag",
        "d360_modal_used_tag",
        "capitalize",
        "join",
    ):
        if OmegaConf.has_resolver(resolver):
            OmegaConf.clear_resolver(resolver)
    module = importlib.import_module(modules[task])
    # If the module was already imported, its import-time registrations are not
    # executed again. Reinstall the common resolvers before resolving this run.
    common_resolvers = {
        "int_multiply": lambda a, b: int(a * b),
        "int_divide": lambda a, b: a // b,
        "join": lambda separator, values: separator.join(map(str, values)),
    }
    for name, resolver in common_resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    # Do not resolve the complete run config here. Some downstream loaders fill
    # values derived from the training dataset before model instantiation. In
    # particular, object classification replaces ``data.object_classes`` and
    # ``data.object_class_weights`` after scanning the datasets. Resolving now
    # would freeze the corresponding ``task.model_task`` interpolations as None.
    # OmegaConf resolves the individual values lazily when the loader and Hydra
    # access them, once all task-specific derived fields are available.
    return module.get_dataloaders(cfg)


def _prepare_config(config_path: Path, run_root: Path, checkpoint: Path) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        # Saved Hydra configs retain ``${hydra:runtime.cwd}``, but backfill is a
        # plain Python entrypoint with no active Hydra runtime. Resolve the
        # original project root explicitly before dataset/cache interpolation.
        OmegaConf.update(
            cfg,
            "paths.work_dir",
            str(Path(__file__).resolve().parents[2]),
            force_add=True,
        )
        OmegaConf.update(cfg, "paths.output_dir", str(run_root), force_add=True)
        OmegaConf.update(cfg, "trainer.save_checkpoint_dir", str(run_root / "checkpoints"), force_add=True)
        OmegaConf.update(cfg, "ckpt_path", None, force_add=True)
        OmegaConf.update(cfg, "task.checkpoint_task", None, force_add=True)
        # Full .ckpt files carry both encoder and task weights. Avoid requiring the
        # old pretrain checkpoint path before the downstream state is loaded.
        if checkpoint.suffix == ".ckpt":
            OmegaConf.update(cfg, "task.checkpoint_encoder", None, force_add=True)
    return cfg


def _artifact_grouping_is_current(task: str, artifact: EvaluationArtifact) -> bool:
    # Existing pose/object artifacts already use their intended recording-level
    # groups. Only force changed semantics from whole recordings to contacts.
    expected = GROUPING_VERSION_BY_TASK.get(task) if task == "force" else None
    return expected is None or artifact.manifest.get("grouping_version") == expected


def _archive_stale_artifact(run_root: Path, artifact: EvaluationArtifact) -> None:
    """Preserve a stale artifact before an inference-only refresh."""
    evaluation_dir = Path(run_root) / "evaluation"
    archive_dir = evaluation_dir / "legacy"
    archive_dir.mkdir(parents=True, exist_ok=True)
    version = str(artifact.manifest.get("grouping_version") or "recording-groups-v1")
    for source, destination in (
        (evaluation_dir / "test_predictions.npz", archive_dir / f"test_predictions.{version}.npz"),
        (evaluation_dir / "manifest.json", archive_dir / f"manifest.{version}.json"),
    ):
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)


def _force_group_lookup(test_loader: Any) -> dict[int, int]:
    dataset = test_loader.dataset
    items = getattr(dataset, "idx_to_episode_idx", None)
    if items is None:
        raise TypeError("Force grouping migration requires a dataset with idx_to_episode_idx")
    lookup = {int(item["sample_id"]): int(item["group_id"]) for item in items}
    if not lookup:
        raise ValueError("Force test dataset produced no sample/group identifiers")
    return lookup


def _map_force_groups(sample_id: np.ndarray, lookup: dict[int, int]) -> np.ndarray:
    unique_ids = np.unique(np.asarray(sample_id, dtype=np.int64))
    missing = [int(value) for value in unique_ids if int(value) not in lookup]
    if missing:
        raise ValueError(
            f"Cannot migrate force grouping: {len(missing)} sample_id values are absent "
            f"from the current test dataset; first missing IDs: {missing[:5]}"
        )
    return np.fromiter(
        (lookup[int(value)] for value in np.asarray(sample_id).reshape(-1)),
        dtype=np.int64,
        count=len(sample_id),
    )


def _migrate_force_artifact_grouping(
    artifact: EvaluationArtifact,
    *,
    run_root: Path,
    config_path: Path,
    checkpoint: Path,
) -> EvaluationArtifact:
    lookup = next(
        (
            candidate
            for candidate in _FORCE_GROUP_LOOKUPS
            if all(int(value) in candidate for value in np.unique(artifact.sample_id))
        ),
        None,
    )
    if lookup is None:
        cfg = _prepare_config(config_path, run_root, checkpoint)
        _, _, test_loader = _loaders_for_task("force", cfg)
        lookup = _force_group_lookup(test_loader)
        _FORCE_GROUP_LOOKUPS.append(lookup)

    group_id = _map_force_groups(artifact.sample_id, lookup)
    _archive_stale_artifact(run_root, artifact)
    save_evaluation_artifact(
        run_root,
        task="force",
        sample_id=artifact.sample_id,
        group_id=group_id,
        y_true=artifact.y_true,
        y_pred=artifact.y_pred,
        checkpoint=str(checkpoint),
        seed=artifact.manifest.get("seed"),
        use_spatial_coords=artifact.manifest.get("use_spatial_coords"),
        extra_manifest={
            "config_path": artifact.manifest.get("config_path") or str(config_path),
            "grouping_migration": "predictions-preserved-from-recording-groups",
        },
    )
    return load_evaluation_artifact(run_root)


def backfill_evaluation_artifact(task: str, run_root: Path) -> Tuple[EvaluationArtifact, Path, Path]:
    """Evaluate the selected downstream checkpoint if its artifact is absent."""
    run_root = Path(run_root).expanduser().resolve()
    stale_artifact = None
    try:
        artifact = load_evaluation_artifact(run_root)
        if _artifact_grouping_is_current(task, artifact):
            checkpoint_value = artifact.manifest.get("checkpoint")
            checkpoint = Path(checkpoint_value) if checkpoint_value else select_checkpoint(run_root)
            config_path = find_run_config(run_root)
            return artifact, checkpoint, config_path
        stale_artifact = artifact
    except FileNotFoundError:
        pass

    checkpoint = select_checkpoint(run_root)
    config_path = find_run_config(run_root)
    if task == "force" and stale_artifact is not None:
        artifact = _migrate_force_artifact_grouping(
            stale_artifact,
            run_root=run_root,
            config_path=config_path,
            checkpoint=checkpoint,
        )
        return artifact, checkpoint, config_path

    cfg = _prepare_config(config_path, run_root, checkpoint)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, _, test_loader = _loaders_for_task(task, cfg)
    model = hydra.utils.instantiate(cfg.task)
    trainer = Trainer(tb_logger=NullSummaryWriter(), **cfg.trainer)
    trainer.evaluate(
        model,
        test_loader,
        ckpt_path_to_eval=str(checkpoint),
        train_loader_for_initialization=train_loader,
    )
    artifact = load_evaluation_artifact(run_root)
    return artifact, checkpoint, config_path

"""Backfill missing test artifacts without starting a training loop."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Tuple

import hydra
import numpy as np
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from tactile_ssl.trainer import Trainer

from .artifacts import EvaluationArtifact, load_evaluation_artifact


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
    OmegaConf.resolve(cfg)
    return module.get_dataloaders(cfg)


def _prepare_config(config_path: Path, run_root: Path, checkpoint: Path) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        OmegaConf.update(cfg, "paths.output_dir", str(run_root), force_add=True)
        OmegaConf.update(cfg, "trainer.save_checkpoint_dir", str(run_root / "checkpoints"), force_add=True)
        OmegaConf.update(cfg, "ckpt_path", None, force_add=True)
        OmegaConf.update(cfg, "task.checkpoint_task", None, force_add=True)
        # Full .ckpt files carry both encoder and task weights. Avoid requiring the
        # old pretrain checkpoint path before the downstream state is loaded.
        if checkpoint.suffix == ".ckpt":
            OmegaConf.update(cfg, "task.checkpoint_encoder", None, force_add=True)
    return cfg


def backfill_evaluation_artifact(task: str, run_root: Path) -> Tuple[EvaluationArtifact, Path, Path]:
    """Evaluate the selected downstream checkpoint if its artifact is absent."""
    run_root = Path(run_root).expanduser().resolve()
    try:
        artifact = load_evaluation_artifact(run_root)
        checkpoint_value = artifact.manifest.get("checkpoint")
        checkpoint = Path(checkpoint_value) if checkpoint_value else select_checkpoint(run_root)
        config_path = find_run_config(run_root)
        return artifact, checkpoint, config_path
    except FileNotFoundError:
        pass

    checkpoint = select_checkpoint(run_root)
    config_path = find_run_config(run_root)
    cfg = _prepare_config(config_path, run_root, checkpoint)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, _, test_loader = _loaders_for_task(task, cfg)
    model = hydra.utils.instantiate(cfg.task)
    trainer = Trainer(tb_logger=NullSummaryWriter(), **cfg.trainer)
    trainer.evaluate(model, test_loader, ckpt_path_to_eval=str(checkpoint))
    artifact = load_evaluation_artifact(run_root)
    return artifact, checkpoint, config_path

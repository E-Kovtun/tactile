#!/usr/bin/env python3
"""Downstream entrypoint that separates dataset and optimization RNG seeds."""

from __future__ import annotations

from datetime import datetime
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf

from tactile_ssl.trainer import Trainer
from tactile_ssl.utils.logging import get_pylogger, print_config_tree


logger = get_pylogger(__name__)
os.environ.setdefault("TACTILE_RUN_TIMESTAMP", datetime.now().strftime("%Y.%m.%d_%H-%M"))
OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("join", lambda separator, values: separator.join(map(str, values)), replace=True)


def source_module(kind: str):
    # Hydra needs these resolvers while composing this wrapper's config, but the
    # legacy task entrypoints register the same names again at import time.
    # Hand ownership back to the selected entrypoint before importing it.
    OmegaConf.clear_resolver("int_multiply")
    OmegaConf.clear_resolver("join")
    if kind == "force":
        import train_task_force as module
    elif kind == "object":
        import train_task_object as module
    elif kind == "pose":
        import train_task_pose_estimation as module
    else:
        raise ValueError(f"Unknown queue_entrypoint={kind!r}")
    return module


@hydra.main(version_base="1.3", config_path="../config", config_name="default_task.yaml")
def main(cfg: DictConfig) -> None:
    module = source_module(str(cfg.queue_entrypoint))
    writer = module.init_tensorboard(cfg.tensorboard)
    OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    head_seed = int(cfg.seed)
    data_seed = int(cfg.get("data_seed", head_seed))
    seed_everything(data_seed, workers=True)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    torch.backends.cudnn.benchmark = True
    train_dataloader, val_dataloader, test_dataloader = module.get_dataloaders(cfg)

    # All stochastic data selection is complete; now initialize the head run.
    seed_everything(head_seed, workers=True)
    np.random.seed(head_seed)
    torch.manual_seed(head_seed)
    model = hydra.utils.instantiate(cfg.task)
    trainer = Trainer(tb_logger=writer, **cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)
    trainer.evaluate(model, test_dataloader)
    writer.close()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

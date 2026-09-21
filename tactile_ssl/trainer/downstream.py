"""Shared downstream training lifecycle; task entrypoints supply their loaders."""

import os
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning.fabric import seed_everything
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from tactile_ssl.trainer import Trainer
from tactile_ssl.utils.logging import print_config_tree

os.environ.setdefault("TACTILE_RUN_TIMESTAMP", datetime.now().strftime("%Y.%m.%d_%H-%M"))
for name, resolver in {
    "int_multiply": lambda a, b: int(a * b),
    "int_divide": lambda a, b: a // b,
    "join": lambda separator, values: separator.join(map(str, values)),
}.items():
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


def init_tensorboard(cfg):
    return SummaryWriter(log_dir=cfg.log_dir)


def train_downstream(cfg, get_dataloaders, *, evaluate_saved_checkpoint=False):
    """Build data before the head, then fit and evaluate using the task protocol."""
    writer = init_tensorboard(cfg.tensorboard)
    try:
        OmegaConf.save(cfg, str(Path(cfg.paths.output_dir) / "config.yaml"))
        print_config_tree(cfg, resolve=True, save_to_file=True)
        data_seed = cfg.get("data_seed", cfg.seed)
        seed_everything(data_seed, workers=True)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        torch.backends.cudnn.benchmark = True
        train_loader, val_loader, test_loader = get_dataloaders(cfg)

        if "data_seed" in cfg:
            seed_everything(cfg.seed, workers=True)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
        model = hydra.utils.instantiate(cfg.task)
        trainer = Trainer(tb_logger=writer, **cfg.trainer)
        trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)
        if evaluate_saved_checkpoint:
            if trainer.use_early_stopping:
                checkpoint = os.path.join(
                    trainer.checkpoint_dir,
                    f"{trainer.early_stopping_checkpoint_name}.ckpt",
                )
            else:
                checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
            if checkpoint is None or not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"No final downstream checkpoint found in {trainer.checkpoint_dir}"
                )
            trainer.evaluate(model, test_loader, ckpt_path_to_eval=checkpoint)
        else:
            trainer.evaluate(model, test_loader)
    finally:
        writer.close()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.utils.data as data
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from tactile_ssl.data.d360.utils import get_experiment_name, get_modality_tag
from tactile_ssl.trainer import Trainer
from tactile_ssl.utils.logging import get_pylogger, print_config_tree

logger = get_pylogger(__name__)

os.environ.setdefault("TACTILE_RUN_TIMESTAMP", datetime.now().strftime("%Y.%m.%d_%H-%M"))

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)


def init_tensorboard(cfg: DictConfig):
    return SummaryWriter(log_dir=cfg.log_dir)


def get_dataloaders_xela_joystick(cfg: DictConfig):
    train_dset, val_dset = hydra.utils.instantiate(cfg.data.dataset)

    if hasattr(cfg.data, "max_train_data"):
        train_dset_size = min(len(train_dset), cfg.data.max_train_data)
        train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    if hasattr(cfg.data, "max_val_data"):
        val_dset_size = min(len(val_dset), cfg.data.max_val_data)
        val_dset, _ = data.random_split(val_dset, [val_dset_size, len(val_dset) - val_dset_size])

    if hasattr(cfg.data, "val_data_budget"):
        val_dset_size = int(len(val_dset) * cfg.data.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [val_dset_size, len(val_dset) - val_dset_size])

    print("Dataset sizes")
    print(f"\t Train dataset size: {len(train_dset)}")
    print(f"\t Val dataset size: {len(val_dset)}")

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader


def get_dataloaders(cfg: DictConfig):
    if cfg.data.sensor != "xela":
        raise NotImplementedError("train_task_joystick.py currently supports only Xela joystick control")
    return get_dataloaders_xela_joystick(cfg)


def attempt_resume(cfg: DictConfig):
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(f"Unable to resume: No checkpoints found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning("Could not find a config.yaml file in the resume directory. Using the current config.")
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")
        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        logger.info(f"Resuming experiment {job_id} from latest checkpoint at {cfg.ckpt_path}")
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating tensorboard ...")
    writer = init_tensorboard(cfg.tensorboard)

    if not resume_state:
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    trainer = Trainer(tb_logger=writer, **cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    writer.close()


@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

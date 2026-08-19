# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
from datetime import datetime
import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

import wandb
from lightning.fabric import seed_everything

from tactile_ssl.utils import get_local_rank

from tactile_ssl.utils.logging import get_pylogger, print_config_tree
from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag, get_modality_used_tag
from tactile_ssl.trainer import Trainer
from tactile_ssl.utils.combined_dataset import CombinedDataset

logger = get_pylogger(__name__)

os.environ.setdefault("TACTILE_RUN_TIMESTAMP", datetime.now().strftime("%Y.%m.%d_%H-%M"))

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("join", lambda separator, values: separator.join(map(str, values)))


def init_tensorboard(cfg: DictConfig):
    writer = SummaryWriter(log_dir=cfg.log_dir)
    return writer


def get_pose_estimation_dataloader_xela(cfg: DictConfig):
    data_cfg = cfg.data
    train_dset, val_dset, test_dset = hydra.utils.instantiate(data_cfg.dataset)

    if data_cfg.get("sensor") == "sock":
        with open_dict(cfg):
            cfg.data.normalization.mean = train_dset.input_mean.tolist()
            cfg.data.normalization.std = train_dset.input_std.tolist()

    if hasattr(data_cfg, "max_train_data"):
        train_dset_size = min(len(train_dset), data_cfg.max_train_data)
        train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    print("Dataset sizes")
    print(f"\t Train dataset size: {len(train_dset)}")
    print(f"\t Val dataset size: {len(val_dset)}")
    print(f"\t Test dataset size: {len(test_dset)}")

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    test_dataloader = data.DataLoader(test_dset, **cfg.data.test_dataloader)

    return train_dataloader, val_dataloader, test_dataloader


def get_dataloaders(cfg: DictConfig):
    data_cfg = cfg.data
    train_dataloader, val_dataloader, test_dataloader = get_pose_estimation_dataloader_xela(cfg)
    return train_dataloader, val_dataloader, test_dataloader

def train(cfg: DictConfig):

    logger.info("Instantiating tensorboard ...")
    writer = init_tensorboard(cfg.tensorboard)
    
    OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    trainer = Trainer(tb_logger=writer, **cfg.trainer)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)
    trainer.evaluate(model, test_dataloader)

    writer.close()


# @hydra.main(version_base="1.3", config_path="config")
@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

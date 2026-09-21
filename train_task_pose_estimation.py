# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import hydra
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.trainer.downstream import init_tensorboard, train_downstream


def get_pose_estimation_dataloader_xela(cfg: DictConfig):
    data_cfg = cfg.data
    if data_cfg.get("sensor", "xela") != "xela":
        raise ValueError("Expected Xela data; use train_task_socks_pose.py for Socks.")
    train_dset, val_dset, test_dset = hydra.utils.instantiate(data_cfg.dataset)

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
    train_dataloader, val_dataloader, test_dataloader = get_pose_estimation_dataloader_xela(cfg)
    return train_dataloader, val_dataloader, test_dataloader

def train(cfg: DictConfig):
    train_downstream(cfg, get_dataloaders)


@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

"""Train and evaluate tactile-socks pose estimation."""

import hydra
import torch
import torch.utils.data as data
from omegaconf import DictConfig, open_dict

from tactile_ssl.trainer.downstream import train_downstream


def get_dataloaders(cfg: DictConfig):
    if cfg.data.sensor != "sock":
        raise ValueError("Expected a sock dataset configuration.")
    train_dset, val_dset, test_dset = hydra.utils.instantiate(cfg.data.dataset)
    with open_dict(cfg):
        cfg.data.normalization.mean = train_dset.input_mean.tolist()
        cfg.data.normalization.std = train_dset.input_std.tolist()
    if "max_train_data" in cfg.data:
        size = min(len(train_dset), cfg.data.max_train_data)
        train_dset, _ = data.random_split(train_dset, [size, len(train_dset) - size])
    return (
        data.DataLoader(train_dset, **cfg.data.train_dataloader),
        data.DataLoader(val_dset, **cfg.data.val_dataloader),
        data.DataLoader(test_dset, **cfg.data.test_dataloader),
    )


def train(cfg: DictConfig):
    from tactile_ssl.utils.run_config import validate_run_config
    validate_run_config(cfg, "train_task_socks_pose.py")
    train_downstream(cfg, get_dataloaders)


@hydra.main(version_base="1.3", config_path="config", config_name="socks/pose/downstream/jepa")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

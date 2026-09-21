"""Train and evaluate the DECO visuo-tactile policy."""

import hydra
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.trainer.downstream import train_downstream


def get_dataloaders(cfg: DictConfig):
    if cfg.data.sensor != "deco":
        raise ValueError("Expected a deco dataset configuration.")
    train_dset, val_dset = hydra.utils.instantiate(cfg.data.dataset)
    test_dset = train_dset.test_dataset
    return (
        data.DataLoader(train_dset, **cfg.data.train_dataloader),
        data.DataLoader(val_dset, **cfg.data.val_dataloader),
        data.DataLoader(test_dset, **cfg.data.val_dataloader),
    )


def train(cfg: DictConfig):
    train_downstream(cfg, get_dataloaders, evaluate_saved_checkpoint=True)


@hydra.main(version_base="1.3", config_path="config", config_name="deco/policy/jepa")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

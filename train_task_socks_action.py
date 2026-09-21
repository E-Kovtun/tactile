"""Train and evaluate tactile-socks action classification."""

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
        cfg.data.object_classes = list(train_dset.classes)
        cfg.data.object_class_weights = train_dset.class_weights.tolist()
    return (
        data.DataLoader(train_dset, **cfg.data.train_dataloader),
        data.DataLoader(val_dset, **cfg.data.val_dataloader),
        data.DataLoader(test_dset, **cfg.data.val_dataloader),
    )


def train(cfg: DictConfig):
    train_downstream(cfg, get_dataloaders, evaluate_saved_checkpoint=True)


@hydra.main(version_base="1.3", config_path="config/paper", config_name="socks/action/downstream/jepa")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

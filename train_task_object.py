# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Optional
import os

import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig, open_dict

from tactile_ssl.data.xela.preprocessing import compute_cached_xela_normalization
from tactile_ssl.data.subsets import deterministic_nested_fractional_subsets
from tactile_ssl.trainer.downstream import init_tensorboard, train_downstream
from tactile_ssl.utils.logging import get_pylogger

logger = get_pylogger(__name__)


def get_xela_dataset(dataset_cfg: DictConfig, dataset_name: str, d_id: int, object_class: Optional[int] = None):
    data_path = f"{dataset_cfg.data_path}"
    data_files = os.listdir(data_path)
    dataset_name_exists = True in [f in f"{dataset_name}" for f in data_files]
    if not dataset_name_exists:
        print(f"Dataset {dataset_name} not found")
        return None
    dataset = hydra.utils.instantiate(
        dataset_cfg,
        data_path=f"{data_path}/{dataset_name}/{d_id}",
        object_class=object_class,
    )
    return dataset


def get_dataloaders_magnetic_based(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor != "xela":
        raise ValueError("Expected Xela data; use the Socks or DECO task entrypoint.")

    def instantiate_xela_tasks(tasks):
        cache_cfg = data_cfg.get("cache", {})
        num_workers = int(cache_cfg.get("num_workers", 0))
        if num_workers > 0 and len(tasks) > 1:
            max_workers = min(num_workers, len(tasks))
            logger.info(f"Loading Xela object datasets with {max_workers} cache workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(lambda task: get_xela_dataset(*task), tasks))
        return [get_xela_dataset(*task) for task in tasks]

    train_datasets, val_datasets, test_datasets = [], [], []
    dataset_list: List = data_cfg.dataset_list
    object_classes = []
    object_class_sizes = []
    for dataset_l in dataset_list:
        assert dataset_l.type == "teleop"
        train_dataset_ids = dataset_l.train_dataset_ids
        val_dataset_ids = dataset_l.val_dataset_ids
        test_dataset_ids = dataset_l.test_dataset_ids
        train_tasks, val_tasks, test_tasks = [], [], []
        for obj in dataset_l.sequence_list:
            object_class = len(object_classes)
            object_classes.append(obj)
            object_class_sizes.append(0)
            train_tasks.extend(
                (deepcopy(dataset_l.dataset), obj, d_id, object_class)
                for d_id in train_dataset_ids
            )
            val_tasks.extend(
                (deepcopy(dataset_l.dataset), obj, d_id, object_class)
                for d_id in val_dataset_ids
            )
            test_tasks.extend(
                (deepcopy(dataset_l.dataset), obj, d_id, object_class)
                for d_id in test_dataset_ids
            )
        for dataset in instantiate_xela_tasks(train_tasks):
            if dataset is not None:
                object_class_sizes[dataset.object_label] += len(dataset)
                train_datasets.append(dataset)
        val_datasets.extend(
            dataset for dataset in instantiate_xela_tasks(val_tasks) if dataset is not None
        )
        test_datasets.extend(
            dataset for dataset in instantiate_xela_tasks(test_tasks) if dataset is not None
        )

    print(f"Object class sizes: {object_class_sizes}")
    object_class_sizes = np.asarray(object_class_sizes)
    object_class_ratios = object_class_sizes / np.sum(object_class_sizes)
    object_class_weights = 1 / object_class_ratios
    object_class_weights = object_class_weights / np.sum(object_class_weights)
    print(f"Object class weights: {object_class_weights}")

    xela_mean, xela_std = compute_cached_xela_normalization(train_datasets)
    logger.info(f"Compute Xela normalization: mean={xela_mean}, std={xela_std}")

    with open_dict(cfg):
        cfg.data.normalization.mean = xela_mean.tolist()
        cfg.data.normalization.std = xela_std.tolist()
        cfg.data.object_classes = object_classes
        cfg.data.object_class_weights = object_class_weights.tolist()

    for dataset in train_datasets + val_datasets + test_datasets:
        dataset.update_normalization(xela_mean, xela_std)

    full_train_size = sum(len(dataset) for dataset in train_datasets)
    train_data_budget = float(data_cfg.get("train_data_budget", 1.0))
    subset_seed = int(cfg.get("data_seed", cfg.seed))
    train_datasets = deterministic_nested_fractional_subsets(
        train_datasets,
        fraction=train_data_budget,
        seed=subset_seed,
    )
    selected_train_size = sum(len(dataset) for dataset in train_datasets)
    logger.info(
        "Xela object train budget: requested=%.4f selected=%d/%d (%.4f), seed=%d",
        train_data_budget,
        selected_train_size,
        full_train_size,
        selected_train_size / full_train_size,
        subset_seed,
    )
    train_dset = data.ConcatDataset(train_datasets)
    val_dset = data.ConcatDataset(val_datasets)
    test_dset = data.ConcatDataset(test_datasets)

    return train_dset, val_dset, test_dset


def get_dataloaders(cfg: DictConfig):
    train_dset, val_dset, test_dset = get_dataloaders_magnetic_based(cfg)
    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    test_dataloader = data.DataLoader(test_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader, test_dataloader


def train(cfg: DictConfig):
    from tactile_ssl.utils.run_config import validate_run_config
    validate_run_config(cfg, "train_task_object.py")
    train_downstream(cfg, get_dataloaders, evaluate_saved_checkpoint=True)


@hydra.main(version_base="1.3", config_path="config", config_name="xela/object/jepa")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import os
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.utils.data as data
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict
from copy import deepcopy

import wandb
from torch.utils.tensorboard import SummaryWriter

from tactile_ssl.trainer import Trainer  # noqa: E402
from tactile_ssl.utils import get_local_rank, get_node_id
from tactile_ssl.utils.logging import get_pylogger, print_config_tree  # noqa: E402
from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag
from tactile_ssl.utils.combined_dataset import CombinedDataset

logger = get_pylogger(__name__)

os.environ.setdefault("TACTILE_RUN_TIMESTAMP", datetime.now().strftime("%Y.%m.%d_%H-%M"))

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)


def init_tensorboard(cfg: DictConfig):
    writer = SummaryWriter(log_dir=cfg.log_dir)
    return writer


def get_dataloaders_magnetic_based(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor == "xela":

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

        def instantiate_xela_tasks(tasks):
            cache_cfg = data_cfg.get("cache", {})
            num_workers = int(cache_cfg.get("num_workers", 0))
            if num_workers > 0 and len(tasks) > 1:
                max_workers = min(num_workers, len(tasks))
                logger.info(f"Loading Xela datasets with {max_workers} cache workers")
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    return list(pool.map(lambda task: get_xela_dataset(*task), tasks))
            return [get_xela_dataset(*task) for task in tasks]

        train_datasets, val_datasets = [], []
        dataset_list: List = data_cfg.dataset_list
        object_classes = []
        object_class_sizes = []
        for dataset_l in dataset_list:
            if dataset_l.type == "teleop":
                train_dataset_ids, val_dataset_ids = (
                    dataset_l.train_dataset_ids,
                    dataset_l.val_dataset_ids,
                )
                train_tasks, val_tasks = [], []
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
                for dataset in instantiate_xela_tasks(train_tasks):
                    if dataset is not None:
                        object_class_sizes[dataset.object_label] += len(dataset)
                        train_datasets.append(dataset)
                val_datasets.extend(dataset for dataset in instantiate_xela_tasks(val_tasks) if dataset is not None)
            elif dataset_l.type == "joystick_control":
                with open_dict(dataset_l.dataset.config):
                    dataset_l.dataset.config.object_label = len(object_classes)
                train_dset, val_dset = hydra.utils.instantiate(dataset_l.dataset)
                object_classes.append("joystick")
                object_class_sizes.append(len(train_dset))
                train_datasets.append(train_dset)
                val_datasets.append(val_dset)

        print(f"Object class sizes: {object_class_sizes}")
        object_class_sizes = np.asarray(object_class_sizes)
        object_class_ratios = object_class_sizes / np.sum(object_class_sizes)
        object_class_weights = 1 / object_class_ratios
        object_class_weights = object_class_weights / np.sum(object_class_weights)
        print(f"Object class weights: {object_class_weights}")

        from tactile_ssl.data.xela.preprocessing import compute_cached_xela_normalization

        xela_mean, xela_std = compute_cached_xela_normalization(train_datasets)
        logger.info(f"Compute Xela normalization: mean={xela_mean}, std={xela_std}")

        with open_dict(cfg):
            cfg.data.normalization.mean = xela_mean.tolist()
            cfg.data.normalization.std = xela_std.tolist()
            cfg.data.object_classes = object_classes
            cfg.data.object_class_weights = object_class_weights.tolist()

        for dataset in train_datasets + val_datasets:
            dataset.update_normalization(xela_mean, xela_std)
        train_dset = data.ConcatDataset(train_datasets)
        val_dset = data.ConcatDataset(val_datasets)

    elif data_cfg.sensor == "sock":
        train_dset, val_dset = hydra.utils.instantiate(data_cfg.dataset)
        with open_dict(cfg):
            cfg.data.normalization.mean = train_dset.input_mean.tolist()
            cfg.data.normalization.std = train_dset.input_std.tolist()
    elif data_cfg.sensor == "tdex":
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * cfg.data.train_val_split)

        train_dset, val_dset = data.random_split(dataset, [train_dset_size, len(dataset) - train_dset_size])
    else:
        raise NotImplementedError

    return train_dset, val_dset


def get_dataloaders_d360_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_main_datasets, val_main_datasets = [], []
    train_supp_datasets, val_supp_datasets = [], []

    dataset_list = data_cfg.dataset_list
    online_probe_dataset_list = (
        data_cfg.online_probe_dataset_list if data_cfg.online_probe_dataset_list is not None else []
    )
    object_labels = []
    action_labels = []
    sizes = []
    for sequence in dataset_list:
        full_dataset = hydra.utils.instantiate(data_cfg.sensor.d360.dataset, sequences=[sequence])
        online_probe_used = sequence in online_probe_dataset_list
        if online_probe_used:
            action_labels.append(full_dataset.action_label[0])
            object_labels.append(full_dataset.object_label[0])
            sizes.append(full_dataset.sequence_sizes[0])

        train_dataset, val_dataset = None, None
        if data_cfg.split.type == "device":
            train_dataset = full_dataset
            val_dataset = deepcopy(train_dataset)
            num_devices = len(train_dataset.imu_acc[0])
            devices = list(range(num_devices))
            device = np.random.randint(len(devices))
            train_dataset.remove(0, [device])
            val_dataset.remove(0, devices[:device] + devices[device + 1 :])
        elif data_cfg.split.type == "random":
            ratio = data_cfg.split.ratio
            train_size = int(ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        elif data_cfg.split.type == "sequence":
            if sequence not in data_cfg.split.test_dataset_list:
                train_dataset = full_dataset
            else:
                val_dataset = full_dataset
        else:
            raise NotImplementedError

        if train_dataset is not None:
            train_main_datasets.append(train_dataset)
            if online_probe_used:
                train_supp_datasets.append(train_dataset)

        if val_dataset is not None:
            val_main_datasets.append(val_dataset)
            if online_probe_used:
                val_supp_datasets.append(val_dataset)

    object_weights = get_weights(object_labels, sizes)
    action_weights = get_weights(action_labels, sizes)

    if not object_weights:
        object_weights["default"] = 1

    if not action_weights:
        action_weights["default"] = 1

    with open_dict(cfg):
        for _, modality in cfg.data.sensor.d360.dataset.config.normalization.items():
            if "normalize" in modality:
                normalize = hydra.utils.instantiate(modality["normalize"])
                modal_avg, modal_std = normalize(train_main_datasets + val_main_datasets)
                modality["avg"] = modal_avg.tolist()
                modality["std"] = modal_std.tolist()
        cfg.data.sensor.d360.dataset.config.data.label.object.cls = list(object_weights.keys())
        cfg.data.sensor.d360.dataset.config.data.label.object.weight = list(object_weights.values())
        cfg.data.sensor.d360.dataset.config.data.label.action.cls = list(action_weights.keys())
        cfg.data.sensor.d360.dataset.config.data.label.action.weight = list(action_weights.values())

    if data_cfg.split.type == "device" or data_cfg.split.type == "sequence":
        for dataset in train_main_datasets + val_main_datasets:
            dataset.update_label_cls("object", object_weights)
            dataset.update_label_cls("action", action_weights)
            dataset.update_normalization(cfg.data.sensor.d360.dataset.config.normalization)
    elif data_cfg.split.type == "random":
        for dataset in train_main_datasets + val_main_datasets:
            dataset.dataset.update_label_cls("object", object_weights)
            dataset.dataset.update_label_cls("action", action_weights)
            dataset.dataset.update_normalization(cfg.data.sensor.d360.dataset.config.normalization)

    train_main_dset = data.ConcatDataset(train_main_datasets)
    val_main_dset = data.ConcatDataset(val_main_datasets)

    train_supp_dset = data.ConcatDataset(train_supp_datasets) if train_supp_datasets else None
    train_dset = CombinedDataset(main_dataset=train_main_dset, supp_dataset=train_supp_dset)

    val_supp_dset = data.ConcatDataset(val_supp_datasets) if val_supp_datasets else None
    val_dset = CombinedDataset(main_dataset=val_main_dset, supp_dataset=val_supp_dset)

    return train_dset, val_dset


def get_dataloaders(cfg: DictConfig):
    if "d360" in cfg.data.sensor:
        train_dset, val_dset = get_dataloaders_d360_based(cfg)
    elif cfg.data.sensor == "deco":
        train_dset, val_dset = hydra.utils.instantiate(cfg.data.dataset)
    elif cfg.data.sensor in ["xela", "sock", "tdex"]:
        train_dset, val_dset = get_dataloaders_magnetic_based(cfg)
    else:
        raise NotImplementedError(f"Sensor type {cfg.data.sensor} is not supported.")

    def build_dataloader(dataset, loader_cfg):
        loader_kwargs = dict(loader_cfg)
        collate_cfg = loader_kwargs.get("collate_fn")
        if isinstance(collate_cfg, (DictConfig, dict)):
            loader_kwargs["collate_fn"] = hydra.utils.instantiate(collate_cfg)
        batch_sampler_cfg = loader_kwargs.pop("batch_sampler", None)
        if isinstance(batch_sampler_cfg, (DictConfig, dict)):
            loader_kwargs["batch_sampler"] = hydra.utils.instantiate(
                batch_sampler_cfg, dataset=dataset
            )
            for incompatible_key in ("batch_size", "shuffle", "sampler", "drop_last"):
                loader_kwargs.pop(incompatible_key, None)
        return data.DataLoader(dataset, **loader_kwargs)

    train_dataloader = build_dataloader(train_dset, cfg.data.train_dataloader)
    val_dataloader = build_dataloader(val_dset, cfg.data.val_dataloader)

    return train_dataloader, val_dataloader


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.environ.get("SLURM_RESTART_COUNT") is not None:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        requeue_count = os.environ.get("SLURM_RESTART_COUNT")
        logger.info(f"requeue count for job {slurm_job_id}: {requeue_count}")
        cfg.resume_id = slurm_job_id
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(f"Unable to resume: No checkpoints found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/wandb/"):
            logger.warning(f"Unable to resume: No wandb logs found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning("Could not find a config.yaml file in the resume directory. Using the current config.")
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")

        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        experiment_name = cfg.experiment_name
        cfg.wandb.id = f"{job_id}_{experiment_name}"
        logger.info(
            f"Resuming experiment {job_id} with wandb_id: {cfg.wandb.id} from latest checkpoint at {cfg.ckpt_path}"
        )
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)
    logger.info(f"Resume state: {resume_state}, {cfg.ckpt_path}")
    logger.info("Instantiating tensorboard ...")
    writer = init_tensorboard(cfg.tensorboard)
    if ~resume_state:
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    n_sensors = len(cfg.data.sensor) if OmegaConf.is_list(cfg.data.sensor) or OmegaConf.is_dict(cfg.data.sensor) else 1
    if OmegaConf.is_dict(cfg.data.sensor):
        sensors_type = [sensor for sensor in cfg.data.sensor]
    else:
        sensors_type = [cfg.data.sensor[i].type for i in range(n_sensors)] if n_sensors > 1 else cfg.data.sensor
    logger.info(f"Instantiating dataset & dataloaders for <{sensors_type}>")
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    trainer = Trainer(tb_logger=writer, **cfg.trainer)

    logger.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm = hydra.utils.instantiate(cfg.algorithm)

    trainer.fit(algorithm, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    writer.close()


@hydra.main(version_base="1.3", config_path="config", config_name="xela/pretrain/jepa")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    logger.info(f"# of nodes: {num_nodes}")
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

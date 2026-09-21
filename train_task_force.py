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
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)
OmegaConf.register_new_resolver("d360_modal_used_tag", get_modality_used_tag)
OmegaConf.register_new_resolver("capitalize", lambda s: s.title())
OmegaConf.register_new_resolver("join", lambda separator, values: separator.join(map(str, values)))


def init_tensorboard(cfg: DictConfig):
    writer = SummaryWriter(log_dir=cfg.log_dir)
    return writer


def get_dataloader_xela(cfg: DictConfig):
    data_cfg = cfg.data

    train_dset, val_dset = hydra.utils.instantiate(data_cfg.dataset)

    if val_dset is None:
        train_dset_size = int(len(train_dset) * 0.8)
        train_dset, val_dset = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    if hasattr(data_cfg, "max_train_data"):
        train_dset_size = min(len(train_dset), data_cfg.max_train_data)
        train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    print("Original dataset sizes")
    print(f"\t Train dataset size: {len(train_dset)}")
    print(f"\t Val dataset size: {len(val_dset)}")

    # adjust training dataset size
    train_dset_size = int(len(train_dset) * data_cfg.train_data_budget)
    train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    val_dset_size = int(len(val_dset) * data_cfg.val_data_budget)
    val_dset, _ = data.random_split(val_dset, [val_dset_size, len(val_dset) - val_dset_size])

    sampler_cfg = cfg.data.get("sampler", None)
    if sampler_cfg is not None:
        train_sampler = hydra.utils.instantiate(sampler_cfg, dataset=train_dset.dataset)
        val_sampler = hydra.utils.instantiate(sampler_cfg, dataset=val_dset.dataset)
        train_dataloader = data.DataLoader(train_dset, sampler=train_sampler, **cfg.data.train_dataloader)
        val_dataloader = data.DataLoader(val_dset, sampler=val_sampler, **cfg.data.val_dataloader)
        return train_dataloader, val_dataloader

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader


def get_train_val_test_dataloader_xela(cfg: DictConfig):
    data_cfg = cfg.data

    train_dset, test_dset = hydra.utils.instantiate(data_cfg.dataset)

    train_dset_size = int(len(train_dset) * data_cfg.train_val_ratio)
    train_dset, val_dset = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    if hasattr(data_cfg, "max_train_data"):
        train_dset_size = min(len(train_dset), data_cfg.max_train_data)
        train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    if hasattr(data_cfg, "max_val_data"):
        val_dset_size = min(len(val_dset), data_cfg.max_val_data)
        val_dset, _ = data.random_split(val_dset, [val_dset_size, len(val_dset) - val_dset_size])

    # adjust training dataset size
    train_dset_size = int(len(train_dset) * data_cfg.train_data_budget)
    train_dset, _ = data.random_split(train_dset, [train_dset_size, len(train_dset) - train_dset_size])

    val_dset_size = int(len(val_dset) * data_cfg.val_data_budget)
    val_dset, _ = data.random_split(val_dset, [val_dset_size, len(val_dset) - val_dset_size])

    print("Dataset sizes")
    print(f"\t Train dataset size: {len(train_dset)}")
    print(f"\t Val dataset size: {len(val_dset)}")
    print(f"\t Test dataset size: {len(test_dset)}")

    # sampler_cfg = cfg.data.get("sampler", None)
    # if sampler_cfg is not None:
    #     train_sampler = hydra.utils.instantiate(sampler_cfg, dataset=train_dset.dataset)
    #     val_sampler = hydra.utils.instantiate(sampler_cfg, dataset=val_dset.dataset)
    #     train_dataloader = data.DataLoader(train_dset, sampler=train_sampler, **cfg.data.train_dataloader)
    #     val_dataloader = data.DataLoader(val_dset, sampler=val_sampler, **cfg.data.val_dataloader)
    #     return train_dataloader, val_dataloader

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    test_dataloader = data.DataLoader(test_dset, **cfg.data.test_dataloader)
    return train_dataloader, val_dataloader, test_dataloader


def get_dataloaders_d360_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_main_datasets, val_main_datasets = [], []

    dataset_list = data_cfg.dataset.config.dataset_list
    for sequence in dataset_list:
        train_main_datasets.append(hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence]))

    dataset_list_test = data_cfg.dataset.config.dataset_list_test
    if len(dataset_list_test) > 0:
        for sequence in dataset_list_test:
            val_main_datasets.append(hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence]))

    with open_dict(cfg):
        for _, modality in cfg.data.dataset.config.normalization.items():
            if "normalize" in modality:
                normalize = hydra.utils.instantiate(modality["normalize"])
                modal_avg, modal_std = normalize(train_main_datasets + val_main_datasets)
                modality["avg"] = modal_avg.tolist()
                modality["std"] = modal_std.tolist()

    for dataset in train_main_datasets + val_main_datasets:
        dataset.update_normalization(cfg.data.dataset.config.normalization)

    train_main_dset = data.ConcatDataset(train_main_datasets)
    if len(val_main_datasets) > 0:
        val_main_dset = data.ConcatDataset(val_main_datasets)
    else:
        train_dset_size = int(len(train_main_dset) * data_cfg.train_val_split)
        train_main_dset, val_main_dset = data.random_split(
            train_main_dset, [train_dset_size, len(train_main_dset) - train_dset_size]
        )

    # adjust training dataset size
    train_dset_size = int(len(train_main_dset) * data_cfg.train_data_budget)
    train_main_dset, _ = data.random_split(train_main_dset, [train_dset_size, len(train_main_dset) - train_dset_size])

    try:
        val_dset_size = int(len(val_main_dset) * data_cfg.val_data_budget)
        val_main_dset, _ = data.random_split(val_main_dset, [val_dset_size, len(val_main_dset) - val_dset_size])
    except Exception as e:
        logger.error(e)

    train_dataloader = data.DataLoader(train_main_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_main_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader


def get_dataloaders_d360_contact_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_datasets, val_datasets = [], []

    dataset_list = data_cfg.dataset.config.dataset_list
    for sequence in dataset_list:
        train_datasets.append(hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence]))

    dataset_list_test = data_cfg.dataset.config.dataset_list_test
    if len(dataset_list_test) > 0:
        for sequence in dataset_list_test:
            val_datasets.append(hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence]))

    with open_dict(cfg):
        for _, modality in cfg.data.dataset.config.normalization.items():
            if "normalize" in modality:
                normalize = hydra.utils.instantiate(modality["normalize"])
                modal_avg, modal_std = normalize(train_datasets + val_datasets)
                modality["avg"] = modal_avg.tolist()
                modality["std"] = modal_std.tolist()

    for dataset in train_datasets + val_datasets:
        dataset.update_normalization(cfg.data.dataset.config.normalization)

    assert len(val_datasets) > 0, "No validation datasets"
    train_dataset = data.ConcatDataset(train_datasets)
    val_dataset = data.ConcatDataset(val_datasets)

    contacts = [
        np.hstack(
            [
                np.array([dataset.get_dev_msg("force", index)[-1] for index in range(len(dataset))]) >= 0.1
                for dataset in datasets
            ]
        )
        for datasets in [train_datasets, val_datasets]
    ]

    no_contact_datasets = []
    on_contact_datasets = []
    for contact, dataset, budget in zip(
        contacts,
        [train_dataset, val_dataset],
        [data_cfg.train_data_budget, data_cfg.val_data_budget],
    ):
        no_contact_dataset = data.Subset(dataset, np.arange(len(dataset))[~contact])
        on_contact_dataset = data.Subset(dataset, np.arange(len(dataset))[contact])

        no_contact_main_dset_size = int(len(no_contact_dataset) * budget)
        no_contact_main_dset, _ = data.random_split(
            no_contact_dataset,
            [
                no_contact_main_dset_size,
                len(no_contact_dataset) - no_contact_main_dset_size,
            ],
        )
        no_contact_datasets.append(no_contact_main_dset)

        on_contact_main_dset_size = int(len(on_contact_dataset) * budget)
        on_contact_main_dset, _ = data.random_split(
            on_contact_dataset,
            [
                on_contact_main_dset_size,
                len(on_contact_dataset) - on_contact_main_dset_size,
            ],
        )
        on_contact_datasets.append(on_contact_main_dset)

    train_no_contact_dset, val_no_contact_dset = no_contact_datasets
    train_on_contact_dset, val_on_contact_dset = on_contact_datasets

    train_dset = CombinedDataset(main_dataset=train_no_contact_dset, supp_dataset=train_on_contact_dset)
    val_dset = CombinedDataset(main_dataset=val_no_contact_dset, supp_dataset=val_on_contact_dset)

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader


def get_dataloaders_d360_classification_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_datasets, val_datasets = [], []

    dataset_list = data_cfg.dataset.config.dataset_list
    classification = data_cfg.label.name
    sub_labels = classification.split("-")

    train_dset_labels = []
    train_dset_sizes = []

    for sequence in dataset_list:
        d360_dataset = hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence])
        label = "-".join([getattr(d360_dataset, f"{sub_label}_label")[0] for sub_label in sub_labels])
        train_dset_labels.append(label)
        train_dset_sizes.append(d360_dataset.sequence_sizes[0])
        train_datasets.append(d360_dataset)

    label_weights = get_weights(train_dset_labels, train_dset_sizes)

    if not label_weights:
        label_weights["default"] = 1

    dataset_list_test = data_cfg.dataset.config.dataset_list_test

    val_dset_labels = []
    if len(dataset_list_test) > 0:
        for sequence in dataset_list_test:
            d360_dataset = hydra.utils.instantiate(data_cfg.dataset, sequences=[sequence])
            label = "-".join([getattr(d360_dataset, f"{sub_label}_label")[0] for sub_label in sub_labels])
            if label not in train_dset_labels:
                logger.error(
                    f"The label of validation dataset {sequence} is {label} and not available in the training datasets"
                )
            val_dset_labels.append(label)
            val_datasets.append(d360_dataset)

    assert len(val_datasets) > 0, "No validation datasets are available"

    for dataset, label in zip(train_datasets + val_datasets, train_dset_labels + val_dset_labels):
        dataset.update_label_cls(classification, label_weights, [label])

    with open_dict(cfg):
        for _, modality in cfg.data.dataset.config.normalization.items():
            if "normalize" in modality:
                normalize = hydra.utils.instantiate(modality["normalize"])
                modal_avg, modal_std = normalize(train_datasets + val_datasets)
                modality["avg"] = modal_avg.tolist()
                modality["std"] = modal_std.tolist()
        cfg.data.label.cls = list(label_weights.keys())
        cfg.data.label.weight = list(label_weights.values())

    assert len(val_datasets) > 0, "No validation datasets"
    train_dataset = data.ConcatDataset(train_datasets)
    val_dataset = data.ConcatDataset(val_datasets)

    train_dataloader = data.DataLoader(train_dataset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dataset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader


def get_dataloaders(cfg: DictConfig):
    data_cfg = cfg.data

    if "d360_contact" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_contact_based(cfg)
    elif "d360_classification" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_classification_based(cfg)
    elif "d360" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_based(cfg)
    elif data_cfg.sensor == "xela":
        train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloader_xela(cfg)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        raise NotImplementedError("Sensor type not implemented yet.")
    return train_dataloader, val_dataloader


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(f"Unable to resume: No checkpoints found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/lightning_logs/"):
            logger.warning(f"Unable to resume: No tensorboard logs found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning("Could not find a config.yaml file in the resume directory. Using the current config.")
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")

        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        experiment_name = cfg.experiment_name
        logger.info(
            f"Resuming experiment {job_id} from latest checkpoint at {cfg.ckpt_path}"
        )
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating tensorboard ...")
    writer = init_tensorboard(cfg.tensorboard)
    
    if not resume_state:
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    data_seed = cfg.get("data_seed", cfg.seed)
    if data_seed is not None:
        seed_everything(data_seed, workers=True)
    _GLOBAL_SEED = data_seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)

    # Keep dataset selection fixed while varying the downstream head seed.
    if "data_seed" in cfg:
        seed_everything(cfg.seed, workers=True)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

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

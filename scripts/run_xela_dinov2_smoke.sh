#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/scripts/outputs/smoke_xela_dinov2"
PRETRAIN_CKPT_DIR="${ROOT}/experiments/pretrain_gat_encoder_dinov2/2026.07.03_09-13/checkpoints"

cd "${ROOT}"
mkdir -p "${OUT}"

python train.py +experiment=xela/dinov2 \
  ckpt_path="${PRETRAIN_CKPT_DIR}" \
  trainer.max_epochs=1 trainer.validation_frequency=1 trainer.checkpoint_frequency=1 \
  data.dataset_list.0.sequence_list='[corn]' data.dataset_list.0.train_dataset_ids='[0]' data.dataset_list.0.val_dataset_ids='[9]' \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 \
  paths.log_dir="${OUT}/pretrain" paths.tensorboard_dir="${OUT}/pretrain/tensorboard"

CKPT="$(find "${OUT}/pretrain" -type f -name '*.ckpt' | sort | tail -n 1)"
test -n "${CKPT}"

python train_task_force.py +experiment=xela/task/force/dinov2 \
  trainer.max_epochs=1 trainer.validation_frequency=1 train_data_budget=0.01 val_data_budget=0.01 data.max_train_data=512 \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 data.test_dataloader.batch_size=8 \
  checkpoint_encoder="${CKPT}" task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/force" paths.tensorboard_dir="${OUT}/force/tensorboard"

python train_task_pose_estimation.py +experiment=xela/task/relative_pose_estimation/dinov2 \
  trainer.max_epochs=1 trainer.validation_frequency=1 train_data_budget=0.01 val_data_budget=0.01 data.max_train_data=512 \
  data.train_dataloader.batch_size=4 data.val_dataloader.batch_size=4 data.test_dataloader.batch_size=4 \
  checkpoint_encoder="${CKPT}" task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/pose_estimation" paths.tensorboard_dir="${OUT}/pose_estimation/tensorboard"

python train_task_object.py +experiment=xela/task/object_classification/dinov2 \
  trainer.max_epochs=1 trainer.validation_frequency=1 \
  data.dataset_list.0.sequence_list='[corn]' data.dataset_list.0.train_dataset_ids='[0]' data.dataset_list.0.val_dataset_ids='[8]' data.dataset_list.0.test_dataset_ids='[9]' \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 data.test_dataloader.batch_size=8 \
  checkpoint_encoder="${CKPT}" task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/object_classification" paths.tensorboard_dir="${OUT}/object_classification/tensorboard"

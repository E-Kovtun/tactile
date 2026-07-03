#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/scripts/outputs/smoke_xela_mae"
DATA_ROOT="${DATA_ROOT:-${ROOT}/sparsh-skin-dataset}"

cd "${ROOT}"
mkdir -p "${OUT}"

python train.py +experiment=xela/mae \
  paths=default \
  trainer.max_epochs=1 trainer.validation_frequency=1 trainer.checkpoint_frequency=1 \
  data.dataset_list.0.sequence_list='[corn]' data.dataset_list.0.train_dataset_ids='[0]' data.dataset_list.0.val_dataset_ids='[9]' \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/pretrain" paths.tensorboard_dir="${OUT}/pretrain/tensorboard" \
  trainer.save_checkpoint_dir="\${paths.output_dir}/checkpoints"

CKPT="$(find "${OUT}/pretrain" -type f -name '*.ckpt' | sort | tail -n 1)"
test -n "${CKPT}"

python train_task_force.py +experiment=xela/task/force/mae \
  paths=default tensorboard=tensorboard_config \
  trainer.max_epochs=1 trainer.validation_frequency=1 train_data_budget=0.01 val_data_budget=0.01 data.max_train_data=512 \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 data.test_dataloader.batch_size=8 \
  data.train_dataloader.drop_last=false data.val_dataloader.drop_last=false data.test_dataloader.drop_last=false \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/force" paths.tensorboard_dir="${OUT}/force/tensorboard"

python train_task_pose_estimation.py +experiment=xela/task/relative_pose_estimation/mae \
  paths=default tensorboard=tensorboard_config \
  trainer.max_epochs=1 trainer.validation_frequency=1 train_data_budget=0.01 val_data_budget=0.01 data.max_train_data=512 \
  data.train_dataloader.batch_size=4 data.val_dataloader.batch_size=4 data.test_dataloader.batch_size=4 \
  data.train_dataloader.drop_last=false data.val_dataloader.drop_last=false data.test_dataloader.drop_last=false \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/pose_estimation" paths.tensorboard_dir="${OUT}/pose_estimation/tensorboard"

python train_task_object.py +experiment=xela/task/object_classification/mae \
  trainer.max_epochs=1 trainer.validation_frequency=1 \
  data.dataset_list.0.sequence_list='[corn]' data.dataset_list.0.train_dataset_ids='[0]' data.dataset_list.0.val_dataset_ids='[8]' data.dataset_list.0.test_dataset_ids='[9]' \
  data.train_dataloader.batch_size=8 data.val_dataloader.batch_size=8 data.test_dataloader.batch_size=8 \
  data.train_dataloader.drop_last=false data.val_dataloader.drop_last=false data.test_dataloader.drop_last=false \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/object_classification" paths.tensorboard_dir="${OUT}/object_classification/tensorboard"

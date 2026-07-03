#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/scripts/outputs/xela_mae"

cd "${ROOT}"
mkdir -p "${OUT}"

python train.py +experiment=xela/mae \
  paths=default \
  paths.log_dir="${OUT}/pretrain" \
  paths.tensorboard_dir="${OUT}/pretrain/tensorboard" \
  trainer.save_checkpoint_dir="\${paths.output_dir}/checkpoints"

CKPT="$(find "${OUT}/pretrain" -type f -name '*.ckpt' | sort | tail -n 1)"
test -n "${CKPT}"

python train_task_force.py +experiment=xela/task/force/mae \
  paths=default tensorboard=tensorboard_config \
  task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/force" paths.tensorboard_dir="${OUT}/force/tensorboard"

python train_task_pose_estimation.py +experiment=xela/task/relative_pose_estimation/mae \
  paths=default tensorboard=tensorboard_config \
  task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/pose_estimation" paths.tensorboard_dir="${OUT}/pose_estimation/tensorboard"

python train_task_object.py +experiment=xela/task/object_classification/mae \
  task.checkpoint_encoder="${CKPT}" \
  paths.log_dir="${OUT}/object_classification" paths.tensorboard_dir="${OUT}/object_classification/tensorboard"

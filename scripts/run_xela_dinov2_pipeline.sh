#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/scripts/outputs/xela_dinov2"
DATA_ROOT="${DATA_ROOT:-${ROOT}/sparsh-skin-dataset}"
PRETRAIN_CKPT_DIR="${ROOT}/experiments/pretrain_gat_encoder_dinov2/2026.07.03_09-13/checkpoints"

cd "${ROOT}"
mkdir -p "${OUT}"

python train.py +experiment=xela/dinov2 \
  ckpt_path="${PRETRAIN_CKPT_DIR}" \
  paths.data_root="${DATA_ROOT}" \
  paths.log_dir="${OUT}/pretrain" \
  paths.tensorboard_dir="${OUT}/pretrain/tensorboard"

CKPT="$(find "${OUT}/pretrain" -type f -name '*.ckpt' | sort | tail -n 1)"
test -n "${CKPT}"

python train_task_force.py +experiment=xela/task/force/dinov2 \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/force" paths.tensorboard_dir="${OUT}/force/tensorboard"

python train_task_pose_estimation.py +experiment=xela/task/relative_pose_estimation/dinov2 \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/pose_estimation" paths.tensorboard_dir="${OUT}/pose_estimation/tensorboard"

python train_task_object.py +experiment=xela/task/object_classification/dinov2 \
  data.dataset_list.0.dataset.config.features.use_spatial_coords=true \
  task.checkpoint_encoder="${CKPT}" \
  paths.data_root="${DATA_ROOT}" paths.log_dir="${OUT}/object_classification" paths.tensorboard_dir="${OUT}/object_classification/tensorboard"

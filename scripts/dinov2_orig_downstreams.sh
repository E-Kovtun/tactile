#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/scripts/outputs/xela_dinov2"
DATA_ROOT="${DATA_ROOT:-${ROOT}/sparsh-skin-dataset}"
PRETRAIN_CKPT_DIR="${ROOT}/experiments/pretrain_gat_encoder_dinov2/2026.07.03_09-13/checkpoints"

cd "${ROOT}"
mkdir -p "${OUT}"


python train_task_force.py \
  +experiment=xela/task/force/dinov2 \
  experiment_name=downstream_force_dinov2 \
  run_name=Original_DiNO_fixed \
  task.checkpoint_encoder=/workspace-SR004.nfs2/konovalov/tactile/experiments/pretrain_dinov2_encoder_dinov2/2026.07.09_17-26/checkpoints/epoch-0500.ckpt
  
python train_task_object.py \
  +experiment=xela/task/object_classification/dinov2 \
  experiment_name=downstream_object_dinov2 \
  run_name=Original_DiNO_fixed \
  task.checkpoint_encoder=/workspace-SR004.nfs2/konovalov/tactile/experiments/pretrain_dinov2_encoder_dinov2/2026.07.09_17-26/checkpoints/epoch-0500.ckpt
  
python train_task_pose_estimation.py \
  +experiment=xela/task/relative_pose_estimation/dinov2 \
  experiment_name=downstream_pose_dinov2 \
  run_name=Original_DiNO_fixed \
  task.checkpoint_encoder=/workspace-SR004.nfs2/konovalov/tactile/experiments/pretrain_dinov2_encoder_dinov2/2026.07.09_17-26/checkpoints/epoch-0500.ckpt
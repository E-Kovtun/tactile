#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="/workspace-SR004.nfs2/konovalov/tactile"
readonly PYTHON_BIN="/workspace-SR004.nfs2/konovalov/conda_tactile_env/bin/python"
readonly NOPOS_DINO_CHECKPOINT="/workspace-SR004.nfs2/kovtun/tactile/experiments/pretrain_nopos_dinov2/2026.05.25_18-32/checkpoints/epoch-0500.ckpt"
readonly SIGNIFICANCE_CONFIG="dino_attention_comparison"

has_completed_run() {
  local experiment_name="$1"
  local run_pattern="$2"
  local run_dir

  while IFS= read -r -d '' run_dir; do
    if [[ -f "${run_dir}/evaluation/test_predictions.npz" ]]; then
      return 0
    fi
    if find "${run_dir}/checkpoints" -maxdepth 1 -type f \
      \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) \
      -print -quit 2>/dev/null | grep -q .; then
      return 0
    fi
  done < <(
    find "${REPO_ROOT}/experiments/${experiment_name}" \
      -mindepth 1 -maxdepth 1 -type d -name "${run_pattern}" -print0 2>/dev/null
  )
  return 1
}

run_if_missing() {
  local label="$1"
  local experiment_name="$2"
  local run_pattern="$3"
  shift 3

  if has_completed_run "$experiment_name" "$run_pattern"; then
    printf 'SKIP: %s already has an evaluation artifact or checkpoint.\n' "$label"
    return 0
  fi

  printf '\nTRAIN: %s\n>>> ' "$label"
  printf '%q ' "$@"
  printf '\n'
  "$@"

  if ! has_completed_run "$experiment_name" "$run_pattern"; then
    printf 'Training command finished but no completed run was found for %s.\n' "$label" >&2
    return 1
  fi
}

run_original_if_missing() {
  local label="$1"
  local entrypoint="$2"
  local experiment="$3"
  local experiment_name="$4"

  run_if_missing \
    "$label" \
    "$experiment_name" \
    'Original_*' \
    "$PYTHON_BIN" "$entrypoint" "+experiment=${experiment}"
}

run_attention_if_missing() {
  local label="$1"
  local entrypoint="$2"
  local experiment="$3"
  local experiment_name="$4"
  local run_pattern="$5"

  run_if_missing \
    "$label" \
    "$experiment_name" \
    "$run_pattern" \
    "$PYTHON_BIN" "$entrypoint" "+experiment=${experiment}" \
    "experiment_name=${experiment_name}" \
    "data.dataset.config.features.use_spatial_coords=true" \
    "task.checkpoint_encoder=${NOPOS_DINO_CHECKPOINT}" \
    "task.model_encoder.in_chans=3"
}

cd "$REPO_ROOT"

for required_file in "$PYTHON_BIN" "$NOPOS_DINO_CHECKPOINT"; do
  if [[ ! -r "$required_file" ]]; then
    printf 'Required file is not readable: %s\n' "$required_file" >&2
    exit 2
  fi
done

# Force: all three attention variants are already expected to exist; these
# checks make the script safely resumable if a run or artifact is removed.
run_original_if_missing \
  'force / Original DINO' \
  train_task_force.py \
  xela/task/force/dinov2 \
  downstream_force_dinov2
run_attention_if_missing \
  'force / attention baseline' \
  train_task_force.py \
  xela/task/force/dinov2_attention_baseline \
  downstream_force_dinov2_nopos \
  'attention_baseline_l1_h12_*'
run_attention_if_missing \
  'force / distance attention' \
  train_task_force.py \
  xela/task/force/dinov2_spatial_distance_attention \
  downstream_force_dinov2_nopos \
  'spatial_distance_attention_l1_h12_*'
run_attention_if_missing \
  'force / directional attention' \
  train_task_force.py \
  xela/task/force/dinov2_spatial_directional_attention \
  downstream_force_dinov2_nopos \
  'spatial_directional_attention_l1_h12_*'

# Pose: missing variants are trained with the same signal-only encoder and
# downstream XYZ access as the force comparison.
run_original_if_missing \
  'pose / Original DINO' \
  train_task_pose_estimation.py \
  xela/task/relative_pose_estimation/dinov2 \
  downstream_pose_dinov2
run_attention_if_missing \
  'pose / attention baseline' \
  train_task_pose_estimation.py \
  xela/task/relative_pose_estimation/dinov2_attention_baseline \
  downstream_pose_dinov2_nopos \
  'attention_baseline_l1_h6_*'
run_attention_if_missing \
  'pose / distance attention' \
  train_task_pose_estimation.py \
  xela/task/relative_pose_estimation/dinov2_spatial_distance_attention \
  downstream_pose_dinov2_nopos \
  'spatial_distance_attention_l1_h6_*'
run_attention_if_missing \
  'pose / directional attention' \
  train_task_pose_estimation.py \
  xela/task/relative_pose_estimation/dinov2_spatial_directional_attention \
  downstream_pose_dinov2_nopos \
  'spatial_directional_attention_l1_h6_*'

printf '\nSTATISTICS: attention baseline vs Original DINO, distance and directional attention\n'
"$PYTHON_BIN" \
  scripts/analyze_downstream_significance.py \
  --config-name "$SIGNIFICANCE_CONFIG"

printf '\nAttention downstreams and statistics completed successfully.\n'

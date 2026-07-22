#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="/workspace-SR004.nfs2/konovalov/tactile"
readonly PYTHON_BIN="/workspace-SR004.nfs2/konovalov/conda_tactile_env/bin/python"
readonly PRETRAIN_ROOT="${REPO_ROOT}/experiments/pretrain_transformer_encoder_jepa_graph"
readonly COMMON_SIGNIFICANCE_CONFIG="purple_only_signal_with_jepa"

shopt -s nullglob

require_file() {
  if [[ ! -r "$1" ]]; then
    printf 'Required file is not readable: %s\n' "$1" >&2
    exit 2
  fi
}

find_pretrain_checkpoint() {
  local run_prefix="$1"
  local -a candidates=("${PRETRAIN_ROOT}/${run_prefix}_"*/checkpoints/epoch-0500.ckpt)
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${candidates[${#candidates[@]} - 1]}"
}

run_pretrain_if_missing() {
  local run_prefix="$1"
  local num_context_masks="$2"
  local num_target_masks="$3"
  local context_scale="$4"
  local target_scale="$5"
  local checkpoint

  if checkpoint="$(find_pretrain_checkpoint "$run_prefix")"; then
    printf 'SKIP pretrain: %s\n' "$checkpoint" >&2
    printf '%s\n' "$checkpoint"
    return 0
  fi

  printf '\nPRETRAIN: %s\n' "$run_prefix" >&2
  "$PYTHON_BIN" train.py \
    +experiment=xela/jepa_graph \
    "run_name=${run_prefix}" \
    "algorithm.num_context_masks=${num_context_masks}" \
    "algorithm.num_target_masks=${num_target_masks}" \
    "algorithm.context_mask_scale=${context_scale}" \
    "algorithm.target_mask_scale=${target_scale}" \
    "data.graph_mask_collator.num_context_masks=${num_context_masks}" \
    "data.graph_mask_collator.num_target_masks=${num_target_masks}" >&2

  if ! checkpoint="$(find_pretrain_checkpoint "$run_prefix")"; then
    printf 'Pretraining finished without epoch-0500.ckpt for %s\n' "$run_prefix" >&2
    return 1
  fi
  printf '%s\n' "$checkpoint"
}

has_completed_downstream() {
  local experiment_name="$1"
  local run_prefix="$2"
  local run_dir
  local -a run_dirs=("${REPO_ROOT}/experiments/${experiment_name}/${run_prefix}_"*)

  for run_dir in "${run_dirs[@]}"; do
    [[ -d "$run_dir" ]] || continue
    if [[ -f "${run_dir}/evaluation/test_predictions.npz" ]]; then
      return 0
    fi
    local -a checkpoints=("${run_dir}/checkpoints/"*.ckpt "${run_dir}/checkpoints/"*.pt "${run_dir}/checkpoints/"*.pth)
    if (( ${#checkpoints[@]} > 0 )); then
      return 0
    fi
  done
  return 1
}

run_downstream_if_missing() {
  local label="$1"
  local entrypoint="$2"
  local experiment="$3"
  local experiment_name="$4"
  local run_prefix="$5"
  local checkpoint_encoder="$6"

  if has_completed_downstream "$experiment_name" "$run_prefix"; then
    printf 'SKIP downstream: %s\n' "$label"
    return 0
  fi

  printf '\nDOWNSTREAM: %s\n' "$label"
  "$PYTHON_BIN" "$entrypoint" \
    "+experiment=${experiment}" \
    "run_name=${run_prefix}" \
    "task.checkpoint_encoder=${checkpoint_encoder}"

  if ! has_completed_downstream "$experiment_name" "$run_prefix"; then
    printf 'Downstream finished without a checkpoint or evaluation artifact: %s\n' "$label" >&2
    return 1
  fi
}

run_all_downstreams() {
  local run_prefix="$1"
  local checkpoint_encoder="$2"

  run_downstream_if_missing \
    "force / ${run_prefix}" \
    train_task_force.py \
    xela/task/force/jepa \
    downstream_force_jepa \
    "$run_prefix" \
    "$checkpoint_encoder"
  run_downstream_if_missing \
    "pose / ${run_prefix}" \
    train_task_pose_estimation.py \
    xela/task/relative_pose_estimation/jepa \
    downstream_pose_estimation_comparison_jepa \
    "$run_prefix" \
    "$checkpoint_encoder"
  run_downstream_if_missing \
    "object classification / ${run_prefix}" \
    train_task_object.py \
    xela/task/object_classification/jepa \
    downstream_object_classification_comparison_jepa \
    "$run_prefix" \
    "$checkpoint_encoder"
}

run_statistics() {
  local config_name="$1"
  printf '\nSTATISTICS: %s\n' "$config_name"
  "$PYTHON_BIN" scripts/analyze_downstream_significance.py --config-name "$config_name"
}

run_variant() {
  local pretrain_run_prefix="$1"
  local downstream_run_prefix="$2"
  local num_context_masks="$3"
  local num_target_masks="$4"
  local context_scale="$5"
  local target_scale="$6"
  local significance_config="$7"
  local checkpoint

  checkpoint="$(run_pretrain_if_missing \
    "$pretrain_run_prefix" \
    "$num_context_masks" \
    "$num_target_masks" \
    "$context_scale" \
    "$target_scale")"
  run_all_downstreams "$downstream_run_prefix" "$checkpoint"
  run_statistics "$significance_config"
}

cd "$REPO_ROOT"
require_file "$PYTHON_BIN"
require_file "${REPO_ROOT}/scripts/analyze_downstream_significance.py"

# 1) Larger-mask (1 context, 2 targets), then its three downstreams and report.
run_variant \
  physical_dijkstra_1c2t_c50-80_t10-22 \
  graph_physical_dijkstra_1c2t_c50-80_t10-22 \
  1 \
  2 \
  '[0.50,0.80]' \
  '[0.10,0.22]' \
  graph_jepa_1c2t_c50_80_t10_22

# 2) Larger-mask (1 context, 4 targets), then its three downstreams and report.
run_variant \
  physical_dijkstra_1c4t_c50-90_t10-18 \
  graph_physical_dijkstra_1c4t_c50-90_t10-18 \
  1 \
  4 \
  '[0.50,0.90]' \
  '[0.10,0.18]' \
  graph_jepa_1c4t_c50_90_t10_18

# 3) Rebuild the complete report after both new variants are available.
run_statistics "$COMMON_SIGNIFICANCE_CONFIG"

printf '\nBoth graph-JEPA variants, six downstreams and all reports completed successfully.\n'

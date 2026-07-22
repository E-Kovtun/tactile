#!/usr/bin/env bash

set -uo pipefail

readonly REPO_ROOT="/workspace-SR004.nfs2/konovalov/tactile"
readonly PYTHON_BIN="/workspace-SR004.nfs2/konovalov/conda_tactile_env/bin/python"
readonly SIGNIFICANCE_CONFIG="graph_jepa_1c4t_vs_2c4t"

readonly CHECKPOINT_2C4T="${REPO_ROOT}/experiments/pretrain_transformer_encoder_jepa_graph/physical_dijkstra_2026.07.21_18-19/checkpoints/epoch-0500.ckpt"
readonly CHECKPOINT_1C4T="${REPO_ROOT}/experiments/pretrain_transformer_encoder_jepa_graph/physical_dijkstra_1_4_2026.07.22_01-15/checkpoints/epoch-0500.ckpt"

readonly -a JEPA_VARIANTS=(
  graph_physical_dijkstra_2c_4t
  graph_physical_dijkstra_1c_4t
)

FAILED_COMMANDS=()

run_command() {
  printf '\n>>> '
  printf '%q ' "$@"
  printf '\n'

  local exit_code
  if "$@"; then
    return 0
  else
    exit_code=$?
    FAILED_COMMANDS+=("exit ${exit_code}: $*")
    printf '!!! Command failed with exit code %s; continuing.\n' "$exit_code" >&2
    return 0
  fi
}

require_file() {
  if [[ ! -r "$1" ]]; then
    printf 'Required file is not readable: %s\n' "$1" >&2
    exit 2
  fi
}

run_downstream() {
  local entrypoint="$1"
  local experiment="$2"
  local variant

  for variant in "${JEPA_VARIANTS[@]}"; do
    run_command \
      "$PYTHON_BIN" \
      "$entrypoint" \
      "+experiment=${experiment}" \
      "jepa_pretrain=${variant}"
  done
}

cd "$REPO_ROOT" || exit 2

require_file "$PYTHON_BIN"
require_file "$CHECKPOINT_2C4T"
require_file "$CHECKPOINT_1C4T"

# Run all six trainings sequentially so each four-GPU downstream has the node
# to itself. Every training entrypoint writes its test predictions artifact.
run_downstream train_task_force.py xela/task/force/jepa
run_downstream train_task_pose_estimation.py xela/task/relative_pose_estimation/jepa
run_downstream train_task_object.py xela/task/object_classification/jepa

# Build paired cluster-bootstrap/permutation statistics from the six test sets.
# Run it even if one training failed; this still produces a report when all
# required completed runs already exist from an earlier invocation.
run_command \
  "$PYTHON_BIN" \
  scripts/analyze_downstream_significance.py \
  --config-name "$SIGNIFICANCE_CONFIG"

if (( ${#FAILED_COMMANDS[@]} > 0 )); then
  printf '\nCompleted with %d failed command(s):\n' "${#FAILED_COMMANDS[@]}" >&2
  printf '  - %s\n' "${FAILED_COMMANDS[@]}" >&2
  exit 1
fi

printf '\nAll six downstream runs and the statistical report completed successfully.\n'

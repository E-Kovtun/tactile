#!/usr/bin/env bash

set -uo pipefail

readonly REPO_ROOT="${REPO_ROOT:-/workspace-SR004.nfs2/konovalov/tactile}"
readonly PYTHON_BIN="${PYTHON_BIN:-/workspace-SR004.nfs2/konovalov/conda_tactile_env/bin/python}"
readonly PRETRAIN_ROOT="${REPO_ROOT}/experiments/pretrain_transformer_encoder_jepa_graph"
readonly BASE_SIGNIFICANCE_CONFIG="${REPO_ROOT}/config/significance/purple_only_signal_with_jepa.yaml"
readonly CUMULATIVE_REGISTRY="${REPO_ROOT}/config/significance/graph_jepa_cumulative_registry.yaml"
readonly CUMULATIVE_CONFIG="${REPO_ROOT}/config/significance/graph_jepa_cumulative.yaml"
readonly CUMULATIVE_CONFIG_NAME="graph_jepa_cumulative"
readonly FINAL_CHECKPOINT_NAME="${FINAL_CHECKPOINT_NAME:-epoch-0500.ckpt}"
readonly GPU_IDS="${GPU_IDS:-0,1,2,3}"
readonly WAIT_SECONDS="${WAIT_SECONDS:-60}"
readonly GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
readonly GPU_FREE_CONFIRM_SECONDS="${GPU_FREE_CONFIRM_SECONDS:-30}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
readonly DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-42}"
readonly RUN_SUFFIX="${RUN_SUFFIX:-}"
readonly GPU_LOCK_NAMESPACE="${GPU_LOCK_NAMESPACE:-$(hostname -s)}"

usage() {
  printf 'Usage: %s <experiment-config>\n' "$(basename "$0")" >&2
  printf 'Example: %s xela/jepa_graph_1c4t_4stratifiedrandom_c50_90_t10_18\n' \
    "$(basename "$0")" >&2
}

if (( $# != 1 )); then
  usage
  exit 2
fi

experiment_config="${1#experiment/}"
experiment_config="${experiment_config%.yaml}"
readonly EXPERIMENT_CONFIG="$experiment_config"
readonly EXPERIMENT_FILE="${REPO_ROOT}/config/experiment/${EXPERIMENT_CONFIG}.yaml"

if [[ ! -f "$EXPERIMENT_FILE" ]]; then
  printf 'Experiment config does not exist: %s\n' "$EXPERIMENT_FILE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable does not exist: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

if ! run_prefix="$(
  "$PYTHON_BIN" -c \
    'import sys, yaml; value=yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("run_name"); assert value, "Experiment config must define run_name"; print(value)' \
    "$EXPERIMENT_FILE"
)"; then
  printf 'Could not read run_name from %s\n' "$EXPERIMENT_FILE" >&2
  exit 1
fi
readonly RUN_PREFIX="${run_prefix}${RUN_SUFFIX}"
readonly DOWNSTREAM_PREFIX="graph_${RUN_PREFIX}"
candidate_id="$(
  printf 'graph_%s' "$RUN_PREFIX" |
    tr '[:upper:]' '[:lower:]' |
    tr -c '[:alnum:]' '_'
)"
candidate_id="${candidate_id%_}"
readonly CANDIDATE_ID="$candidate_id"
readonly CANDIDATE_NAME="Graph JEPA — ${RUN_PREFIX}"

config_value() {
  local dotted_key="$1"
  local fallback="$2"
  "$PYTHON_BIN" -c \
    'import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    if not isinstance(value, dict) or key not in value:
        print(sys.argv[3])
        raise SystemExit
    value = value[key]
if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)' \
    "$EXPERIMENT_FILE" "$dotted_key" "$fallback"
}

readonly ENCODER_IN_CHANS="$(config_value algorithm.encoder.in_chans 3)"
readonly INPUT_FUSION="$(config_value algorithm.encoder.input_fusion joint)"
readonly SIGNAL_CHANS="$(config_value algorithm.encoder.signal_chans 3)"
readonly COORDINATE_CHANS="$(config_value algorithm.encoder.coordinate_chans 3)"
readonly RANDOM_EMBEDDING_STD="$(config_value algorithm.encoder.random_embedding_std 1.0)"
readonly USE_SPATIAL_COORDS="$(config_value data.features.use_spatial_coords false)"

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
readonly NUM_DEVICES="${NUM_DEVICES:-${#gpu_array[@]}}"
if (( NUM_DEVICES <= 0 )); then
  printf 'No GPU devices configured.\n' >&2
  exit 1
fi
previous_gpu=-1
for gpu_id in "${gpu_array[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    printf 'GPU_IDS must contain comma-separated numeric indices: %s\n' \
      "$GPU_IDS" >&2
    exit 1
  fi
  if (( gpu_id <= previous_gpu )); then
    printf 'GPU_IDS must be unique and sorted increasingly: %s\n' \
      "$GPU_IDS" >&2
    exit 1
  fi
  previous_gpu="$gpu_id"
done

per_device_batch() {
  local global_batch="$1"
  if (( global_batch % NUM_DEVICES != 0 )); then
    printf 'Global batch %s is not divisible by %s devices.\n' \
      "$global_batch" "$NUM_DEVICES" >&2
    return 1
  fi
  printf '%s\n' "$(( global_batch / NUM_DEVICES ))"
}

readonly PRETRAIN_BATCH_PER_GPU="${PRETRAIN_BATCH_PER_GPU:-$(per_device_batch 256)}"
readonly FORCE_TRAIN_BATCH="${FORCE_TRAIN_BATCH:-$(per_device_batch 64)}"
readonly FORCE_EVAL_BATCH="${FORCE_EVAL_BATCH:-$(per_device_batch 256)}"
readonly POSE_TRAIN_BATCH="${POSE_TRAIN_BATCH:-$(per_device_batch 16)}"
readonly POSE_EVAL_BATCH="${POSE_EVAL_BATCH:-$(per_device_batch 64)}"
readonly OBJECT_TRAIN_BATCH="${OBJECT_TRAIN_BATCH:-$(per_device_batch 64)}"
readonly OBJECT_EVAL_BATCH="${OBJECT_EVAL_BATCH:-$(per_device_batch 256)}"

shopt -s nullglob

gpu_lock_fds=()

acquire_gpu_locks() {
  local gpu_id
  local lock_file
  local lock_fd
  for gpu_id in "${gpu_array[@]}"; do
    lock_file="${REPO_ROOT}/experiments/.graph-jepa-gpu-${GPU_LOCK_NAMESPACE}-${gpu_id}.lock"
    exec {lock_fd}>"$lock_file"
    if ! flock -n "$lock_fd"; then
      printf 'GPU %s is reserved by another Graph-JEPA pipeline; waiting for its lock.\n' \
        "$gpu_id"
      flock "$lock_fd"
    fi
    gpu_lock_fds+=("$lock_fd")
  done
}

selected_gpu_processes() {
  local gpu_id
  local process_rows
  for gpu_id in "${gpu_array[@]}"; do
    process_rows="$(
      nvidia-smi \
        --id="$gpu_id" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits
    )"
    if [[ -n "$process_rows" ]]; then
      while IFS= read -r process_row; do
        [[ -n "$process_row" ]] || continue
        printf 'GPU %s: %s\n' "$gpu_id" "$process_row"
      done <<< "$process_rows"
    fi
  done
}

wait_for_selected_gpus() {
  local busy_processes
  local announced=false
  while true; do
    busy_processes="$(selected_gpu_processes)"
    if [[ -n "$busy_processes" ]]; then
      if [[ "$announced" == false ]]; then
        printf 'Selected GPUs are occupied; waiting for CUDA processes to exit:\n%s\n' \
          "$busy_processes"
        announced=true
      fi
      sleep "$GPU_POLL_SECONDS"
      continue
    fi

    # A second observation avoids entering a short gap between sequential jobs.
    sleep "$GPU_FREE_CONFIRM_SECONDS"
    busy_processes="$(selected_gpu_processes)"
    if [[ -z "$busy_processes" ]]; then
      if [[ "$announced" == true ]]; then
        printf 'Selected GPUs are free and remained free for %s seconds.\n' \
          "$GPU_FREE_CONFIRM_SECONDS"
      fi
      return
    fi
    if [[ "$announced" == false ]]; then
      printf 'GPU activity appeared during the free confirmation window; waiting:\n%s\n' \
        "$busy_processes"
      announced=true
    fi
    sleep "$GPU_POLL_SECONDS"
  done
}

saved_run_name() {
  local config_file="$1"
  "$PYTHON_BIN" -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("run_name", ""))' \
    "$config_file"
}

matching_pretrain_dirs() {
  local run_dir
  local config_file
  local -a candidates=("${PRETRAIN_ROOT}/${RUN_PREFIX}_"*)
  for run_dir in "${candidates[@]}"; do
    config_file="${run_dir}/.hydra/config.yaml"
    [[ -f "$config_file" ]] || continue
    [[ "$(saved_run_name "$config_file")" == "$RUN_PREFIX" ]] || continue
    printf '%s\n' "$run_dir"
  done
}

find_completed_pretrain() {
  local run_dir
  local checkpoint
  local index
  local -a candidates=()
  while IFS= read -r run_dir; do
    candidates+=("$run_dir")
  done < <(matching_pretrain_dirs)
  for (( index=${#candidates[@]} - 1; index >= 0; index-- )); do
    checkpoint="${candidates[index]}/checkpoints/${FINAL_CHECKPOINT_NAME}"
    if [[ -f "$checkpoint" ]]; then
      printf '%s\n' "$checkpoint"
      return
    fi
  done
  return 1
}

find_resume_dir() {
  local run_dir
  local index
  local -a candidates=()
  while IFS= read -r run_dir; do
    candidates+=("$run_dir")
  done < <(matching_pretrain_dirs)
  for (( index=${#candidates[@]} - 1; index >= 0; index-- )); do
    run_dir="${candidates[index]}"
    if [[ -f "${run_dir}/checkpoints/last.ckpt" ]]; then
      printf '%s\n' "$run_dir"
      return
    fi
  done
  return 1
}

pretrain_is_active() {
  pgrep -f "[t]rain.py.*[+]experiment=${EXPERIMENT_CONFIG}([[:space:]]|$)" >/dev/null
}

wait_for_active_pretrain() {
  if ! pretrain_is_active; then
    return
  fi
  printf 'Existing pretrain is active; waiting for it to finish: %s\n' \
    "$EXPERIMENT_CONFIG"
  while pretrain_is_active; do
    sleep "$WAIT_SECONDS"
  done
}

run_pretrain() {
  local checkpoint
  local resume_dir

  if checkpoint="$(find_completed_pretrain)"; then
    printf 'SKIP completed pretrain: %s\n' "$checkpoint"
    return
  fi

  wait_for_active_pretrain
  if checkpoint="$(find_completed_pretrain)"; then
    printf 'FOUND completed pretrain after waiting: %s\n' "$checkpoint"
    return
  fi

  wait_for_selected_gpus
  if resume_dir="$(find_resume_dir)"; then
    printf 'RESUME pretrain in the same run directory: %s\n' "$resume_dir"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" train.py \
      "+experiment=${EXPERIMENT_CONFIG}" \
      "run_name=${RUN_PREFIX}" \
      "seed=${PRETRAIN_SEED}" \
      "+trainer.devices=${NUM_DEVICES}" \
      "data.train_dataloader.batch_size=${PRETRAIN_BATCH_PER_GPU}" \
      "data.val_dataloader.batch_size=${PRETRAIN_BATCH_PER_GPU}" \
      "ckpt_path=${resume_dir}/checkpoints" \
      "hydra.run.dir=${resume_dir}"
  else
    printf 'START new pretrain: %s\n' "$RUN_PREFIX"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" train.py \
      "+experiment=${EXPERIMENT_CONFIG}" \
      "run_name=${RUN_PREFIX}" \
      "seed=${PRETRAIN_SEED}" \
      "+trainer.devices=${NUM_DEVICES}" \
      "data.train_dataloader.batch_size=${PRETRAIN_BATCH_PER_GPU}" \
      "data.val_dataloader.batch_size=${PRETRAIN_BATCH_PER_GPU}"
  fi

  if ! checkpoint="$(find_completed_pretrain)"; then
    printf 'Pretrain ended without %s: %s\n' \
      "$FINAL_CHECKPOINT_NAME" "$RUN_PREFIX" >&2
    return 1
  fi
}

find_downstream_dir() {
  local experiment_name="$1"
  local run_dir
  local config_file
  local index
  local -a candidates=(
    "${REPO_ROOT}/experiments/${experiment_name}/${DOWNSTREAM_PREFIX}_"*
  )
  for (( index=${#candidates[@]} - 1; index >= 0; index-- )); do
    run_dir="${candidates[index]}"
    [[ -f "${run_dir}/evaluation/test_predictions.npz" ]] || continue
    config_file="${run_dir}/.hydra/config.yaml"
    [[ -f "$config_file" ]] || continue
    [[ "$(saved_run_name "$config_file")" == "$DOWNSTREAM_PREFIX" ]] || continue
    printf '%s\n' "$run_dir"
    return
  done
  return 1
}

run_downstream() {
  local label="$1"
  local entrypoint="$2"
  local experiment="$3"
  local experiment_name="$4"
  local train_batch="$5"
  local eval_batch="$6"
  local encoder_checkpoint="$7"
  local run_dir
  local -a feature_args=()

  if run_dir="$(find_downstream_dir "$experiment_name")"; then
    printf 'SKIP completed downstream %s: %s\n' "$label" "$run_dir"
    return
  fi

  if [[ "$USE_SPATIAL_COORDS" == "true" ]]; then
    if [[ "$label" == "object" ]]; then
      feature_args=(
        "++data.dataset_list.0.dataset.config.features.use_spatial_coords=true"
      )
    else
      feature_args=(
        "++data.dataset.config.features.use_spatial_coords=true"
      )
    fi
  fi

  wait_for_selected_gpus
  printf 'RUN downstream %s on GPUs %s\n' "$label" "$GPU_IDS"
  if ! CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$entrypoint" \
    "+experiment=${experiment}" \
    "run_name=${DOWNSTREAM_PREFIX}" \
    "seed=${DOWNSTREAM_SEED}" \
    "trainer.devices=${NUM_DEVICES}" \
    "data.train_dataloader.batch_size=${train_batch}" \
    "data.val_dataloader.batch_size=${eval_batch}" \
    "data.test_dataloader.batch_size=${eval_batch}" \
    "${feature_args[@]}" \
    "task.model_encoder.in_chans=${ENCODER_IN_CHANS}" \
    "++task.model_encoder.input_fusion=${INPUT_FUSION}" \
    "++task.model_encoder.signal_chans=${SIGNAL_CHANS}" \
    "++task.model_encoder.coordinate_chans=${COORDINATE_CHANS}" \
    "++task.model_encoder.random_embedding_std=${RANDOM_EMBEDDING_STD}" \
    "task.checkpoint_encoder=${encoder_checkpoint}"; then
    printf 'Downstream command failed: %s\n' "$label" >&2
    return 1
  fi

  find_downstream_dir "$experiment_name" >/dev/null || {
    printf 'Downstream ended without evaluation artifact: %s\n' "$label" >&2
    return 1
  }
}

print_plan() {
  local checkpoint=""
  local resume_dir=""
  local busy_processes=""
  checkpoint="$(find_completed_pretrain || true)"
  resume_dir="$(find_resume_dir || true)"
  printf 'Experiment config: %s\n' "$EXPERIMENT_CONFIG"
  printf 'Run prefix:       %s\n' "$RUN_PREFIX"
  printf 'Downstream prefix:%s\n' "$DOWNSTREAM_PREFIX"
  printf 'Candidate id:     %s\n' "$CANDIDATE_ID"
  printf 'GPUs/devices:     %s / %s\n' "$GPU_IDS" "$NUM_DEVICES"
  printf 'GPU lock scope:   %s\n' "$GPU_LOCK_NAMESPACE"
  printf 'Seeds:            pretrain=%s, downstream=%s\n' \
    "$PRETRAIN_SEED" "$DOWNSTREAM_SEED"
  printf 'Pretrain batch:   %s per GPU (%s global)\n' \
    "$PRETRAIN_BATCH_PER_GPU" "$(( PRETRAIN_BATCH_PER_GPU * NUM_DEVICES ))"
  printf 'Encoder input:    %sch, fusion=%s, spatial_coords=%s\n' \
    "$ENCODER_IN_CHANS" "$INPUT_FUSION" "$USE_SPATIAL_COORDS"
  printf 'Active pretrain:  %s\n' "$(pretrain_is_active && printf yes || printf no)"
  busy_processes="$(selected_gpu_processes)"
  if [[ -n "$busy_processes" ]]; then
    printf 'GPU processes:\n%s\n' "$busy_processes"
  else
    printf 'GPU processes:    none\n'
  fi
  printf 'Completed ckpt:   %s\n' "${checkpoint:-none}"
  printf 'Resume directory: %s\n' "${resume_dir:-none}"
}

cd "$REPO_ROOT"

# Compose the Hydra config before doing any work.
if ! "$PYTHON_BIN" train.py "+experiment=${EXPERIMENT_CONFIG}" \
  "run_name=${RUN_PREFIX}" "seed=${PRETRAIN_SEED}" --cfg job >/dev/null; then
  printf 'Hydra composition failed: %s\n' "$EXPERIMENT_CONFIG" >&2
  exit 1
fi
print_plan

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if ! acquire_gpu_locks; then
  printf 'Could not acquire GPU locks for %s\n' "$EXPERIMENT_CONFIG" >&2
  exit 1
fi
if ! run_pretrain; then
  printf 'Skipping downstreams because pretrain is incomplete: %s\n' \
    "$EXPERIMENT_CONFIG" >&2
  exit 1
fi
if ! encoder_checkpoint="$(find_completed_pretrain)"; then
  printf 'Completed pretrain checkpoint disappeared: %s\n' "$RUN_PREFIX" >&2
  exit 1
fi

failed_downstreams=()

if ! run_downstream \
  force \
  train_task_force.py \
  xela/task/force/jepa \
  downstream_force_jepa \
  "$FORCE_TRAIN_BATCH" \
  "$FORCE_EVAL_BATCH" \
  "$encoder_checkpoint"; then
  failed_downstreams+=(force)
fi

if ! run_downstream \
  pose \
  train_task_pose_estimation.py \
  xela/task/relative_pose_estimation/jepa \
  downstream_pose_estimation_comparison_jepa \
  "$POSE_TRAIN_BATCH" \
  "$POSE_EVAL_BATCH" \
  "$encoder_checkpoint"; then
  failed_downstreams+=(pose)
fi

if ! run_downstream \
  object \
  train_task_object.py \
  xela/task/object_classification/jepa \
  downstream_object_classification_comparison_jepa \
  "$OBJECT_TRAIN_BATCH" \
  "$OBJECT_EVAL_BATCH" \
  "$encoder_checkpoint"; then
  failed_downstreams+=(object)
fi

if (( ${#failed_downstreams[@]} > 0 )); then
  printf 'Experiment finished with failed downstreams: %s (%s)\n' \
    "$RUN_PREFIX" "${failed_downstreams[*]}" >&2
  printf 'Candidate was not added to the cumulative table; rerunning the same config will skip completed stages.\n' >&2
  exit 1
fi

if ! force_dir="$(find_downstream_dir downstream_force_jepa)" ||
   ! pose_dir="$(find_downstream_dir downstream_pose_estimation_comparison_jepa)" ||
   ! object_dir="$(find_downstream_dir downstream_object_classification_comparison_jepa)"; then
  printf 'Could not resolve all completed downstream directories for %s\n' \
    "$RUN_PREFIX" >&2
  exit 1
fi

# Both GPU servers share the same NFS registry and generated cumulative config.
# Serialize only this short reporting stage; training locks remain host-local.
exec {cumulative_lock_fd}>"${REPO_ROOT}/experiments/.graph-jepa-cumulative.lock"
flock "$cumulative_lock_fd"

if ! "$PYTHON_BIN" scripts/register_graph_jepa_cumulative.py \
  --base-config "$BASE_SIGNIFICANCE_CONFIG" \
  --registry "$CUMULATIVE_REGISTRY" \
  --output-config "$CUMULATIVE_CONFIG" \
  --candidate-id "$CANDIDATE_ID" \
  --candidate-name "$CANDIDATE_NAME" \
  --force-dir "$force_dir" \
  --pose-dir "$pose_dir" \
  --object-dir "$object_dir"; then
  printf 'Could not register cumulative result for %s\n' "$RUN_PREFIX" >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/analyze_downstream_significance.py \
  --config-name "$CUMULATIVE_CONFIG_NAME"; then
  printf 'Cumulative-table analysis failed for %s\n' "$RUN_PREFIX" >&2
  exit 1
fi

printf 'Graph-JEPA pipeline completed: %s\n' "$RUN_PREFIX"
printf 'Cumulative table: %s/significance/graph_jepa_cumulative\n' \
  "${REPO_ROOT}/experiments"

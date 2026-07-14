#!/usr/bin/env bash

# Controlled downstream-force comparison for DINOv2 checkpoints immediately
# before and after the 2026-07-13 resume. The script is intentionally
# non-destructive: it only creates a new timestamped diagnostics directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/workspace-SR004.nfs2/konovalov/tactile/experiments/pretrain_dinov2_encoder_dinov2/2026.07.08_19-25/checkpoints}"
RESUMED_CKPT_DIR="${RESUMED_CKPT_DIR:-/workspace-SR004.nfs2/konovalov/tactile/experiments/pretrain_dinov2_encoder_dinov2/2026.07.13_17-44/checkpoints}"

SERVER_PYTHON="/workspace-SR004.nfs2/konovalov/conda_tactile_env/bin/python"
if [[ -x "${SERVER_PYTHON}" ]]; then
    DEFAULT_PYTHON="${SERVER_PYTHON}"
else
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

TIMESTAMP="$(date +%Y.%m.%d_%H-%M-%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/diagnostics/outputs/dinov2_resume_quality_${TIMESTAMP}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dinov2_resume_quality_diagnostic}"

LABELS=(
    pre_resume_epoch_0350
    pre_resume_last_0358
    post_resume_epoch_0360
    post_resume_epoch_0400
    post_resume_epoch_0450
    post_resume_epoch_0500
)

CHECKPOINTS=(
    "${SOURCE_CKPT_DIR}/epoch-0350.ckpt"
    "${SOURCE_CKPT_DIR}/last.ckpt"
    "${RESUMED_CKPT_DIR}/epoch-0360.ckpt"
    "${RESUMED_CKPT_DIR}/epoch-0400.ckpt"
    "${RESUMED_CKPT_DIR}/epoch-0450.ckpt"
    "${RESUMED_CKPT_DIR}/epoch-0500.ckpt"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "${ROOT}/train_task_force.py" ]] || die "train_task_force.py not found under ${ROOT}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c \
    'from tensorboard.backend.event_processing.event_accumulator import EventAccumulator' \
    >/dev/null 2>&1 || die "TensorBoard is unavailable in ${PYTHON_BIN}; metrics cannot be collected"

# Never mix a new diagnostic with an existing run and never remove old output.
[[ ! -e "${OUTPUT_ROOT}" ]] || die "Output already exists; choose a new OUTPUT_ROOT: ${OUTPUT_ROOT}"

missing=0
for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        missing=1
    fi
done
[[ "${missing}" -eq 0 ]] || die "Preflight failed; no training was started"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/tensorboard"

SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
MANIFEST_TSV="${OUTPUT_ROOT}/checkpoint_manifest.tsv"
printf '%s\n' 'label,checkpoint,status,exit_code,test_rmse,test_rmse_x,test_rmse_y,test_rmse_z,best_val_rmse,last_val_rmse,last_train_rmse,event_file' > "${SUMMARY_CSV}"
printf 'label\tcheckpoint\tbytes\tmodified_at\n' > "${MANIFEST_TSV}"

{
    echo "created_at=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname=$(hostname)"
    echo "repo_root=${ROOT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "python_bin=${PYTHON_BIN}"
    echo "python_version=$(${PYTHON_BIN} --version 2>&1)"
    echo "source_checkpoint_dir=${SOURCE_CKPT_DIR}"
    echo "resumed_checkpoint_dir=${RESUMED_CKPT_DIR}"
    echo "normalization_override=none"
    echo "seed=42"
    echo "trainer_devices=4"
    echo "train_batch_size_per_device=16"
    echo "train_data_budget=1.0"
    echo "val_data_budget=1.0"
    echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status_begin"
    git -C "${ROOT}" status --short 2>/dev/null || true
    echo "git_status_end"
} > "${OUTPUT_ROOT}/metadata.txt"

for index in "${!LABELS[@]}"; do
    label="${LABELS[${index}]}"
    checkpoint="${CHECKPOINTS[${index}]}"
    checkpoint_bytes="$(stat -c '%s' "${checkpoint}" 2>/dev/null || stat -f '%z' "${checkpoint}")"
    checkpoint_mtime="$(stat -c '%y' "${checkpoint}" 2>/dev/null || stat -f '%Sm' "${checkpoint}")"
    printf '%s\t%s\t%s\t%s\n' "${label}" "${checkpoint}" "${checkpoint_bytes}" "${checkpoint_mtime}" >> "${MANIFEST_TSV}"
done

cd "${ROOT}" || die "Cannot enter repo root"

echo "Diagnostic output: ${OUTPUT_ROOT}"
echo "No normalization override will be applied."
echo "The six downstream runs are sequential and use the same seed/config."

for index in "${!LABELS[@]}"; do
    label="${LABELS[${index}]}"
    checkpoint="${CHECKPOINTS[${index}]}"
    run_dir="${OUTPUT_ROOT}/runs/${label}"
    tb_dir="${OUTPUT_ROOT}/tensorboard/${label}"
    log_file="${OUTPUT_ROOT}/logs/${label}.log"

    mkdir -p "${run_dir}" "${tb_dir}"

    cmd=(
        "${PYTHON_BIN}" train_task_force.py
        +experiment=xela/task/force/dinov2
        "experiment_name=${EXPERIMENT_NAME}"
        "run_name=${label}"
        "task.checkpoint_encoder=${checkpoint}"
        ckpt_path=null
        resume_id=null
        seed=42
        trainer.devices=4
        data.train_dataloader.batch_size=16
        train_data_budget=1.0
        val_data_budget=1.0
        "hydra.run.dir=${run_dir}"
        "trainer.save_checkpoint_dir=${run_dir}/checkpoints"
        "tensorboard.log_dir=${tb_dir}"
    )

    {
        echo
        echo "===== ${label} ====="
        echo "checkpoint=${checkpoint}"
        printf 'command='
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } | tee "${log_file}"

    set +e
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
    train_exit_code=${PIPESTATUS[0]}
    set -e

    if [[ "${train_exit_code}" -eq 0 ]]; then
        run_status="ok"
    else
        run_status="train_failed"
    fi

    # Parse rank-zero TensorBoard scalars. The parser chooses the event file
    # containing test/rmse, so auxiliary distributed-rank event files are ignored.
    "${PYTHON_BIN}" - \
        "${label}" "${checkpoint}" "${run_status}" "${train_exit_code}" \
        "${tb_dir}" "${run_dir}/metrics.json" "${SUMMARY_CSV}" <<'PY'
import csv
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

label, checkpoint, status, exit_code, tb_dir, json_path, csv_path = sys.argv[1:]
event_files = sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True))

selected = None
selected_accumulator = None
for event_file in event_files:
    try:
        accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
        accumulator.Reload()
    except Exception:
        continue
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if "test/rmse" in scalar_tags:
        if selected is None or os.path.getsize(event_file) > os.path.getsize(selected):
            selected = event_file
            selected_accumulator = accumulator

def scalar_values(tag):
    if selected_accumulator is None:
        return []
    try:
        return [float(event.value) for event in selected_accumulator.Scalars(tag)]
    except (KeyError, IndexError):
        return []

def last(tag):
    values = scalar_values(tag)
    return values[-1] if values else None

val_values = scalar_values("val/rmse")
metrics = {
    "label": label,
    "checkpoint": checkpoint,
    "status": status if selected_accumulator is not None else f"{status}_missing_test_metrics",
    "exit_code": int(exit_code),
    "test_rmse": last("test/rmse"),
    "test_rmse_x": last("test/rmse_x"),
    "test_rmse_y": last("test/rmse_y"),
    "test_rmse_z": last("test/rmse_z"),
    "best_val_rmse": min(val_values) if val_values else None,
    "last_val_rmse": val_values[-1] if val_values else None,
    "last_train_rmse": last("train/rmse"),
    "event_file": selected,
    "all_event_files": event_files,
}

with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

fields = [
    "label", "checkpoint", "status", "exit_code", "test_rmse",
    "test_rmse_x", "test_rmse_y", "test_rmse_z", "best_val_rmse",
    "last_val_rmse", "last_train_rmse", "event_file",
]
with open(csv_path, "a", encoding="utf-8", newline="") as handle:
    csv.writer(handle).writerow([metrics[field] for field in fields])
PY

    echo "Finished ${label}: exit_code=${train_exit_code}"
done

"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${OUTPUT_ROOT}/summary.md" <<'PY'
import csv
import sys

csv_path, markdown_path = sys.argv[1:]
with open(csv_path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

def fmt(value):
    if value in (None, "", "None"):
        return "-"
    try:
        return f"{float(value):.6f}"
    except ValueError:
        return value

lines = [
    "# DINOv2 resume quality diagnostic",
    "",
    "All checkpoints were evaluated with the same downstream-force config, seed 42, "
    "four devices, batch size 16 per device, and without a normalization override.",
    "",
    "| checkpoint label | status | test RMSE | Fx | Fy | Fz | best val RMSE |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['label']} | {row['status']} | {fmt(row['test_rmse'])} | "
        f"{fmt(row['test_rmse_x'])} | {fmt(row['test_rmse_y'])} | "
        f"{fmt(row['test_rmse_z'])} | {fmt(row['best_val_rmse'])} |"
    )

with open(markdown_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY

echo
echo "Done. Send me this directory for analysis:"
echo "${OUTPUT_ROOT}"
echo "Main comparison: ${OUTPUT_ROOT}/summary.md"

# Reproduce the Xela workbook dated 2026-10-09

This directory covers **exactly the 38 experiment IDs** in
the privately retained Xela workbook: the current mask matrix,
the three preceding Xela series, and historical JEPA2L2R, connected-context
JEPA2L2R, I-JEPA and MAE. The worksheet snapshot is 09 October, 11:44 Moscow time.
Its partial results remain partial; exporting a recipe does not mean it finished.

## Contents and provenance

- `manifest.json`: names, baseline differences, original commands, original
  config hashes, seeds, resource requirements and pretrain-to-head dependencies.
- `configs/`: 1128 full Hydra configs with storage locations made portable:
  **102 pretrains + 1026 heads**. No ignored legacy config directory is needed.
- `original_configs/`: the corresponding original configs before path changes.
  1012 were read from saved `.hydra/config.yaml` files; 116 unstarted stages were
  composed from their recorded launch commands. Each stage identifies its source.
- The workbook and numerical experiment results are retained privately and
  are not included in this source-only branch or its history.
- `environment.yml`, `environment.json`: the exported Conda environment and
  observed Python/package/driver versions at packaging time.
- `data_manifest.json`, `data_files.txt`: input file identities and transfer list.
- `directory_order.json`: original episode-directory enumeration used by the
  Force/Pose split logic, preserved even when files are copied to another system.
- `code_sha256.json`: frozen implementation hashes, checked before launching.
- `validation.json`: checks performed on the packaged implementation.

The code preserves the experimental flags: random context crop, overlap measured
against the smaller target, constrained sampling, teacher `eval()`, size-weighted
target loss, and the optional A2/A4 batch policy. Omitted loss settings still mean
the original MSE with equal target weights. Unrelated DECO tokenizer experiments
and live queue controllers are not part of this change.

The old runs did **not** record a complete immutable source revision per stage.
This source-only branch freezes a compatible implementation and the recorded protocols; it
does not reconstruct an unknown historical commit or promise bitwise-identical
metrics. Teacher `eval()` is independent of whole-program numerical determinism.
The original CUDA/cuDNN behavior, including the head entrypoint's benchmark mode,
is retained. Full 500-epoch retraining was not performed to validate this export.

## Environment and data

Use Linux with NVIDIA GPUs and the recorded Conda environment. From the repository
root, create a separate environment (do not alter one running other experiments):

```bash
conda env create -f repro/xela_20261009/environment.yml
conda activate tactile-xela-repro
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
```

The environment snapshot includes the CUDA 12.6 PyTorch wheel index. Environment
installation on a second machine is separate from the clean-checkout validation;
binary compatibility still depends on its NVIDIA driver and platform.

Set the dataset path to the existing Sparsh-skin directory. It must contain
`xela/pretraining/extracted/` and `downstream_tasks/`. The latter comes from
`facebook/sparsh-skin-dataset` at commit
`fc1d66893d775293d8bd7a411cdcda1339c2b810`; the extracted pretraining files are
outside that dataset's Git tree and are identified by freshly calculated SHA-256.
The dataset Git checkout alone is therefore insufficient: obtain the extracted
pretraining release too, or copy the existing verified files listed in
`data_files.txt`. Raw data and model checkpoints are external to this source repo.

```bash
export XELA_DATA_ROOT=/absolute/path/to/sparsh-skin-dataset
python -m repro.xela_20261009.run verify-data --hashes
```

`verify-data` without `--hashes` performs a faster existence/size check. LFS hashes
in the manifest are the upstream expected content hashes; extracted pretraining
files were streamed and hashed during packaging. All downstream LFS working files
were also checked against their expected hashes. Train/validation recording lists,
normalization and preprocessing settings are in the full configs. Fresh outputs
use a separate cache, avoiding dependence on a cache from the experiment server.
The runner sets `XELA_REPRO_DIRECTORY_ORDER` so that the original filesystem order
is restored before seeded splitting. Without that variable the existing data
loaders keep their normal behavior. This opt-in changes ordering on a copied
dataset to match the captured source, not the original split algorithm.

## Inspect, validate and launch

```bash
python -m repro.xela_20261009.run list
python -m repro.xela_20261009.run validate --models
python -m repro.xela_20261009.run plan --case a2 --output-root /absolute/path/to/new-results
```

`validate --models` resolves every selected config, checks source and config
hashes, confirms that only storage paths changed, imports all Hydra targets and
constructs distinct encoder/probe/algorithm configurations on CPU. It does not
load the datasets or start a training job. `plan` only prints stage commands and
dependencies; it never starts or changes a queue.
Construction checks use explicit zero/one normalization and 14 placeholder class
labels for values normally filled by the data loader. Training computes the real
values from the dataset; the stored configs are not changed by these checks.

Launch one stage in its own tmux window with an allocated GPU, from the repo root:

```bash
python -m repro.xela_20261009.run run --case a2 --pretrain-seed 42 --stage pretrain \
  --output-root /absolute/path/to/new-results --gpus 0
python -m repro.xela_20261009.run run --case a2 --pretrain-seed 42 --stage force-h42 \
  --output-root /absolute/path/to/new-results --gpus 0
```

Each invocation is an independent process with its own log, output directory and
exit marker. A failed stage does not cancel sibling stages. The runner refuses to
overwrite a prior output directory. `run` always selects one explicit stage.

Standard pretrain seeds and head seeds are **42, 17, 3407**. Every regular case
has three pretrains, each with three heads for each of Force/Object/Pose (27 heads).
`a2-lr5e5`, `a2-lr2e4`, `a4-lr5e5`, `a4-lr2e4` have no new pretraining: their heads
use the matching seed of A2 or A4. Run those dependencies first with the same
output root. The full recipes also preserve withdrawn DP01 seeds, deferred Q4,
and paused old-both; these are explicitly marked in the manifest and do not
imply an instruction to resume them in the live queue.

Current matrix pretrains use one GPU with batch 256. Historical configurations
can require four GPUs with local batch 64: preserve the recorded GPU count and
local batch because context cropping depends on the local batch. The runner
checks the number of allocated GPUs. Heads use one GPU per process and can share
an allocated GPU three at a time, as in the experiment protocol. On the existing
primary server, physical GPU 3 remains reserved; do not launch a four-GPU recipe
there under the current reservation.

## Metrics and checks

```bash
# Compute the same statistics from a new reproduction tree.
python -m repro.xela_20261009.summarize --output-root /absolute/path/to/new-results > xela_rerun.csv
python -m pip install pytest==8.4.2
python -m pytest -q tests/test_jepa_graph_masking.py tests/test_jepa_batch_policy.py \
  tests/test_xela_ijepa_2d.py tests/test_xela_repro_protocol.py
python tests/test_xela_target_loss_size_weighting.py
```

Force is the pooled RMSE in N; Object is accuracy; Pose is the fraction below
2 cm for X/Y and below 5 degrees for theta. Mean and sample standard deviation
(`ddof=1`) are over completed heads, also reported separately for each pretrain.
Nine heads per task represent three independent pretrains, not nine. Incomplete
rows use only completed heads and retain their counts. Model selection and early
stopping remain the recorded validation protocol; the saved test metrics are
for reporting.

The private Excel snapshot was checked locally before packaging. Public validation
metadata records that check without including its numerical results. This branch
is based directly on the original main commit and never contains the private
workbook/results in its ancestry.

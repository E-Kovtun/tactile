"""Backfill downstream predictions and build paired significance reports."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Mapping

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from tactile_ssl.evaluation.artifacts import load_run_metadata
from tactile_ssl.evaluation.backfill import backfill_evaluation_artifact
from tactile_ssl.evaluation.report import write_reports
from tactile_ssl.evaluation.statistics_cache import analyze_pair_cached


TASK_ALIASES = {
    "force": "force",
    "pose": "pose",
    "relative_pose": "pose",
    "relative_pose_estimation": "pose",
    "object": "object_classification",
    "object_classification": "object_classification",
}


def _has_evaluation_or_checkpoint(run_root: Path) -> bool:
    if (run_root / "evaluation" / "test_predictions.npz").is_file():
        return True
    checkpoint_dir = run_root / "checkpoints"
    return checkpoint_dir.is_dir() and any(
        path.is_file() and path.suffix in {".ckpt", ".pth", ".pt"}
        for path in checkpoint_dir.iterdir()
    )


def _resolve_run_directory(experiment: Mapping[str, Any]) -> Path:
    raw_directory = hydra.utils.to_absolute_path(str(experiment["directory"]))
    if not glob.has_magic(raw_directory):
        return Path(raw_directory).expanduser().resolve()

    candidates = sorted(
        path.resolve()
        for value in glob.glob(raw_directory, recursive=True)
        if (path := Path(value)).is_dir() and _has_evaluation_or_checkpoint(path)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed run with an evaluation artifact or checkpoint matches {raw_directory!r}"
        )
    selection = str(experiment.get("select", "error"))
    if len(candidates) > 1 and selection != "latest":
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            f"Experiment pattern {raw_directory!r} is ambiguous. "
            "Set select: latest or use an exact directory:\n" + formatted
        )
    selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    print(f"Resolved {raw_directory} -> {selected}")
    return selected


def _variant(use_spatial_coords: Any, override: Any = None) -> str:
    if override is not None:
        normalized = str(override).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"only_signal", "signal_only", "no_coords"}:
            return "only signal"
        if normalized in {"signal_pos", "signal_plus_pos", "base"}:
            return "base (signal+pos)"
        raise ValueError(f"Unknown report_variant override: {override!r}")
    if use_spatial_coords is None:
        raise ValueError("Could not determine use_spatial_coords from the resolved experiment config")
    return "base (signal+pos)" if bool(use_spatial_coords) else "only signal"


def _row(
    *,
    task: str,
    block_id: str,
    block_name: str,
    baseline_id: str,
    baseline_name: str,
    statistics_cache_key: str,
    statistics_cache_hit: bool,
    experiment: Mapping[str, Any],
    is_baseline: bool,
    variant: str,
    result: Any,
    artifact: Any,
) -> Dict[str, Any]:
    manifest = artifact.manifest
    return {
        "task": task,
        "comparison_block_id": block_id,
        "comparison_block_name": block_name,
        "baseline_id": baseline_id,
        "baseline_name": baseline_name,
        "statistics_cache_key": statistics_cache_key,
        "statistics_cache_hit": statistics_cache_hit,
        "experiment_id": str(experiment["id"]),
        "method": str(experiment["name"]),
        "is_baseline": is_baseline,
        "variant": variant,
        "metric": result.metric,
        "estimate": result.estimate,
        "bootstrap_se": result.bootstrap_se,
        "delta": result.delta,
        "delta_se": result.delta_se,
        "improvement": result.improvement,
        "raw_pvalue": result.raw_pvalue,
        "checkpoint": manifest.get("checkpoint"),
        "artifact_path": str(artifact.path),
        "num_examples": manifest.get("num_examples"),
        "num_groups": manifest.get("num_groups"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "seed": manifest.get("seed"),
    }


def _comparison_blocks(task_name: str, task_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize the block config while retaining legacy single-baseline support."""
    configured_blocks = task_cfg.get("comparison_blocks")
    if configured_blocks:
        blocks = [dict(block) for block in configured_blocks]
        block_ids = [str(block.get("id", "")) for block in blocks]
        if any(not block_id for block_id in block_ids):
            raise ValueError(f"Every comparison block in {task_name} must have a non-empty id")
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"Duplicate comparison block id in task {task_name}")
        return blocks

    # Legacy format: baseline points to one entry in the flat experiment list.
    experiments = [dict(experiment) for experiment in task_cfg.get("experiments", [])]
    if not experiments:
        return []
    baseline_id = str(task_cfg.get("baseline"))
    matching = [experiment for experiment in experiments if str(experiment["id"]) == baseline_id]
    if len(matching) != 1:
        raise ValueError(
            f"Baseline {baseline_id!r} is not listed exactly once in {task_name}.experiments"
        )
    return [
        {
            "id": "default",
            "name": "",
            "baseline": matching[0],
            "experiments": [
                experiment for experiment in experiments if str(experiment["id"]) != baseline_id
            ],
        }
    ]


def run_analysis(cfg: DictConfig) -> tuple:
    statistics_cfg = cfg.statistics
    output_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_dir))).resolve()
    statistics_cache_dir = output_dir / "statistics_cache"
    report_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []

    for configured_task, task_cfg in cfg.tasks.items():
        if task_cfg is None:
            continue
        if configured_task not in TASK_ALIASES:
            raise ValueError(f"Unknown task section: {configured_task}")
        task = TASK_ALIASES[configured_task]
        blocks = _comparison_blocks(configured_task, task_cfg)
        artifact_cache: Dict[Path, Any] = {}

        for block in blocks:
            block_id = str(block["id"])
            block_name = str(block.get("name", block_id))
            if not isinstance(block.get("baseline"), Mapping):
                raise ValueError(f"Comparison block {configured_task}.{block_id} must define baseline")
            baseline_experiment = dict(block["baseline"])
            candidates = [dict(experiment) for experiment in block.get("experiments", [])]
            experiments = [baseline_experiment, *candidates]
            ids = [str(experiment["id"]) for experiment in experiments]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"Duplicate experiment id in comparison block {configured_task}.{block_id}"
                )
            baseline_id = str(baseline_experiment["id"])
            baseline_name = str(baseline_experiment["name"])

            loaded = {}
            for experiment in experiments:
                run_root = _resolve_run_directory(experiment)
                if run_root not in artifact_cache:
                    artifact_cache[run_root] = backfill_evaluation_artifact(task, run_root)
                artifact, checkpoint, config_path = artifact_cache[run_root]
                if artifact.manifest.get("task") != task:
                    raise ValueError(
                        f"Artifact {artifact.path} declares task={artifact.manifest.get('task')!r}, "
                        f"expected {task!r}"
                    )
                metadata = load_run_metadata(run_root)
                spatial = artifact.manifest.get("use_spatial_coords")
                if spatial is None:
                    spatial = metadata["use_spatial_coords"]
                variant = _variant(spatial, experiment.get("report_variant"))
                experiment_id = str(experiment["id"])
                loaded[experiment_id] = (experiment, artifact, variant)
                provenance_rows.append(
                    {
                        "task": task,
                        "comparison_block_id": block_id,
                        "comparison_block_name": block_name,
                        "baseline_id": baseline_id,
                        "baseline_name": baseline_name,
                        "experiment_id": experiment_id,
                        "method": str(experiment["name"]),
                        "is_baseline": experiment_id == baseline_id,
                        "variant": variant,
                        "directory": str(run_root),
                        "checkpoint": str(checkpoint),
                        "artifact_path": str(artifact.path),
                        "config_path": str(config_path),
                        "dataset_fingerprint": artifact.manifest.get("dataset_fingerprint"),
                        "num_examples": artifact.manifest.get("num_examples"),
                        "num_groups": artifact.manifest.get("num_groups"),
                        "seed": artifact.manifest.get("seed"),
                    }
                )

            _, baseline_artifact, baseline_variant = loaded[baseline_id]
            baseline_results, baseline_cache_key, baseline_cache_hit = analyze_pair_cached(
                statistics_cache_dir,
                task,
                baseline_artifact,
                None,
                bootstrap_samples=int(statistics_cfg.bootstrap_samples),
                permutation_samples=int(statistics_cfg.permutation_samples),
                seed=int(statistics_cfg.seed),
            )
            print(
                f"Statistics cache {'HIT' if baseline_cache_hit else 'MISS'}: "
                f"{task}/{block_id}/{baseline_id}"
            )
            report_rows.extend(
                _row(
                    task=task,
                    block_id=block_id,
                    block_name=block_name,
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    statistics_cache_key=baseline_cache_key,
                    statistics_cache_hit=baseline_cache_hit,
                    experiment=baseline_experiment,
                    is_baseline=True,
                    variant=baseline_variant,
                    result=result,
                    artifact=baseline_artifact,
                )
                for result in baseline_results
            )
            for experiment in candidates:
                experiment_id = str(experiment["id"])
                candidate_experiment, candidate_artifact, candidate_variant = loaded[experiment_id]
                candidate_results, candidate_cache_key, candidate_cache_hit = analyze_pair_cached(
                    statistics_cache_dir,
                    task=task,
                    baseline=baseline_artifact,
                    candidate=candidate_artifact,
                    bootstrap_samples=int(statistics_cfg.bootstrap_samples),
                    permutation_samples=int(statistics_cfg.permutation_samples),
                    seed=int(statistics_cfg.seed),
                )
                print(
                    f"Statistics cache {'HIT' if candidate_cache_hit else 'MISS'}: "
                    f"{task}/{block_id}/{experiment_id} vs {baseline_id}"
                )
                report_rows.extend(
                    _row(
                        task=task,
                        block_id=block_id,
                        block_name=block_name,
                        baseline_id=baseline_id,
                        baseline_name=baseline_name,
                        statistics_cache_key=candidate_cache_key,
                        statistics_cache_hit=candidate_cache_hit,
                        experiment=candidate_experiment,
                        is_baseline=False,
                        variant=candidate_variant,
                        result=result,
                        artifact=candidate_artifact,
                    )
                    for result in candidate_results
                )

    if not report_rows:
        raise ValueError("No experiments configured under tasks")
    return write_reports(output_dir, report_rows, provenance_rows)


@hydra.main(version_base="1.3", config_path="../config/significance", config_name="default")
def main(cfg: DictConfig) -> None:
    xlsx_path, csv_path = run_analysis(cfg)
    print(f"XLSX report: {xlsx_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()

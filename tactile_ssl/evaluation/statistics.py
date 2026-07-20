"""Paired, group-aware downstream significance statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import EvaluationArtifact


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str  # rmse or accuracy
    axis: Optional[int] = None
    threshold: Optional[float] = None


@dataclass(frozen=True)
class MetricResult:
    metric: str
    estimate: float
    bootstrap_se: float
    delta: Optional[float] = None
    delta_se: Optional[float] = None
    improvement: Optional[float] = None
    raw_pvalue: Optional[float] = None


TASK_METRICS: Mapping[str, Sequence[MetricSpec]] = {
    "force": (
        MetricSpec("rmse", "rmse"),
        MetricSpec("rmse_x", "rmse", axis=0),
        MetricSpec("rmse_y", "rmse", axis=1),
        MetricSpec("rmse_z", "rmse", axis=2),
    ),
    "pose": (
        MetricSpec("rmse_x", "rmse", axis=0),
        MetricSpec("rmse_y", "rmse", axis=1),
        MetricSpec("rmse_theta", "rmse", axis=2),
        MetricSpec("acc_x", "accuracy", axis=0, threshold=0.02),
        MetricSpec("acc_y", "accuracy", axis=1, threshold=0.02),
        MetricSpec("acc_theta", "accuracy", axis=2, threshold=5.0),
    ),
    "object_classification": (MetricSpec("acc", "accuracy"),),
}


def _occurrence_keys(sample_ids: np.ndarray) -> List[Tuple[int, int]]:
    counts: Dict[int, int] = {}
    keys = []
    for raw in np.asarray(sample_ids).reshape(-1):
        sample_id = int(raw)
        occurrence = counts.get(sample_id, 0)
        keys.append((sample_id, occurrence))
        counts[sample_id] = occurrence + 1
    return keys


def align_candidate(baseline: EvaluationArtifact, candidate: EvaluationArtifact) -> EvaluationArtifact:
    """Reorder candidate by ``(sample_id, occurrence)`` and reject any mismatch."""
    baseline_version = baseline.manifest.get("metric_version")
    candidate_version = candidate.manifest.get("metric_version")
    if baseline_version != candidate_version:
        raise ValueError(
            f"Metric versions differ: baseline={baseline_version!r}, candidate={candidate_version!r}"
        )
    baseline_keys = _occurrence_keys(baseline.sample_id)
    candidate_keys = _occurrence_keys(candidate.sample_id)
    if set(baseline_keys) != set(candidate_keys) or len(baseline_keys) != len(candidate_keys):
        baseline_multiset: Dict[int, int] = {}
        candidate_multiset: Dict[int, int] = {}
        for sample_id, _ in baseline_keys:
            baseline_multiset[sample_id] = baseline_multiset.get(sample_id, 0) + 1
        for sample_id, _ in candidate_keys:
            candidate_multiset[sample_id] = candidate_multiset.get(sample_id, 0) + 1
        missing = sorted((baseline_multiset.keys() - candidate_multiset.keys()))[:5]
        extra = sorted((candidate_multiset.keys() - baseline_multiset.keys()))[:5]
        raise ValueError(
            "Baseline and candidate evaluate different sample_id multisets: "
            f"baseline={len(baseline_keys)}, candidate={len(candidate_keys)}, "
            f"missing examples={missing}, extra examples={extra}"
        )
    position = {key: index for index, key in enumerate(candidate_keys)}
    order = np.asarray([position[key] for key in baseline_keys], dtype=np.int64)
    aligned = EvaluationArtifact(
        sample_id=candidate.sample_id[order],
        group_id=candidate.group_id[order],
        y_true=candidate.y_true[order],
        y_pred=candidate.y_pred[order],
        manifest=candidate.manifest,
        path=candidate.path,
    )
    if not np.array_equal(np.asarray(baseline.group_id), np.asarray(aligned.group_id)):
        raise ValueError("group_id differs after matching sample occurrences")
    if baseline.y_true.shape != aligned.y_true.shape or not np.allclose(
        baseline.y_true, aligned.y_true, rtol=0.0, atol=1e-8, equal_nan=True
    ):
        raise ValueError("y_true differs between baseline and candidate after sample matching")
    return aligned


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray, spec: MetricSpec) -> np.ndarray:
    if spec.axis is not None:
        if y_true.ndim < 2 or y_true.shape[-1] <= spec.axis:
            raise ValueError(f"Metric {spec.name} requires coordinate axis {spec.axis}")
        y_true = y_true[..., spec.axis]
        y_pred = y_pred[..., spec.axis]
    if spec.kind == "rmse":
        return np.square(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    if spec.threshold is None:
        return np.equal(y_pred, y_true).astype(np.float64)
    return (np.abs(np.asarray(y_pred) - np.asarray(y_true)) < spec.threshold).astype(np.float64)


def _group_sufficient_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_id: np.ndarray,
    spec: MetricSpec,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    values = _metric_values(y_true, y_pred, spec)
    values = values.reshape(values.shape[0], -1)
    numerator = np.empty(len(groups), dtype=np.float64)
    denominator = np.empty(len(groups), dtype=np.float64)
    for index, group in enumerate(groups):
        selected = values[np.asarray(group_id) == group]
        numerator[index] = selected.sum(dtype=np.float64)
        denominator[index] = selected.size
    return numerator, denominator


def _score(numerator: np.ndarray, denominator: np.ndarray, kind: str) -> np.ndarray:
    value = numerator / denominator
    return np.sqrt(value) if kind == "rmse" else value


def _bootstrap_scores(
    baseline_stats: Tuple[np.ndarray, np.ndarray],
    candidate_stats: Tuple[np.ndarray, np.ndarray],
    *,
    kind: str,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    group_count = len(baseline_stats[0])
    baseline_values = np.empty(samples, dtype=np.float64)
    candidate_values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, group_count, size=(stop - start, group_count))
        base_num = baseline_stats[0][indices].sum(axis=1)
        base_den = baseline_stats[1][indices].sum(axis=1)
        cand_num = candidate_stats[0][indices].sum(axis=1)
        cand_den = candidate_stats[1][indices].sum(axis=1)
        baseline_values[start:stop] = _score(base_num, base_den, kind)
        candidate_values[start:stop] = _score(cand_num, cand_den, kind)
    return baseline_values, candidate_values


def _paired_permutation_pvalue(
    baseline_stats: Tuple[np.ndarray, np.ndarray],
    candidate_stats: Tuple[np.ndarray, np.ndarray],
    *,
    kind: str,
    observed_delta: float,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 1024,
) -> float:
    group_count = len(baseline_stats[0])
    extreme = 0
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        swap = rng.integers(0, 2, size=(count, group_count), dtype=np.int8).astype(bool)
        base_num = np.where(swap, candidate_stats[0], baseline_stats[0]).sum(axis=1)
        base_den = np.where(swap, candidate_stats[1], baseline_stats[1]).sum(axis=1)
        cand_num = np.where(swap, baseline_stats[0], candidate_stats[0]).sum(axis=1)
        cand_den = np.where(swap, baseline_stats[1], candidate_stats[1]).sum(axis=1)
        permuted_delta = _score(cand_num, cand_den, kind) - _score(base_num, base_den, kind)
        extreme += int(np.count_nonzero(np.abs(permuted_delta) >= abs(observed_delta) - 1e-15))
    return (extreme + 1.0) / (samples + 1.0)


def analyze_pair(
    task: str,
    baseline: EvaluationArtifact,
    candidate: Optional[EvaluationArtifact],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> List[MetricResult]:
    if task not in TASK_METRICS:
        raise ValueError(f"Unsupported downstream task: {task}")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    if permutation_samples < 1:
        raise ValueError("permutation_samples must be positive")
    candidate = align_candidate(baseline, candidate) if candidate is not None else None
    groups = np.unique(baseline.group_id)
    if groups.size == 0:
        raise ValueError("Cannot analyze an empty evaluation artifact")

    seed_sequence = np.random.SeedSequence(seed)
    metric_seeds = seed_sequence.spawn(len(TASK_METRICS[task]))
    results = []
    for spec, metric_seed in zip(TASK_METRICS[task], metric_seeds):
        baseline_stats = _group_sufficient_statistics(
            baseline.y_true, baseline.y_pred, baseline.group_id, spec, groups
        )
        compared = baseline if candidate is None else candidate
        candidate_stats = _group_sufficient_statistics(
            compared.y_true, compared.y_pred, compared.group_id, spec, groups
        )
        baseline_estimate = float(_score(baseline_stats[0].sum(), baseline_stats[1].sum(), spec.kind))
        candidate_estimate = float(_score(candidate_stats[0].sum(), candidate_stats[1].sum(), spec.kind))
        bootstrap_rng, permutation_rng = [np.random.default_rng(child) for child in metric_seed.spawn(2)]
        baseline_boot, candidate_boot = _bootstrap_scores(
            baseline_stats,
            candidate_stats,
            kind=spec.kind,
            samples=bootstrap_samples,
            rng=bootstrap_rng,
        )
        if candidate is None:
            results.append(
                MetricResult(
                    metric=spec.name,
                    estimate=baseline_estimate,
                    bootstrap_se=float(np.std(baseline_boot, ddof=1)),
                )
            )
            continue

        delta = candidate_estimate - baseline_estimate
        if baseline_estimate == 0:
            improvement = np.nan
        elif spec.kind == "rmse":
            improvement = (baseline_estimate - candidate_estimate) / baseline_estimate
        else:
            improvement = (candidate_estimate - baseline_estimate) / baseline_estimate
        results.append(
            MetricResult(
                metric=spec.name,
                estimate=candidate_estimate,
                bootstrap_se=float(np.std(candidate_boot, ddof=1)),
                delta=delta,
                delta_se=float(np.std(candidate_boot - baseline_boot, ddof=1)),
                improvement=float(improvement),
                raw_pvalue=_paired_permutation_pvalue(
                    baseline_stats,
                    candidate_stats,
                    kind=spec.kind,
                    observed_delta=delta,
                    samples=permutation_samples,
                    rng=permutation_rng,
                ),
            )
        )
    return results

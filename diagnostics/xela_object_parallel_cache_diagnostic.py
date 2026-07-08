#!/usr/bin/env python
"""Validate object-classification Xela dataset loading with cache workers.

This diagnostic targets train_task_object.py changes directly:

- object train/val/test split construction
- object labels and class-size accounting
- sequential vs parallel XelaSSLDataset construction
- cache disabled, cold-cache, and warm-cache behavior
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEQUENCES = [
    "corn",
    "rubikscube",
    "metalcup",
    "cup",
    "ball",
    "drill",
    "mustard",
    "lego",
    "pringle",
    "legopool",
    "watermelon",
    "popcorn",
    "loofah",
    "slipper",
]

COMPARE_ARRAY_KEYS = [
    "timestamps",
    "xela_array",
    "joint_angles",
    "joint_effort",
    "joint_poses",
    "sensor_positions",
    "data_idxs",
]


@dataclass(frozen=True)
class TaskSpec:
    split: str
    sequence: str
    episode_id: int
    object_label: int
    path: Path


@dataclass
class Report:
    path: Path
    failures: list[str]

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def line(self, text: str = "") -> None:
        print(text)
        self._fh.write(text + "\n")
        self._fh.flush()

    def section(self, title: str) -> None:
        self.line()
        self.line("=" * 100)
        self.line(title)
        self.line("=" * 100)

    def kv(self, key: str, value: Any) -> None:
        self.line(f"{key}: {value}")

    def fail(self, name: str, detail: str) -> None:
        message = f"{name}: {detail}"
        self.failures.append(message)
        self.line(f"  RESULT: FAIL - {detail}")


def resolve_pretrain_base(data_root: Path) -> Path:
    candidates = [
        data_root / "xela/pretraining/extracted",
        data_root / "pretraining",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_baseline_and_urdf(data_root: Path) -> tuple[Path, Path]:
    baseline_candidates = [
        data_root / "xela/pretraining/extracted/baseline/xela/data.pkl",
        data_root / "pretraining/baseline/xela/data.pkl",
    ]
    urdf_candidates = [
        data_root / "xela/pretraining/extracted/urdf/ahrcpcpn.urdf",
        data_root / "pretraining/urdf/ahrcpcpn.urdf",
    ]
    baseline = next((path for path in baseline_candidates if path.exists()), baseline_candidates[0])
    urdf = next((path for path in urdf_candidates if path.exists()), urdf_candidates[0])
    return baseline, urdf


def make_dataset_cfg(data_root: Path, cache_root: Path, cache_enabled: bool):
    from omegaconf import OmegaConf

    pretrain_base = resolve_pretrain_base(data_root)
    baseline, urdf = resolve_baseline_and_urdf(data_root)
    return OmegaConf.create(
        {
            "_target_": "tactile_ssl.data.xela_tactile.XelaSSLDataset",
            "config": {
                "window_time": 0.1,
                "window_overlap": 0.0,
                "interpolating_freq": 100,
                "subtract_baseline": True,
                "smooth_data": False,
                "bias_noise_std": 0,
                "bias_range": 0,
                "cache": {
                    "enabled": cache_enabled,
                    "root": str(cache_root),
                    "force_recompute": False,
                    "fingerprint_mode": "stats",
                    "log_hits": True,
                    "num_workers": 0,
                },
                "features": {"use_spatial_coords": True},
            },
            "data_path": str(pretrain_base),
            "baseline_signal_path": str(baseline),
            "xela_urdf_path": str(urdf),
            "load_images": False,
        }
    )


def build_task_specs(data_root: Path, max_sequences: int | None) -> list[TaskSpec]:
    base = resolve_pretrain_base(data_root)
    specs: list[TaskSpec] = []
    selected_sequences = SEQUENCES[:max_sequences] if max_sequences is not None else SEQUENCES
    split_ids = {
        "train": list(range(8)),
        "val": [8],
        "test": [9],
    }
    for object_label, sequence in enumerate(selected_sequences):
        for split, episode_ids in split_ids.items():
            for episode_id in episode_ids:
                path = base / sequence / str(episode_id)
                if (path / "xela/data.pkl").exists():
                    specs.append(TaskSpec(split, sequence, episode_id, object_label, path))
    return specs


def load_dataset(dataset_cfg, spec: TaskSpec):
    import hydra

    return hydra.utils.instantiate(
        deepcopy(dataset_cfg),
        data_path=str(spec.path),
        object_class=spec.object_label,
    )


def load_many(dataset_cfg, specs: list[TaskSpec], num_workers: int):
    if num_workers > 0 and len(specs) > 1:
        max_workers = min(num_workers, len(specs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda spec: load_dataset(dataset_cfg, spec), specs))
    return [load_dataset(dataset_cfg, spec) for spec in specs]


def split_datasets(specs: list[TaskSpec], datasets: list) -> dict[str, list]:
    result = {"train": [], "val": [], "test": []}
    for spec, dataset in zip(specs, datasets):
        result[spec.split].append(dataset)
    return result


def object_class_sizes(datasets: list, num_classes: int) -> np.ndarray:
    sizes = np.zeros(num_classes, dtype=np.int64)
    for dataset in datasets:
        sizes[int(dataset.object_label)] += len(dataset)
    return sizes


def dataset_value(dataset, key: str):
    if key == "data_idxs":
        return dataset.data_idxs
    return getattr(dataset, key)


def compare_arrays(name: str, left: Any, right: Any, report: Report, atol: float = 1e-6, rtol: float = 1e-5) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    report.line(f"[COMPARE] {name}")
    report.kv("  left_shape", left.shape)
    report.kv("  right_shape", right.shape)
    if left.shape != right.shape:
        report.fail(name, "shape mismatch")
        return
    if left.size == 0 and right.size == 0:
        report.line("  RESULT: PASS - both empty")
        return
    allclose = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
    diff = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    abs_diff = np.abs(diff)
    with np.errstate(all="ignore"):
        max_abs_diff = float(np.nanmax(abs_diff)) if abs_diff.size else 0.0
        over_tol = int(np.count_nonzero(abs_diff > (atol + rtol * np.abs(right))))
    report.kv("  allclose", allclose)
    report.kv("  max_abs_diff", max_abs_diff)
    report.kv("  over_tol_count", over_tol)
    if allclose:
        report.line("  RESULT: PASS")
    else:
        report.fail(name, f"not allclose, max_abs_diff={max_abs_diff}, over_tol_count={over_tol}")


def compare_dataset_lists(name: str, specs: list[TaskSpec], left: list, right: list, report: Report) -> None:
    report.section(f"Compare Dataset Lists: {name}")
    if len(left) != len(right):
        report.fail(name, f"dataset count mismatch: {len(left)} != {len(right)}")
        return
    for spec, left_dataset, right_dataset in zip(specs, left, right):
        prefix = f"{name}.{spec.split}.{spec.sequence}/{spec.episode_id}"
        report.line(f"[DATASET] {prefix}")
        if len(left_dataset) != len(right_dataset):
            report.fail(f"{prefix}.len", f"{len(left_dataset)} != {len(right_dataset)}")
        if left_dataset.object_label != right_dataset.object_label:
            report.fail(f"{prefix}.object_label", f"{left_dataset.object_label} != {right_dataset.object_label}")
        if set(left_dataset.artifact_keys) != set(right_dataset.artifact_keys):
            report.fail(f"{prefix}.artifact_keys.keys", f"{left_dataset.artifact_keys} != {right_dataset.artifact_keys}")
        for artifact_name, left_key in left_dataset.artifact_keys.items():
            right_key = right_dataset.artifact_keys.get(artifact_name)
            if left_key != right_key:
                report.fail(f"{prefix}.artifact_keys.{artifact_name}", f"{left_key} != {right_key}")
        for key in COMPARE_ARRAY_KEYS:
            compare_arrays(f"{prefix}.{key}", dataset_value(left_dataset, key), dataset_value(right_dataset, key), report)


def compare_object_task_state(name: str, specs: list[TaskSpec], left: list, right: list, report: Report) -> None:
    from tactile_ssl.data.xela.preprocessing import compute_cached_xela_normalization

    report.section(f"Compare Object Task State: {name}")
    num_classes = max(spec.object_label for spec in specs) + 1
    left_splits = split_datasets(specs, left)
    right_splits = split_datasets(specs, right)
    for split in ["train", "val", "test"]:
        report.kv(f"{split}.left_count", len(left_splits[split]))
        report.kv(f"{split}.right_count", len(right_splits[split]))
        if len(left_splits[split]) != len(right_splits[split]):
            report.fail(f"{name}.{split}.count", f"{len(left_splits[split])} != {len(right_splits[split])}")

    left_sizes = object_class_sizes(left_splits["train"], num_classes)
    right_sizes = object_class_sizes(right_splits["train"], num_classes)
    compare_arrays(f"{name}.object_class_sizes", left_sizes, right_sizes, report)

    left_ratios = left_sizes / np.sum(left_sizes)
    right_ratios = right_sizes / np.sum(right_sizes)
    compare_arrays(f"{name}.object_class_ratios", left_ratios, right_ratios, report)

    left_weights = 1 / left_ratios
    left_weights = left_weights / np.sum(left_weights)
    right_weights = 1 / right_ratios
    right_weights = right_weights / np.sum(right_weights)
    compare_arrays(f"{name}.object_class_weights", left_weights, right_weights, report)

    left_mean, left_std = compute_cached_xela_normalization(left_splits["train"])
    right_mean, right_std = compute_cached_xela_normalization(right_splits["train"])
    compare_arrays(f"{name}.normalization.mean", left_mean, right_mean, report)
    compare_arrays(f"{name}.normalization.std", left_std, right_std, report)


def cache_inventory(name: str, cache_root: Path, report: Report) -> None:
    report.section(f"Cache Inventory: {name}")
    if not cache_root.exists():
        report.line("cache root does not exist")
        return
    files = sorted(path for path in cache_root.rglob("*") if path.is_file())
    report.kv("cache_root", cache_root)
    report.kv("file_count", len(files))
    by_suffix: dict[str, int] = {}
    for path in files:
        by_suffix[path.suffix or "<no_suffix>"] = by_suffix.get(path.suffix or "<no_suffix>", 0) + 1
    report.kv("by_suffix", json.dumps(by_suffix, sort_keys=True))


@contextlib.contextmanager
def timed(report: Report, name: str):
    start = time.perf_counter()
    report.section(name)
    try:
        yield
    finally:
        report.kv("elapsed_sec", round(time.perf_counter() - start, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./sparsh-skin-dataset")
    parser.add_argument("--out", default="diagnostics/xela_object_parallel_cache_report.txt")
    parser.add_argument("--cache-root", default=None, help="Default: temporary cache under /tmp.")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    temp_dir = None
    if args.cache_root:
        cache_base = Path(args.cache_root).expanduser().resolve()
        cache_base.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="xela_object_parallel_cache_diag_"))
        cache_base = temp_dir / "xela_artifacts"

    report = Report(out_path, failures=[])
    try:
        report.section("Environment")
        report.kv("repo_root", REPO_ROOT)
        report.kv("python", sys.executable)
        report.kv("python_version", sys.version.replace("\n", " "))
        report.kv("platform", platform.platform())
        report.kv("data_root", data_root)
        report.kv("cache_base", cache_base)
        report.kv("num_workers", args.num_workers)
        report.kv("max_sequences", args.max_sequences)

        if not data_root.exists():
            report.fail("data_root", f"does not exist: {data_root}")
            return 2
        pretrain_base = resolve_pretrain_base(data_root)
        baseline, urdf = resolve_baseline_and_urdf(data_root)
        report.kv("pretrain_base", pretrain_base)
        if not baseline.exists():
            report.fail("baseline", f"does not exist: {baseline}")
            return 2
        if not urdf.exists():
            report.fail("urdf", f"does not exist: {urdf}")
            return 2

        specs = build_task_specs(data_root, args.max_sequences)
        report.section("Task Selection")
        report.kv("task_count", len(specs))
        for spec in specs[:80]:
            report.line(f"  {spec.split}: {spec.sequence}/{spec.episode_id} label={spec.object_label}")
        if not specs:
            report.fail("task_selection", "no object-classification task episodes found")
            return 2

        disabled_cfg = make_dataset_cfg(data_root, cache_base / "cache_disabled", cache_enabled=False)
        cold_seq_cfg = make_dataset_cfg(data_root, cache_base / "cold_sequential", cache_enabled=True)
        cold_parallel_cfg = make_dataset_cfg(data_root, cache_base / "cold_parallel", cache_enabled=True)
        warm_cfg = make_dataset_cfg(data_root, cache_base / "warm_shared", cache_enabled=True)

        with timed(report, "Load Cache Disabled Sequential"):
            uncached = load_many(disabled_cfg, specs, num_workers=0)
        with timed(report, "Load Cold Cache Sequential"):
            cold_sequential = load_many(cold_seq_cfg, specs, num_workers=0)
        with timed(report, "Load Cold Cache Parallel"):
            cold_parallel = load_many(cold_parallel_cfg, specs, num_workers=args.num_workers)
        with timed(report, "Populate Warm Cache Sequential"):
            _ = load_many(warm_cfg, specs, num_workers=0)
        with timed(report, "Load Warm Cache Sequential"):
            warm_sequential = load_many(warm_cfg, specs, num_workers=0)
        with timed(report, "Load Warm Cache Parallel"):
            warm_parallel = load_many(warm_cfg, specs, num_workers=args.num_workers)

        compare_dataset_lists("uncached_vs_cold_sequential", specs, uncached, cold_sequential, report)
        compare_dataset_lists("uncached_vs_cold_parallel", specs, uncached, cold_parallel, report)
        compare_dataset_lists("cold_sequential_vs_cold_parallel", specs, cold_sequential, cold_parallel, report)
        compare_dataset_lists("warm_sequential_vs_warm_parallel", specs, warm_sequential, warm_parallel, report)
        compare_dataset_lists("uncached_vs_warm_parallel", specs, uncached, warm_parallel, report)

        compare_object_task_state("uncached_vs_cold_parallel", specs, uncached, cold_parallel, report)
        compare_object_task_state("warm_sequential_vs_warm_parallel", specs, warm_sequential, warm_parallel, report)

        cache_inventory("cold_sequential", cache_base / "cold_sequential", report)
        cache_inventory("cold_parallel", cache_base / "cold_parallel", report)
        cache_inventory("warm_shared", cache_base / "warm_shared", report)

        report.section("Result")
        if report.failures:
            report.kv("status", "FAIL")
            report.kv("failure_count", len(report.failures))
            for failure in report.failures:
                report.line(f"  {failure}")
            return 1
        report.kv("status", "PASS")
        report.kv("report", out_path)
        report.kv("cache_preserved", bool(args.keep_cache or args.cache_root))
        return 0
    except Exception:
        report.section("Fatal Error")
        report.line(traceback.format_exc())
        return 2
    finally:
        report.close()
        if temp_dir is not None and not args.keep_cache:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

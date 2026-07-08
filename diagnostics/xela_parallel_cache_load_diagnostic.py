#!/usr/bin/env python
"""Validate parallel Xela SSL dataset construction against sequential loading.

This diagnostic targets the cache.num_workers / ThreadPoolExecutor startup path
used by train.py. It is training-free and compares:

- cache disabled baseline
- cold-cache sequential loads
- cold-cache parallel loads
- warm-cache sequential loads
- warm-cache parallel loads
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
class EpisodeSpec:
    name: str
    path: Path
    object_label: int


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


def make_cfg(cache_root: Path, cache_enabled: bool, force_recompute: bool = False):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
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
                "force_recompute": force_recompute,
                "fingerprint_mode": "stats",
                "log_hits": True,
                "num_workers": 0,
            },
            "preprocessing": {
                "outlier_min": 20000,
                "outlier_max": 60000,
                "timestamp_policy": "overlap_minmax_v1",
                "baseline_policy": "mean_over_all_baseline_frames_v1",
            },
            "features": {"use_spatial_coords": False},
        }
    )


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


def find_episode_specs(data_root: Path, max_episodes: int, ids: list[int]) -> list[EpisodeSpec]:
    base = resolve_pretrain_base(data_root)
    specs: list[EpisodeSpec] = []
    for object_label, sequence in enumerate(SEQUENCES):
        for episode_id in ids:
            path = base / sequence / str(episode_id)
            if (path / "xela/data.pkl").exists():
                specs.append(EpisodeSpec(f"{sequence}/{episode_id}", path, object_label))
                if len(specs) >= max_episodes:
                    return specs
    return specs


def load_dataset(spec: EpisodeSpec, cache_root: Path, cache_enabled: bool, baseline: Path, urdf: Path):
    from tactile_ssl.data.xela_tactile import XelaSSLDataset

    cfg = make_cfg(cache_root, cache_enabled=cache_enabled)
    return XelaSSLDataset(
        config=cfg,
        data_path=str(spec.path),
        xela_urdf_path=str(urdf),
        baseline_signal_path=str(baseline),
        object_class=spec.object_label,
        load_images=False,
    )


def load_many(
    specs: list[EpisodeSpec],
    cache_root: Path,
    cache_enabled: bool,
    baseline: Path,
    urdf: Path,
    num_workers: int,
):
    if num_workers > 0 and len(specs) > 1:
        max_workers = min(num_workers, len(specs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda spec: load_dataset(spec, cache_root, cache_enabled, baseline, urdf),
                    specs,
                )
            )
    return [load_dataset(spec, cache_root, cache_enabled, baseline, urdf) for spec in specs]


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
    report.kv("  left_dtype", left.dtype)
    report.kv("  right_dtype", right.dtype)
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
        mean_abs_diff = float(np.nanmean(abs_diff)) if abs_diff.size else 0.0
        over_tol = int(np.count_nonzero(abs_diff > (atol + rtol * np.abs(right))))
    report.kv("  allclose", allclose)
    report.kv("  max_abs_diff", max_abs_diff)
    report.kv("  mean_abs_diff", mean_abs_diff)
    report.kv("  over_tol_count", over_tol)
    if allclose:
        report.line("  RESULT: PASS")
    else:
        report.fail(name, f"not allclose, max_abs_diff={max_abs_diff}, over_tol_count={over_tol}")


def compare_dataset_lists(name: str, left: list, right: list, specs: list[EpisodeSpec], report: Report) -> None:
    report.section(f"Compare Dataset Lists: {name}")
    if len(left) != len(right):
        report.fail(name, f"dataset count mismatch: {len(left)} != {len(right)}")
        return
    for spec, left_dataset, right_dataset in zip(specs, left, right):
        report.line(f"[DATASET] {spec.name}")
        if len(left_dataset) != len(right_dataset):
            report.fail(f"{name}.{spec.name}.len", f"{len(left_dataset)} != {len(right_dataset)}")
        if left_dataset.object_label != right_dataset.object_label:
            report.fail(
                f"{name}.{spec.name}.object_label",
                f"{left_dataset.object_label} != {right_dataset.object_label}",
            )
        if set(left_dataset.artifact_keys) != set(right_dataset.artifact_keys):
            report.fail(
                f"{name}.{spec.name}.artifact_keys.keys",
                f"{set(left_dataset.artifact_keys)} != {set(right_dataset.artifact_keys)}",
            )
        for artifact_name, left_key in left_dataset.artifact_keys.items():
            right_key = right_dataset.artifact_keys.get(artifact_name)
            if left_key != right_key:
                report.fail(
                    f"{name}.{spec.name}.artifact_keys.{artifact_name}",
                    f"{left_key} != {right_key}",
                )
        for key in COMPARE_ARRAY_KEYS:
            compare_arrays(
                f"{name}.{spec.name}.{key}",
                dataset_value(left_dataset, key),
                dataset_value(right_dataset, key),
                report,
            )


def compare_normalization(name: str, left: list, right: list, report: Report) -> None:
    from tactile_ssl.data.xela.preprocessing import compute_cached_xela_normalization

    report.section(f"Compare Normalization: {name}")
    left_mean, left_std = compute_cached_xela_normalization(left)
    right_mean, right_std = compute_cached_xela_normalization(right)
    compare_arrays(f"{name}.mean", left_mean, right_mean, report)
    compare_arrays(f"{name}.std", left_std, right_std, report)


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
    for path in files[:80]:
        report.line(f"  {path.relative_to(cache_root)} size={path.stat().st_size}")
    if len(files) > 80:
        report.line(f"  ... truncated {len(files) - 80} files")


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
    parser.add_argument("--out", default="diagnostics/xela_parallel_cache_load_report.txt")
    parser.add_argument("--cache-root", default=None, help="Default: temporary cache under /tmp.")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--episode-ids", default="0,1,2,3,4,5,6,7,8,9")
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
        temp_dir = Path(tempfile.mkdtemp(prefix="xela_parallel_cache_diag_"))
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
        report.kv("max_episodes", args.max_episodes)

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

        ids = [int(item) for item in args.episode_ids.split(",") if item.strip()]
        specs = find_episode_specs(data_root, args.max_episodes, ids)
        report.section("Episode Selection")
        for spec in specs:
            report.line(f"  {spec.name}: {spec.path}")
        if not specs:
            report.fail("episode_selection", "no pretrain episodes found")
            return 2

        disabled_root = cache_base / "cache_disabled"
        cold_seq_root = cache_base / "cold_sequential"
        cold_parallel_root = cache_base / "cold_parallel"
        warm_root = cache_base / "warm_shared"

        with timed(report, "Load Cache Disabled Sequential Baseline"):
            uncached = load_many(specs, disabled_root, False, baseline, urdf, num_workers=0)

        with timed(report, "Load Cold Cache Sequential"):
            cold_sequential = load_many(specs, cold_seq_root, True, baseline, urdf, num_workers=0)

        with timed(report, "Load Cold Cache Parallel"):
            cold_parallel = load_many(specs, cold_parallel_root, True, baseline, urdf, num_workers=args.num_workers)

        with timed(report, "Populate Warm Cache Sequential"):
            _ = load_many(specs, warm_root, True, baseline, urdf, num_workers=0)

        with timed(report, "Load Warm Cache Sequential"):
            warm_sequential = load_many(specs, warm_root, True, baseline, urdf, num_workers=0)

        with timed(report, "Load Warm Cache Parallel"):
            warm_parallel = load_many(specs, warm_root, True, baseline, urdf, num_workers=args.num_workers)

        compare_dataset_lists("uncached_vs_cold_sequential", uncached, cold_sequential, specs, report)
        compare_dataset_lists("uncached_vs_cold_parallel", uncached, cold_parallel, specs, report)
        compare_dataset_lists("cold_sequential_vs_cold_parallel", cold_sequential, cold_parallel, specs, report)
        compare_dataset_lists("warm_sequential_vs_warm_parallel", warm_sequential, warm_parallel, specs, report)
        compare_dataset_lists("uncached_vs_warm_parallel", uncached, warm_parallel, specs, report)

        compare_normalization("uncached_vs_cold_parallel", uncached, cold_parallel, report)
        compare_normalization("warm_sequential_vs_warm_parallel", warm_sequential, warm_parallel, report)

        cache_inventory("cold_sequential", cold_seq_root, report)
        cache_inventory("cold_parallel", cold_parallel_root, report)
        cache_inventory("warm_shared", warm_root, report)

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

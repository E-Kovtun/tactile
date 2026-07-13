#!/usr/bin/env python
"""Diagnose Xela pretrain normalization through the real train.py data path.

This script intentionally goes through train.get_dataloaders_magnetic_based()
instead of hand-assembling episode arrays. It is meant to answer whether the
runtime pretrain path computes the same normalization as the reference code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")

    def close(self) -> None:
        self._f.close()

    def line(self, text: str = "") -> None:
        print(text)
        self._f.write(text + "\n")
        self._f.flush()

    def section(self, title: str) -> None:
        self.line()
        self.line("=" * 100)
        self.line(title)
        self.line("=" * 100)

    def kv(self, key: str, value) -> None:
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        self.line(f"{key}: {value}")


def run_cmd(args: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compare_vectors(name: str, left, right, report: Report) -> None:
    left = as_numpy(left)
    right = as_numpy(right)
    report.line(f"[COMPARE] {name}")
    report.kv("  left", left.tolist())
    report.kv("  right", right.tolist())
    report.kv("  allclose", bool(np.allclose(left, right, rtol=1e-6, atol=1e-6)))
    diff = np.abs(left - right)
    report.kv("  max_abs_diff", float(np.nanmax(diff)) if diff.size else None)
    report.kv("  mean_abs_diff", float(np.nanmean(diff)) if diff.size else None)


def unwrap_concat_dataset(dataset) -> list:
    if hasattr(dataset, "datasets"):
        return list(dataset.datasets)
    return [dataset]


def iter_dataset_arrays(datasets: Iterable) -> Iterable[np.ndarray]:
    for dataset in datasets:
        xela_array = np.asarray(dataset.xela_array)
        if xela_array.shape[-1] == 4:
            xela_array = xela_array[..., 1:]
        yield xela_array


def streaming_legacy_stats(datasets: list, report: Report) -> tuple[np.ndarray, np.ndarray, int, tuple[int, ...]]:
    """Compute legacy zero-as-NaN stats without keeping a full concatenation copy."""
    total = None
    total_sq = None
    count = None
    nan_count = 0
    total_shape = [0, None, None]

    for dataset in datasets:
        arr = np.asarray(dataset.xela_array)
        if arr.shape[-1] == 4:
            arr = arr[..., 1:]
        if arr.shape[-1] != 3 or arr.shape[-2] != 368:
            raise ValueError(f"Unexpected xela_array shape for {dataset.data_path}: {arr.shape}")

        zero_mask = arr == 0
        valid = ~zero_mask
        arr64 = arr.astype(np.float64, copy=False)

        if total is None:
            total = np.zeros((3,), dtype=np.float64)
            total_sq = np.zeros((3,), dtype=np.float64)
            count = np.zeros((3,), dtype=np.int64)
            total_shape[1] = arr.shape[-2]
            total_shape[2] = arr.shape[-1]

        total += np.where(valid, arr64, 0.0).sum(axis=(0, 1))
        total_sq += np.where(valid, arr64 * arr64, 0.0).sum(axis=(0, 1))
        count += valid.sum(axis=(0, 1))
        nan_count += int(zero_mask.sum())
        total_shape[0] += int(arr.shape[0])

    if total is None or total_sq is None or count is None:
        raise ValueError("No train datasets were loaded")

    mean = total / count
    variance = total_sq / count - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    report.kv("streaming_valid_count_per_channel", count.tolist())
    return mean, std, nan_count, tuple(int(x) for x in total_shape)


def concatenate_legacy_stats(datasets: list, max_concat_frames: int | None) -> dict | None:
    if max_concat_frames is not None and max_concat_frames <= 0:
        return None

    arrays = []
    frames = 0
    for arr in iter_dataset_arrays(datasets):
        if max_concat_frames is not None and frames >= max_concat_frames:
            break
        if max_concat_frames is not None and frames + arr.shape[0] > max_concat_frames:
            arr = arr[: max_concat_frames - frames]
        arrays.append(arr)
        frames += arr.shape[0]

    if not arrays:
        return None

    xela_array = np.concatenate(arrays, axis=0)
    xela_nan = np.where(xela_array == 0, np.nan, xela_array)
    return {
        "shape": tuple(int(x) for x in xela_array.shape),
        "nan_count": int(np.isnan(xela_nan).sum()),
        "mean": np.nanmean(xela_nan, axis=(0, 1)),
        "std": np.nanstd(xela_nan, axis=(0, 1)),
    }


def dataset_inventory(datasets: list, report: Report, limit: int) -> None:
    report.section("Train Dataset Inventory")
    report.kv("train_dataset_count", len(datasets))
    for i, dataset in enumerate(datasets[:limit]):
        arr = np.asarray(dataset.xela_array)
        arr3 = arr[..., 1:] if arr.shape[-1] == 4 else arr
        nan_ready = np.where(arr3 == 0, np.nan, arr3)
        report.line(f"[dataset {i}]")
        report.kv("  data_path", getattr(dataset, "data_path", "<missing>"))
        report.kv("  object_label", getattr(dataset, "object_label", "<missing>"))
        report.kv("  len", len(dataset))
        report.kv("  xela_array_shape", tuple(int(x) for x in arr.shape))
        report.kv("  zero_count", int((arr3 == 0).sum()))
        report.kv("  legacy_mean", np.nanmean(nan_ready, axis=(0, 1)).tolist())
        report.kv("  legacy_std", np.nanstd(nan_ready, axis=(0, 1)).tolist())
        artifact_keys = getattr(dataset, "artifact_keys", None)
        if artifact_keys is not None:
            report.kv("  artifact_keys", artifact_keys)
    if len(datasets) > limit:
        report.line(f"... truncated {len(datasets) - limit} datasets")


def compose_cfg(args: argparse.Namespace):
    import hydra
    from omegaconf import open_dict

    overrides = [args.experiment]
    override_keys = {override.split("=", 1)[0] for override in args.override if "=" in override}
    if "paths.work_dir" not in override_keys:
        overrides.append(f"paths.work_dir={REPO_ROOT}")
    if "paths.output_dir" not in override_keys:
        overrides.append(f"paths.output_dir={REPO_ROOT / 'diagnostics' / 'hydra_dummy_output'}")
    if args.data_root is not None:
        overrides.append(f"paths.data_root={args.data_root}")
    if args.cache_root is not None:
        overrides.append(f"data.cache.root={args.cache_root}")
    if args.cache_enabled is not None:
        overrides.append(f"data.cache.enabled={str(args.cache_enabled).lower()}")
    if args.force_recompute:
        overrides.append("data.cache.force_recompute=true")
    if args.cache_num_workers is not None:
        overrides.append(f"data.cache.num_workers={args.cache_num_workers}")
    overrides.extend(args.override)

    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        cfg = hydra.compose(config_name="default.yaml", overrides=overrides)
    if args.cache_root is None and "data.cache.root" not in override_keys:
        with open_dict(cfg):
            cfg.data.cache.root = str(REPO_ROOT / ".cache" / "xela_train_path_normalization")
    return cfg, overrides


def environment_section(report: Report, args: argparse.Namespace, overrides: list[str]) -> None:
    report.section("Environment")
    report.kv("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    report.kv("repo_root", REPO_ROOT)
    report.kv("cwd", Path.cwd())
    report.kv("python", sys.executable)
    report.kv("python_version", sys.version.replace("\n", " "))
    report.kv("platform", platform.platform())
    report.kv("argv", sys.argv)
    report.kv("git_head", run_cmd(["git", "rev-parse", "HEAD"]))
    report.kv("git_branch", run_cmd(["git", "branch", "--show-current"]))
    report.line("[git status --short]")
    report.line(run_cmd(["git", "status", "--short"]))
    report.kv("hydra_overrides", overrides)
    report.kv("reference_mean", args.reference_mean)
    report.kv("reference_std", args.reference_std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="+experiment=xela/dinov2")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--cache-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--cache-num-workers", type=int, default=0)
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override. Can be repeated.")
    parser.add_argument("--out", default="diagnostics/xela_train_path_normalization_report.txt")
    parser.add_argument("--inventory-limit", type=int, default=20)
    parser.add_argument(
        "--max-concat-frames",
        type=int,
        default=0,
        help="Optional exact np.nanmean/np.nanstd concat check. 0 disables; use -1 for all frames.",
    )
    parser.add_argument(
        "--reference-mean",
        nargs=3,
        type=float,
        default=[-9.90929854, 4.82080161, -76.11141836],
    )
    parser.add_argument(
        "--reference-std",
        nargs=3,
        type=float,
        default=[93.05753748, 100.04659228, 108.84810989],
    )
    parser.add_argument(
        "--reference-atol",
        type=float,
        default=2e-4,
        help="Absolute tolerance for the final reference verdict.",
    )
    args = parser.parse_args()
    if args.cache_enabled is not None:
        args.cache_enabled = args.cache_enabled == "true"
    if args.max_concat_frames < 0:
        args.max_concat_frames = None
    return args


@contextlib.contextmanager
def guarded(report: Report, title: str):
    try:
        yield
    except Exception:
        report.section(f"ERROR: {title}")
        report.line(traceback.format_exc())
        raise


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    report = Report(out_path)
    try:
        cfg, overrides = compose_cfg(args)
        environment_section(report, args, overrides)

        report.section("Resolved Runtime Config")
        report.kv("paths.data_root", cfg.paths.data_root)
        report.kv("data.dataset_list", cfg.data.dataset_list)
        report.kv("data.cache", cfg.data.get("cache", "<missing>"))
        report.kv("data.features", cfg.data.get("features", "<missing>"))
        report.kv("dataset.config", cfg.data.dataset_list[0].dataset.config)

        with guarded(report, "build train dataloaders through train.py"):
            import train as train_module

            report.section("Build Dataloaders Through train.py")
            start = time.perf_counter()
            train_dset, val_dset = train_module.get_dataloaders_magnetic_based(cfg)
            elapsed = time.perf_counter() - start
            report.kv("build_elapsed_sec", elapsed)
            report.kv("train_concat_len", len(train_dset))
            report.kv("val_concat_len", len(val_dset))
            report.kv("cfg.data.normalization.mean_after_train_path", cfg.data.normalization.mean)
            report.kv("cfg.data.normalization.std_after_train_path", cfg.data.normalization.std)

        train_datasets = unwrap_concat_dataset(train_dset)
        val_datasets = unwrap_concat_dataset(val_dset)
        report.kv("train_underlying_dataset_count", len(train_datasets))
        report.kv("val_underlying_dataset_count", len(val_datasets))

        dataset_inventory(train_datasets, report, args.inventory_limit)

        report.section("Normalization Comparisons")
        train_path_mean = np.asarray(cfg.data.normalization.mean, dtype=np.float64)
        train_path_std = np.asarray(cfg.data.normalization.std, dtype=np.float64)
        reference_mean = np.asarray(args.reference_mean, dtype=np.float64)
        reference_std = np.asarray(args.reference_std, dtype=np.float64)
        compare_vectors("train_path.mean_vs_reference", train_path_mean, reference_mean, report)
        compare_vectors("train_path.std_vs_reference", train_path_std, reference_std, report)

        from tactile_ssl.data.xela.utils import compute_xela_normalization
        from tactile_ssl.data.xela.preprocessing import compute_cached_xela_normalization, compute_xela_normalization_from_arrays

        legacy_mean, legacy_std = compute_xela_normalization(train_datasets)
        report.kv("utils.compute_xela_normalization.mean", np.asarray(legacy_mean).tolist())
        report.kv("utils.compute_xela_normalization.std", np.asarray(legacy_std).tolist())
        compare_vectors("utils.mean_vs_train_path", legacy_mean, train_path_mean, report)
        compare_vectors("utils.std_vs_train_path", legacy_std, train_path_std, report)
        compare_vectors("utils.mean_vs_reference", legacy_mean, reference_mean, report)
        compare_vectors("utils.std_vs_reference", legacy_std, reference_std, report)

        cached_mean, cached_std = compute_cached_xela_normalization(train_datasets)
        report.kv("preprocessing.compute_cached_xela_normalization.mean", np.asarray(cached_mean).tolist())
        report.kv("preprocessing.compute_cached_xela_normalization.std", np.asarray(cached_std).tolist())
        compare_vectors("cached_wrapper.mean_vs_utils", cached_mean, legacy_mean, report)
        compare_vectors("cached_wrapper.std_vs_utils", cached_std, legacy_std, report)

        array_norm = compute_xela_normalization_from_arrays(list(iter_dataset_arrays(train_datasets)))
        report.kv("preprocessing.compute_xela_normalization_from_arrays.mean", array_norm["mean"].tolist())
        report.kv("preprocessing.compute_xela_normalization_from_arrays.std", array_norm["std"].tolist())
        report.kv("preprocessing.compute_xela_normalization_from_arrays.nan_count", int(array_norm["nan_count"][0]))
        compare_vectors("arrays_wrapper.mean_vs_utils", array_norm["mean"], legacy_mean, report)
        compare_vectors("arrays_wrapper.std_vs_utils", array_norm["std"], legacy_std, report)

        streaming_mean, streaming_std, streaming_nan_count, total_shape = streaming_legacy_stats(train_datasets, report)
        report.kv("streaming.total_xela_shape", total_shape)
        report.kv("streaming.nan_count_zero_as_nan", streaming_nan_count)
        report.kv("streaming.mean", streaming_mean.tolist())
        report.kv("streaming.std", streaming_std.tolist())
        compare_vectors("streaming.mean_vs_utils", streaming_mean, legacy_mean, report)
        compare_vectors("streaming.std_vs_utils", streaming_std, legacy_std, report)
        compare_vectors("streaming.mean_vs_reference", streaming_mean, reference_mean, report)
        compare_vectors("streaming.std_vs_reference", streaming_std, reference_std, report)

        concat_stats = concatenate_legacy_stats(train_datasets, args.max_concat_frames)
        if concat_stats is not None:
            report.section("Optional Concatenation Check")
            report.kv("concat.shape", concat_stats["shape"])
            report.kv("concat.nan_count", concat_stats["nan_count"])
            report.kv("concat.mean", concat_stats["mean"].tolist())
            report.kv("concat.std", concat_stats["std"].tolist())
            compare_vectors("concat.mean_vs_utils", concat_stats["mean"], legacy_mean, report)
            compare_vectors("concat.std_vs_utils", concat_stats["std"], legacy_std, report)

        report.section("Interpretation")
        report.kv("reference_verdict_atol", args.reference_atol)
        if np.allclose(train_path_mean, reference_mean, rtol=0.0, atol=args.reference_atol) and np.allclose(
            train_path_std, reference_std, rtol=0.0, atol=args.reference_atol
        ):
            report.line("RESULT: train.py runtime path matches the reference normalization.")
        elif np.allclose(legacy_mean, reference_mean, rtol=0.0, atol=args.reference_atol) and np.allclose(
            legacy_std, reference_std, rtol=0.0, atol=args.reference_atol
        ):
            report.line("RESULT: raw dataset arrays match the reference, but train.py cached normalization differs.")
        else:
            report.line("RESULT: current train.py runtime path does not match the reference normalization.")
            report.line("Check per-dataset inventory and cache settings above to identify where the drift enters.")

        report.section("Done")
        report.kv("report", out_path)
        return 0
    finally:
        report.close()


if __name__ == "__main__":
    raise SystemExit(main())

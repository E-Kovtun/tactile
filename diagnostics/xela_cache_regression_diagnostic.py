#!/usr/bin/env python
"""Compare legacy and cached Xela preprocessing on a small diagnostic subset.

This script is intentionally read-heavy and training-free. It writes a text
report that can be shared back for debugging regressions introduced around the
Xela cache/preprocessing transition.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


@dataclass
class Report:
    path: Path

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


def run_cmd(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def summarize_array(name: str, arr: Any, report: Report, max_values: int = 6) -> None:
    arr = to_numpy(arr)
    report.line(f"[{name}]")
    report.kv("  shape", arr.shape)
    report.kv("  dtype", arr.dtype)
    if arr.size == 0:
        report.line("  empty array")
        return
    finite = np.isfinite(arr)
    report.kv("  finite", f"{int(finite.sum())}/{arr.size}")
    report.kv("  nan", int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0)
    report.kv("  zeros", int((arr == 0).sum()))
    with np.errstate(all="ignore"):
        report.kv("  min", float(np.nanmin(arr)))
        report.kv("  max", float(np.nanmax(arr)))
        report.kv("  mean", float(np.nanmean(arr)))
        report.kv("  std", float(np.nanstd(arr)))
        flat = arr.reshape(-1)
        report.kv("  first_values", flat[:max_values].tolist())


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compare_arrays(
    name: str,
    left: Any,
    right: Any,
    report: Report,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_examples: int = 8,
) -> None:
    left = to_numpy(left)
    right = to_numpy(right)
    report.line(f"[COMPARE] {name}")
    report.kv("  left_shape", left.shape)
    report.kv("  right_shape", right.shape)
    report.kv("  left_dtype", left.dtype)
    report.kv("  right_dtype", right.dtype)
    if left.shape != right.shape:
        report.line("  RESULT: SHAPE_MISMATCH")
        return
    if left.size == 0:
        report.line("  RESULT: BOTH_EMPTY")
        return

    diff = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    abs_diff = np.abs(diff)
    finite = np.isfinite(abs_diff)
    with np.errstate(all="ignore"):
        report.kv("  allclose", bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True)))
        report.kv("  max_abs_diff", float(np.nanmax(abs_diff)))
        report.kv("  mean_abs_diff", float(np.nanmean(abs_diff)))
        report.kv("  p99_abs_diff", float(np.nanpercentile(abs_diff, 99)))
        report.kv("  nonzero_diff_count", int(np.count_nonzero(abs_diff > 0)))
        report.kv("  over_tol_count", int(np.count_nonzero(abs_diff > (atol + rtol * np.abs(right)))))
        report.kv("  finite_diff", f"{int(finite.sum())}/{abs_diff.size}")

    flat = abs_diff.reshape(-1)
    if flat.size:
        top = np.argsort(flat)[-max_examples:][::-1]
        examples = []
        for flat_idx in top:
            if not np.isfinite(flat[flat_idx]) or flat[flat_idx] == 0:
                continue
            idx = np.unravel_index(int(flat_idx), abs_diff.shape)
            examples.append(
                {
                    "idx": tuple(int(x) for x in idx),
                    "left": float(np.asarray(left[idx])),
                    "right": float(np.asarray(right[idx])),
                    "abs_diff": float(abs_diff[idx]),
                }
            )
        report.kv("  largest_diffs", json.dumps(examples, ensure_ascii=False))


def stats_dict(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    with np.errstate(all="ignore"):
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
            "std": float(np.nanstd(arr)) if arr.size else None,
            "zeros": int((arr == 0).sum()) if arr.size else 0,
            "nan": int(np.isnan(arr).sum()) if arr.size and np.issubdtype(arr.dtype, np.floating) else 0,
        }


def make_cfg(
    cache_root: Path,
    cache_enabled: bool,
    force_recompute: bool,
    use_spatial_coords: bool,
    num_workers: int = 0,
):
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
            "normalize": False,
            "normal_force_contact_threshold": 0.15,
            "max_normal_force": [1.0, 1.0, 5.0],
            "cache": {
                "enabled": cache_enabled,
                "root": str(cache_root),
                "force_recompute": force_recompute,
                "fingerprint_mode": "stats",
                "log_hits": True,
                "num_workers": num_workers,
            },
            "preprocessing": {
                "outlier_min": 20000,
                "outlier_max": 60000,
                "timestamp_policy": "overlap_minmax_v1",
                "baseline_policy": "mean_over_all_baseline_frames_v1",
            },
            "features": {"use_spatial_coords": use_spatial_coords},
        }
    )


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_pretrain_episode(data_root: Path, sequence: str | None, episode_id: int | None) -> Path | None:
    base = data_root / "xela/pretraining/extracted"
    if sequence is not None and episode_id is not None:
        path = base / sequence / str(episode_id)
        return path if path.exists() else None
    candidates = []
    for seq in ([sequence] if sequence else SEQUENCES):
        if seq is None:
            continue
        ids = [episode_id] if episode_id is not None else range(10)
        for idx in ids:
            candidates.append(base / seq / str(idx))
    return find_first_existing(candidates)


def find_downstream_episode(data_root: Path, task_dir: str, stage: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = data_root / path
        return path if path.exists() else None
    stage_dir = data_root / "downstream_tasks" / task_dir / stage
    if not stage_dir.exists():
        return None
    dirs = sorted([p for p in stage_dir.iterdir() if p.is_dir()])
    for path in dirs:
        if (path / "xela/data.pkl").exists():
            return path
    for path in dirs:
        nested_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
        for nested_path in nested_dirs:
            if (nested_path / "xela/data.pkl").exists():
                return nested_path
    return dirs[0] if dirs else None


def legacy_sensor_positions(joint_poses: np.ndarray) -> np.ndarray:
    import einops
    from scipy.spatial.transform import Rotation as R
    from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER, get_sensor_grid

    joint_sensor_poses = []
    for i, (key, num_sensors) in enumerate(XELA_FLATTEN_ORDER.items()):
        joint_sensor_pose = einops.repeat(joint_poses[i : i + 1], "1 t c -> t s c", s=num_sensors)
        xx, yy, d = get_sensor_grid(key)
        sensor_positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        sensor_positions = np.concatenate([sensor_positions, np.zeros_like(sensor_positions)], axis=-1)
        sensor_positions[..., -2] = d
        sensor_positions[..., -1] = 1

        transform = np.eye(4)
        transform = einops.repeat(transform, "i j -> (t s) i j", t=joint_poses.shape[1], s=num_sensors)
        sensor_pose = einops.rearrange(joint_sensor_pose, "t s c -> (t s) c")
        sensor_positions = einops.repeat(sensor_positions, "s c -> (t s) c", t=joint_poses.shape[1])
        transform[..., :3, 3] = sensor_pose[..., :3]
        transform[..., :3, :3] = R.from_quat(sensor_pose[..., 3:]).as_matrix()
        sensor_pose = np.einsum("m i j, m j -> m i", transform, sensor_positions)
        joint_sensor_poses.append(sensor_pose[..., :3].reshape(joint_poses.shape[1], num_sensors, 3))
    return np.concatenate(joint_sensor_poses, axis=1).astype(np.float32)


def legacy_pretrain_episode(
    data_path: Path,
    baseline_signal_path: Path | None,
    urdf_path: Path,
    cfg: Any,
) -> dict[str, np.ndarray]:
    import einops
    import pytorch_kinematics as pk
    from scipy.spatial.transform import Rotation as R
    from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER, compute_interp_timestamps, read_allegro_joint_data, read_xela_data

    xela_array = np.asarray(load_pickle(data_path / "xela/data.pkl"))
    allegro_path = data_path / "allegro/data.pkl"
    if allegro_path.exists():
        allegro_array = np.asarray(load_pickle(allegro_path)["joint_states"])
        timestamps, num_frames = compute_interp_timestamps([xela_array[:, 0, 0], allegro_array[:, 0]], cfg.interpolating_freq)
    else:
        allegro_array = None
        timestamps, num_frames = compute_interp_timestamps([xela_array[:, 0, 0]], cfg.interpolating_freq)

    xela_processed = read_xela_data(xela_array, timestamps, cfg.interpolating_freq, cfg.smooth_data)
    if allegro_array is not None:
        joint_angles, joint_effort = read_allegro_joint_data(allegro_array, timestamps, cfg.interpolating_freq, cfg.smooth_data)
    else:
        joint_angles = np.zeros([xela_array.shape[0], 16])
        joint_effort = np.zeros([xela_array.shape[0], 16])

    chain = pk.build_chain_from_urdf(urdf_path.read_text())
    joint_angles_torch = __import__("torch").from_numpy(joint_angles.astype(np.float32))
    joint_poses_raw = chain.forward_kinematics(joint_angles_torch)
    poses = []
    for key in XELA_FLATTEN_ORDER.keys():
        transform_matrix = joint_poses_raw[key].get_matrix()
        rotation = np.asarray(transform_matrix[:, :3, :3])
        translation = np.asarray(transform_matrix[:, :3, 3])
        quat = R.from_matrix(rotation).as_quat(canonical=True)
        poses.append(np.concatenate([translation, quat], axis=-1))
    joint_poses = np.asarray(poses, dtype=np.float32)

    xela_processed[..., 1:] = np.where(xela_processed[..., 1:] < 20000, 0, xela_processed[..., 1:])
    xela_processed[..., 1:] = np.where(xela_processed[..., 1:] > 60000, 0, xela_processed[..., 1:])

    baseline = None
    if baseline_signal_path is not None and baseline_signal_path.exists() and cfg.subtract_baseline:
        baseline_signal = np.asarray(load_pickle(baseline_signal_path))
        baseline = np.mean(baseline_signal[:, :, 1:], axis=0)
        mask = xela_processed[:, ..., 1] != 0
        baseline_t = einops.repeat(baseline, "k c -> b k c", b=xela_processed.shape[0])
        xela_processed[mask, 1:] = xela_processed[mask, 1:] - baseline_t[mask, :]

    sensor_positions = legacy_sensor_positions(joint_poses)
    return {
        "timestamps": timestamps.astype(np.float64),
        "num_frames": np.asarray(num_frames, dtype=np.int64),
        "xela_with_timestamp": xela_processed,
        "xela_array": xela_processed[..., 1:].astype(np.float32),
        "joint_angles": joint_angles.astype(np.float32),
        "joint_effort": joint_effort.astype(np.float32),
        "joint_poses": joint_poses.astype(np.float32),
        "sensor_positions": sensor_positions.astype(np.float32),
        "sensor_6ch": np.concatenate([xela_processed[..., 1:].astype(np.float32), sensor_positions], axis=-1),
        "baseline_mean": np.asarray([] if baseline is None else baseline, dtype=np.float32),
    }


def compare_pretrain(data_root: Path, args: argparse.Namespace, report: Report, cache_root: Path) -> None:
    from tactile_ssl.data.cache import ArtifactCache
    from tactile_ssl.data.xela.preprocessing import compute_xela_normalization_from_arrays, load_cached_xela_sequence

    report.section("Pretrain Xela Pipeline")
    episode = find_pretrain_episode(data_root, args.pretrain_sequence, args.pretrain_id)
    if episode is None:
        report.line("SKIP: no pretrain episode found")
        return

    baseline = data_root / "xela/pretraining/extracted/baseline/xela/data.pkl"
    urdf = data_root / "xela/pretraining/extracted/urdf/ahrcpcpn.urdf"
    report.kv("episode", episode)
    report.kv("baseline", baseline)
    report.kv("urdf", urdf)
    if not baseline.exists() or not urdf.exists():
        report.line("SKIP: missing baseline or URDF")
        return

    cfg = make_cfg(cache_root, cache_enabled=True, force_recompute=True, use_spatial_coords=True)
    legacy = legacy_pretrain_episode(episode, baseline, urdf, cfg)

    cache = ArtifactCache(root=str(cache_root), enabled=True, force_recompute=True, log_hits=True)
    cached = load_cached_xela_sequence(cache, cfg, str(episode), str(urdf), str(baseline))

    for key in ["timestamps", "xela_array", "joint_angles", "joint_effort", "joint_poses", "sensor_positions"]:
        summarize_array(f"legacy.{key}", legacy[key], report)
        summarize_array(f"cached.{key}", cached[key], report)
        compare_arrays(f"pretrain.{key}", legacy[key], cached[key], report)

    compare_arrays(
        "pretrain.sensor_6ch_legacy_vs_cached_concat",
        legacy["sensor_6ch"],
        np.concatenate([cached["xela_array"], cached["sensor_positions"]], axis=-1),
        report,
    )

    window = slice(0, min(args.max_frames, legacy["xela_array"].shape[0]))
    compare_arrays("pretrain.first_window.xela_array", legacy["xela_array"][window], cached["xela_array"][window], report)
    compare_arrays(
        "pretrain.first_window.sensor_positions",
        legacy["sensor_positions"][window],
        cached["sensor_positions"][window],
        report,
    )

    legacy_norm = normalization_legacy([legacy["xela_array"]])
    cached_norm = compute_xela_normalization_from_arrays([cached["xela_array"]])
    new_norm_on_legacy = compute_xela_normalization_from_arrays([legacy["xela_array"]])
    legacy_norm_on_cached = normalization_legacy([cached["xela_array"]])
    report.line("[normalization legacy]")
    report.kv("  mean", legacy_norm["mean"].tolist())
    report.kv("  std", legacy_norm["std"].tolist())
    report.line("[normalization cached]")
    report.kv("  mean", cached_norm["mean"].tolist())
    report.kv("  std", cached_norm["std"].tolist())
    compare_arrays("pretrain.normalization.mean", legacy_norm["mean"], cached_norm["mean"], report)
    compare_arrays("pretrain.normalization.std", legacy_norm["std"], cached_norm["std"], report)
    compare_arrays(
        "pretrain.normalization.mean.same_legacy_array.old_formula_vs_new_formula",
        legacy_norm["mean"],
        new_norm_on_legacy["mean"],
        report,
    )
    compare_arrays(
        "pretrain.normalization.std.same_legacy_array.old_formula_vs_new_formula",
        legacy_norm["std"],
        new_norm_on_legacy["std"],
        report,
    )
    compare_arrays(
        "pretrain.normalization.mean.same_cached_array.old_formula_vs_new_formula",
        legacy_norm_on_cached["mean"],
        cached_norm["mean"],
        report,
    )
    compare_arrays(
        "pretrain.normalization.std.same_cached_array.old_formula_vs_new_formula",
        legacy_norm_on_cached["std"],
        cached_norm["std"],
        report,
    )

    cached_hit = ArtifactCache(root=str(cache_root), enabled=True, force_recompute=False, log_hits=True)
    cached_again = load_cached_xela_sequence(cached_hit, cfg, str(episode), str(urdf), str(baseline))
    compare_arrays("pretrain.cache_hit.xela_array", cached["xela_array"], cached_again["xela_array"], report)
    compare_arrays("pretrain.cache_hit.sensor_positions", cached["sensor_positions"], cached_again["sensor_positions"], report)
    report.kv("cached.artifact_keys", json.dumps(cached.get("artifact_keys", {}), ensure_ascii=False, indent=2))


def normalization_legacy(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    xela_array = np.concatenate(arrays, axis=0)
    xela_array = np.where(xela_array == 0, np.nan, xela_array)
    return {
        "mean": np.nanmean(xela_array, axis=(0, 1)).astype(np.float32),
        "std": np.nanstd(xela_array, axis=(0, 1)).astype(np.float32),
    }


def compare_force(data_root: Path, args: argparse.Namespace, report: Report, cache_root: Path) -> None:
    report.section("Force Downstream Pipeline")
    episode = find_downstream_episode(data_root, "force_estimation", args.force_stage, args.force_episode)
    if episode is None:
        report.line("SKIP: no force episode found")
        return

    import tactile_ssl.data.xela_force as xela_force_module

    ForceDataset = xela_force_module.ForceDataset
    has_episode_helpers = hasattr(xela_force_module, "_load_cached_force_episode") and hasattr(
        xela_force_module, "_load_force_episode_uncached"
    )

    baseline = data_root / "downstream_tasks/force_estimation/base_line_fremont_hand/xela/data.pkl"
    urdf = data_root / "xela/pretraining/extracted/urdf/ahrcpcpn.urdf"
    report.kv("episode", episode)
    report.kv("baseline", baseline)
    report.kv("urdf", urdf)
    if not baseline.exists() or not urdf.exists():
        report.line("SKIP: missing baseline or URDF")
        return

    cfg = make_cfg(cache_root, cache_enabled=True, force_recompute=True, use_spatial_coords=True)
    params = {
        "normal_force_contact_threshold": float(cfg.normal_force_contact_threshold),
        "nominal_freq": int(cfg.interpolating_freq),
        "force_nominal_freq": int(cfg.interpolating_freq),
        "max_normal_force": list(cfg.max_normal_force),
        "subtract_baseline": bool(cfg.subtract_baseline),
        "use_spatial_coords": bool(cfg.features.use_spatial_coords),
        "window_time": float(cfg.window_time),
    }
    cache_config = {
        "root": str(cache_root),
        "enabled": True,
        "force_recompute": True,
        "log_hits": True,
        "num_workers": 0,
    }

    if has_episode_helpers:
        report.line("force episode helpers: available")
        uncached = xela_force_module._load_force_episode_uncached(str(episode), str(urdf), str(baseline), params)
        cached = xela_force_module._load_cached_force_episode(
            (str(episode), str(urdf), str(baseline), params, cache_config)
        )
        for key in ["xela_array", "xela_force_array", "force_data", "timestamps", "num_frames"]:
            summarize_array(f"force.uncached.{key}", uncached[key], report)
            summarize_array(f"force.cached.{key}", cached[key], report)
            compare_arrays(f"force.{key}", uncached[key], cached[key], report)

        cache_config["force_recompute"] = False
        cached_again = xela_force_module._load_cached_force_episode(
            (str(episode), str(urdf), str(baseline), params, cache_config)
        )
        compare_arrays("force.cache_hit.xela_array", cached["xela_array"], cached_again["xela_array"], report)
        compare_arrays("force.cache_hit.force_data", cached["force_data"], cached_again["force_data"], report)
    else:
        report.line(
            "force episode helpers: not available in this checkout; falling back to ForceDataset cache on/off comparison"
        )

    cfg_uncached = make_cfg(cache_root, cache_enabled=False, force_recompute=False, use_spatial_coords=True)
    cfg_cached = make_cfg(cache_root, cache_enabled=True, force_recompute=False, use_spatial_coords=True)
    dset_uncached = ForceDataset(cfg_uncached, [episode], str(urdf), str(baseline))
    dset_cached = ForceDataset(cfg_cached, [episode], str(urdf), str(baseline))
    report.kv("force.dataset.len.uncached", len(dset_uncached))
    report.kv("force.dataset.len.cached", len(dset_cached))
    compare_arrays("force.dataset.xela_array", dset_uncached.xela_array, dset_cached.xela_array, report)
    compare_arrays("force.dataset.force_data", dset_uncached.force_data, dset_cached.force_data, report)
    compare_arrays("force.dataset.timestamps", dset_uncached.timestamps, dset_cached.timestamps, report)

    if len(dset_uncached) and len(dset_cached):
        for idx in [0, min(len(dset_uncached), len(dset_cached)) // 2, min(len(dset_uncached), len(dset_cached)) - 1]:
            if idx < 0:
                continue
            left = dset_uncached[idx]
            right = dset_cached[idx]
            for key in ["timestamp", "sensor", "sensor_force", "force"]:
                compare_arrays(f"force.sample[{idx}].{key}", left[key], right[key], report)


def compare_relative_pose(data_root: Path, args: argparse.Namespace, report: Report, cache_root: Path) -> None:
    report.section("Relative Pose Downstream Pipeline")
    episode = find_downstream_episode(data_root, "relative_pose_estimation", args.relative_pose_stage, args.relative_pose_episode)
    if episode is None:
        report.line("SKIP: no relative pose episode found")
        return

    try:
        from tactile_ssl.data.xela_pose_estimation import _load_cached_relative_pose_episode, _load_relative_pose_episode_uncached
    except Exception:
        report.line("SKIP: could not import relative pose helpers")
        report.line(traceback.format_exc())
        return

    baseline = data_root / "downstream_tasks/relative_pose_estimation/base_line_fremont_hand/xela/data.pkl"
    if not baseline.exists():
        baseline = data_root / "xela/pretraining/extracted/baseline/xela/data.pkl"
    urdf = data_root / "xela/pretraining/extracted/urdf/ahrcpcpn.urdf"
    report.kv("episode", episode)
    report.kv("baseline", baseline)
    report.kv("urdf", urdf)
    if not baseline.exists() or not urdf.exists():
        report.line("SKIP: missing baseline or URDF")
        return

    cfg = make_cfg(cache_root, cache_enabled=True, force_recompute=True, use_spatial_coords=True)
    params = {
        "nominal_freq": int(cfg.interpolating_freq),
        "pose_nominal_freq": int(cfg.interpolating_freq) // 10,
        "subtract_baseline": bool(cfg.subtract_baseline),
        "use_spatial_coords": bool(cfg.features.use_spatial_coords),
        "window_time": float(cfg.window_time),
        "use_relative_poses": False,
    }
    cache_config = {
        "root": str(cache_root),
        "enabled": True,
        "force_recompute": True,
        "log_hits": True,
        "num_workers": 0,
    }
    uncached = _load_relative_pose_episode_uncached(str(episode), str(urdf), str(baseline), params)
    cached = _load_cached_relative_pose_episode((str(episode), str(urdf), str(baseline), params, cache_config))
    for key in ["xela_array", "relative_pose_data", "relative_pose_planar", "timestamps", "num_frames"]:
        summarize_array(f"relative_pose.uncached.{key}", uncached[key], report)
        summarize_array(f"relative_pose.cached.{key}", cached[key], report)
        compare_arrays(f"relative_pose.{key}", uncached[key], cached[key], report)


def cache_inventory(cache_root: Path, report: Report) -> None:
    report.section("Cache Inventory")
    if not cache_root.exists():
        report.line("cache root does not exist")
        return
    files = sorted([p for p in cache_root.rglob("*") if p.is_file()])
    report.kv("cache_root", cache_root)
    report.kv("file_count", len(files))
    for path in files[:200]:
        rel = path.relative_to(cache_root)
        report.line(f"  {rel} size={path.stat().st_size}")
    if len(files) > 200:
        report.line(f"  ... truncated {len(files) - 200} files")


def environment_section(report: Report, args: argparse.Namespace, cache_root: Path) -> None:
    report.section("Environment")
    report.kv("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    report.kv("repo_root", REPO_ROOT)
    report.kv("cwd", Path.cwd())
    report.kv("python", sys.executable)
    report.kv("python_version", sys.version.replace("\n", " "))
    report.kv("platform", platform.platform())
    report.kv("argv", " ".join(sys.argv))
    report.kv("data_root", args.data_root)
    report.kv("cache_root", cache_root)
    report.kv("git_head", run_cmd(["git", "rev-parse", "HEAD"]))
    report.kv("git_branch", run_cmd(["git", "branch", "--show-current"]))
    report.line("[git status --short]")
    report.line(run_cmd(["git", "status", "--short"]))
    for module in ["numpy", "torch", "omegaconf", "pytorch_kinematics", "scipy"]:
        try:
            mod = __import__(module)
            report.kv(f"version.{module}", getattr(mod, "__version__", "<no __version__>"))
        except Exception as exc:
            report.kv(f"version.{module}", f"<import failed: {exc}>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./sparsh-skin-dataset")
    parser.add_argument("--out", default="diagnostics/xela_cache_regression_report.txt")
    parser.add_argument("--cache-root", default=None, help="Default: a temporary cache under /tmp.")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--pretrain-sequence", default=None)
    parser.add_argument("--pretrain-id", type=int, default=None)
    parser.add_argument("--force-stage", default="train")
    parser.add_argument("--force-episode", default=None)
    parser.add_argument("--relative-pose-stage", default="train")
    parser.add_argument("--relative-pose-episode", default=None)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-force", action="store_true")
    parser.add_argument("--skip-relative-pose", action="store_true")
    parser.add_argument("--max-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    temp_dir = None
    if args.cache_root:
        cache_root = Path(args.cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="xela_diag_cache_"))
        cache_root = temp_dir / "xela_artifacts"

    report = Report(out_path)
    try:
        environment_section(report, args, cache_root)
        if not data_root.exists():
            report.section("Fatal")
            report.line(f"data_root does not exist: {data_root}")
            return 2
        if not args.skip_pretrain:
            with guarded_section(report, "pretrain"):
                compare_pretrain(data_root, args, report, cache_root)
        if not args.skip_force:
            with guarded_section(report, "force"):
                compare_force(data_root, args, report, cache_root)
        if not args.skip_relative_pose:
            with guarded_section(report, "relative_pose"):
                compare_relative_pose(data_root, args, report, cache_root)
        cache_inventory(cache_root, report)
        report.section("Done")
        report.kv("report", out_path)
        report.kv("cache_preserved", bool(args.keep_cache or args.cache_root))
        return 0
    finally:
        report.close()
        if temp_dir is not None and not args.keep_cache:
            shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def guarded_section(report: Report, name: str):
    try:
        yield
    except Exception:
        report.section(f"ERROR in {name}")
        report.line(traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())

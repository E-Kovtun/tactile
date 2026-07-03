from pathlib import Path
from typing import Optional

import einops
import numpy as np
import pickle
import torch
import yaml
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

from tactile_ssl.data.cache import ArtifactCache, CacheSpec
from tactile_ssl.data.cache.fingerprint import file_fingerprint, stable_hash
from tactile_ssl.data.xela.utils import (
    XELA_FLATTEN_ORDER,
    compute_interp_timestamps,
    get_sensor_grid,
    read_allegro_joint_data,
    read_xela_data,
)


def compute_baseline_mean(baseline_signal_path: str):
    with open(baseline_signal_path, "rb") as f:
        baseline_signal = np.asarray(pickle.load(f))
    return {"baseline_mean": np.mean(baseline_signal[:, :, 1:], axis=0).astype(np.float32)}


def compute_xela_interpolated(
    xela_array: np.ndarray,
    timestamps: np.ndarray,
    interpolating_freq: int,
    smooth_data: bool,
    outlier_min: float,
    outlier_max: float,
    subtract_baseline: bool,
    baseline_mean: Optional[np.ndarray],
):
    xela_processed = read_xela_data(xela_array, timestamps, interpolating_freq, smooth_data)
    xela_processed[..., 1:] = np.where(xela_processed[..., 1:] < outlier_min, 0, xela_processed[..., 1:])
    xela_processed[..., 1:] = np.where(xela_processed[..., 1:] > outlier_max, 0, xela_processed[..., 1:])

    if subtract_baseline and baseline_mean is not None:
        mask = xela_processed[..., 1] != 0
        baseline = einops.repeat(baseline_mean, "k c -> b k c", b=xela_processed.shape[0])
        xela_processed[mask, 1:] = xela_processed[mask, 1:] - baseline[mask, :]

    return {
        "timestamps": timestamps.astype(np.float64),
        "xela_array": xela_processed[..., 1:].astype(np.float32),
    }


def compute_allegro_interpolated(
    xela_array: np.ndarray,
    allegro_array: Optional[np.ndarray],
    timestamps: np.ndarray,
    interpolating_freq: int,
    smooth_data: bool,
):
    if allegro_array is not None:
        joint_angles, joint_effort = read_allegro_joint_data(
            allegro_array,
            timestamps,
            interpolating_freq,
            smooth_data,
        )
    else:
        joint_angles = np.zeros([len(timestamps), 16])
        joint_effort = np.zeros([len(timestamps), 16])

    return {
        "timestamps": timestamps.astype(np.float64),
        "joint_angles": joint_angles.astype(np.float32),
        "joint_effort": joint_effort.astype(np.float32),
    }


def compute_joint_poses(xela_kinematic_chain, joint_angles: np.ndarray):
    joint_angles_torch = torch.from_numpy(joint_angles.astype(np.float32))
    joint_poses = xela_kinematic_chain.forward_kinematics(joint_angles_torch)

    poses = []
    for key in XELA_FLATTEN_ORDER.keys():
        transform_matrix = joint_poses[key].get_matrix()
        rotation = np.asarray(transform_matrix[:, :3, :3])
        translation = np.asarray(transform_matrix[:, :3, 3])
        quat = R.from_matrix(rotation).as_quat(canonical=True)
        poses.append(np.concatenate([translation, quat], axis=-1))
    return {"joint_poses": np.asarray(poses, dtype=np.float32)}


def compute_sensor_positions(joint_poses: np.ndarray):
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

    return {"sensor_positions": np.concatenate(joint_sensor_poses, axis=1).astype(np.float32)}


def compute_xela_normalization_from_arrays(xela_arrays: list[np.ndarray], per_sensor: bool = False):
    count = None
    total = None
    total_sq = None
    nan_count = 0
    for xela_array in xela_arrays:
        assert xela_array.shape[-1] == 3, "Expected 3 channels"
        assert xela_array.shape[-2] == 368, "Expected 368 sensors"
        axis = 0 if per_sensor else (0, 1)
        valid_mask = xela_array != 0
        values = np.where(valid_mask, xela_array, 0).astype(np.float64, copy=False)
        batch_count = valid_mask.sum(axis=axis).astype(np.float64)
        batch_total = values.sum(axis=axis)
        batch_total_sq = np.square(values).sum(axis=axis)
        if count is None:
            count, total, total_sq = batch_count, batch_total, batch_total_sq
        else:
            count += batch_count
            total += batch_total
            total_sq += batch_total_sq
        nan_count += int((~valid_mask).sum())

    mean = total / count
    variance = np.maximum((total_sq / count) - np.square(mean), 0)
    return {
        "mean": mean.astype(np.float32),
        "std": np.sqrt(variance).astype(np.float32),
        "nan_count": np.asarray([nan_count], dtype=np.int64),
    }


def compute_cached_xela_normalization(xela_datasets: list, per_sensor: bool = False):
    cache = getattr(xela_datasets[0], "cache", None) if xela_datasets else None
    if cache is None:
        arrays = compute_xela_normalization_from_arrays([dataset.xela_array for dataset in xela_datasets], per_sensor)
        return arrays["mean"], arrays["std"]

    sequence_keys = {
        str(dataset.data_path): getattr(dataset, "artifact_keys", {}).get("xela_array", "missing")
        for dataset in xela_datasets
    }
    spec = CacheSpec(
        artifact="xela_normalization",
        schema_version=1,
        semantic_params={
            "per_sensor": per_sensor,
            "normalization_policy": "ignore_zero_values_v1",
            "train_sequences": list(sequence_keys.keys()),
        },
        producer_functions=(compute_xela_normalization_from_arrays,),
        upstream_keys=sequence_keys,
    )
    artifact, _ = cache.get_or_compute(
        spec,
        lambda: compute_xela_normalization_from_arrays(
            [dataset.xela_array for dataset in xela_datasets],
            per_sensor=per_sensor,
        ),
    )
    return artifact["mean"], artifact["std"]


def _load_raw_sequence(data_path: str):
    data_path = Path(data_path)
    with open(data_path / "xela" / "data.pkl", "rb") as f:
        xela_array = np.asarray(pickle.load(f))
    allegro_path = data_path / "allegro" / "data.pkl"
    if allegro_path.exists():
        with open(allegro_path, "rb") as f:
            allegro_array = np.asarray(pickle.load(f)["joint_states"])
    else:
        allegro_array = None
    return xela_array, allegro_array


def _sequence_timestamps(xela_array: np.ndarray, allegro_array: Optional[np.ndarray], interpolating_freq: int):
    if allegro_array is not None:
        timestamps, _ = compute_interp_timestamps([xela_array[:, 0, 0], allegro_array[:, 0]], interpolating_freq)
    else:
        timestamps, _ = compute_interp_timestamps([xela_array[:, 0, 0]], interpolating_freq)
    return timestamps


def write_cache_locks(cache: ArtifactCache, dataset_lock: dict, preprocessing_contract: dict):
    if not cache.enabled:
        return
    cache.root.mkdir(parents=True, exist_ok=True)
    dataset_lock_dir = cache.root / "dataset_locks"
    dataset_lock_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_lock_dir / f"{stable_hash(dataset_lock)[:24]}.yaml", "w") as f:
        yaml.safe_dump(dataset_lock, f, sort_keys=True)
    with open(cache.root / "preprocessing_contract.yaml", "w") as f:
        yaml.safe_dump(preprocessing_contract, f, sort_keys=True)


def build_preprocessing_contract(config) -> dict:
    preprocessing_cfg = config.get("preprocessing", {})
    return {
        "interpolating_freq": int(config.interpolating_freq),
        "smooth_data": bool(config.smooth_data),
        "subtract_baseline": bool(config.subtract_baseline),
        "outlier_min": float(preprocessing_cfg.get("outlier_min", 20000)),
        "outlier_max": float(preprocessing_cfg.get("outlier_max", 60000)),
        "timestamp_policy": str(preprocessing_cfg.get("timestamp_policy", "overlap_minmax_v1")),
        "baseline_policy": str(
            preprocessing_cfg.get("baseline_policy", "mean_over_all_baseline_frames_v1")
        ),
    }


def load_cached_xela_sequence(
    cache: ArtifactCache,
    config,
    data_path: str,
    xela_urdf_path: str,
    baseline_signal_path: Optional[str],
):
    data_path_obj = Path(data_path)
    xela_file = data_path_obj / "xela" / "data.pkl"
    allegro_file = data_path_obj / "allegro" / "data.pkl"
    urdf_file = Path(xela_urdf_path)
    preprocessing_contract = build_preprocessing_contract(config)
    dataset_lock = {
        "data_path": str(data_path_obj),
        "xela": file_fingerprint(str(xela_file)),
        "allegro": file_fingerprint(str(allegro_file)),
        "baseline": file_fingerprint(baseline_signal_path),
        "urdf": file_fingerprint(str(urdf_file)),
    }
    write_cache_locks(cache, dataset_lock, preprocessing_contract)

    baseline_arrays = None
    baseline_key = None
    if baseline_signal_path is not None and preprocessing_contract["subtract_baseline"]:
        baseline_spec = CacheSpec(
            artifact="baseline_mean",
            schema_version=1,
            semantic_params={
                "baseline": file_fingerprint(baseline_signal_path),
                "baseline_policy": preprocessing_contract["baseline_policy"],
            },
            producer_functions=(compute_baseline_mean,),
        )
        baseline_arrays, baseline_key = cache.get_or_compute(
            baseline_spec,
            lambda: compute_baseline_mean(baseline_signal_path),
        )

    raw_cache = {}

    def get_raw():
        if "raw" not in raw_cache:
            raw_cache["raw"] = _load_raw_sequence(data_path)
        return raw_cache["raw"]

    def get_timestamps():
        if "timestamps" not in raw_cache:
            xela_array_, allegro_array_ = get_raw()
            raw_cache["timestamps"] = _sequence_timestamps(
                xela_array_,
                allegro_array_,
                preprocessing_contract["interpolating_freq"],
            )
        return raw_cache["timestamps"]

    xela_spec = CacheSpec(
        artifact="xela_array",
        schema_version=1,
        semantic_params={
            "xela": file_fingerprint(str(xela_file)),
            "allegro": file_fingerprint(str(allegro_file)),
            "preprocessing": preprocessing_contract,
        },
        producer_functions=(compute_xela_interpolated, read_xela_data, compute_interp_timestamps),
        upstream_keys={"baseline_mean": baseline_key or "none"},
    )
    xela_artifact, xela_key = cache.get_or_compute(
        xela_spec,
        lambda: (
            lambda raw: compute_xela_interpolated(
                raw[0],
                get_timestamps(),
                preprocessing_contract["interpolating_freq"],
                preprocessing_contract["smooth_data"],
                preprocessing_contract["outlier_min"],
                preprocessing_contract["outlier_max"],
                preprocessing_contract["subtract_baseline"],
                None if baseline_arrays is None else baseline_arrays["baseline_mean"],
            )
        )(get_raw()),
    )

    allegro_spec = CacheSpec(
        artifact="allegro_interp",
        schema_version=1,
        semantic_params={
            "xela": file_fingerprint(str(xela_file)),
            "allegro": file_fingerprint(str(allegro_file)),
            "interpolating_freq": preprocessing_contract["interpolating_freq"],
            "smooth_data": preprocessing_contract["smooth_data"],
            "timestamp_policy": preprocessing_contract["timestamp_policy"],
        },
        producer_functions=(compute_allegro_interpolated, read_allegro_joint_data, compute_interp_timestamps),
    )
    allegro_artifact, allegro_key = cache.get_or_compute(
        allegro_spec,
        lambda: (
            lambda raw: compute_allegro_interpolated(
                raw[0],
                raw[1],
                get_timestamps(),
                preprocessing_contract["interpolating_freq"],
                preprocessing_contract["smooth_data"],
            )
        )(get_raw()),
    )

    joint_poses_spec = CacheSpec(
        artifact="joint_poses",
        schema_version=1,
        semantic_params={"urdf": file_fingerprint(str(urdf_file))},
        producer_functions=(compute_joint_poses,),
        producer_constants={"XELA_FLATTEN_ORDER": XELA_FLATTEN_ORDER},
        upstream_keys={"allegro_interp": allegro_key},
    )
    joint_poses_artifact, joint_poses_key = cache.get_or_compute(
        joint_poses_spec,
        lambda: compute_joint_poses(
            pk.build_chain_from_urdf(open(xela_urdf_path).read()),
            allegro_artifact["joint_angles"],
        ),
    )

    sensor_positions_spec = CacheSpec(
        artifact="sensor_positions",
        schema_version=1,
        semantic_params={"coordinate_policy": "sensor_xyz_v1"},
        producer_functions=(compute_sensor_positions, get_sensor_grid),
        producer_constants={"XELA_FLATTEN_ORDER": XELA_FLATTEN_ORDER},
        upstream_keys={"joint_poses": joint_poses_key},
    )
    sensor_positions_artifact, sensor_positions_key = cache.get_or_compute(
        sensor_positions_spec,
        lambda: compute_sensor_positions(joint_poses_artifact["joint_poses"]),
    )

    return {
        "timestamps": xela_artifact["timestamps"],
        "xela_array": xela_artifact["xela_array"],
        "joint_angles": allegro_artifact["joint_angles"],
        "joint_effort": allegro_artifact["joint_effort"],
        "joint_poses": joint_poses_artifact["joint_poses"],
        "sensor_positions": sensor_positions_artifact["sensor_positions"],
        "artifact_keys": {
            "xela_array": xela_key,
            "allegro_interp": allegro_key,
            "joint_poses": joint_poses_key,
            "sensor_positions": sensor_positions_key,
        },
        "preprocessing_contract": preprocessing_contract,
        "dataset_lock_hash": stable_hash(dataset_lock),
    }

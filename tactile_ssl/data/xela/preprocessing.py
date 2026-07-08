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
from tactile_ssl.graph.builders import build_sensor_graph


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
    total = None
    total_sq = None
    count = None
    nan_count = 0
    for xela_array in xela_arrays:
        xela_array = np.asarray(xela_array)
        assert xela_array.shape[-1] == 3, "Expected 3 channels"
        assert xela_array.shape[-2] == 368, "Expected 368 sensors"

        if per_sensor:
            out_shape = xela_array.shape[-2:]
            reduce_axis = 0
        else:
            out_shape = (xela_array.shape[-1],)
            reduce_axis = (0, 1)

        if total is None:
            total = np.zeros(out_shape, dtype=np.float64)
            total_sq = np.zeros(out_shape, dtype=np.float64)
            count = np.zeros(out_shape, dtype=np.int64)

        invalid = (xela_array == 0) | np.isnan(xela_array)
        valid = ~invalid
        xela_array = xela_array.astype(np.float64, copy=False)
        total += np.where(valid, xela_array, 0.0).sum(axis=reduce_axis)
        total_sq += np.where(valid, xela_array * xela_array, 0.0).sum(axis=reduce_axis)
        count += valid.sum(axis=reduce_axis)
        nan_count += int(invalid.sum())

    if total is None or total_sq is None or count is None:
        raise ValueError("No Xela arrays provided for normalization")

    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    variance = np.divide(total_sq, count, out=np.full_like(total_sq, np.nan), where=count > 0) - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    if per_sensor:
        assert mean.shape == (368, 3), f"Expected per-sensor normalization shape (368, 3), got {mean.shape}"

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
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
        schema_version=2,
        semantic_params={
            "per_sensor": per_sensor,
            "normalization_policy": "legacy_zero_as_nan_float64_streaming_v1",
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


def compute_window_data_idxs(num_frames: int, num_frames_per_window: int, shift_per_window: int) -> np.ndarray:
    max_length = num_frames - (num_frames % num_frames_per_window)
    max_length = max_length - num_frames_per_window
    return np.arange(0, max_length, shift_per_window, dtype=np.int64)


def _directed_edge_index_and_attr(edge_index: np.ndarray, edge_weight: np.ndarray):
    directed_edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1).astype(np.int64)
    directed_edge_attr = np.concatenate([edge_weight, edge_weight], axis=0).astype(np.float32)[:, None]
    return directed_edge_index, directed_edge_attr


def _edge_attr_for_positions(edge_index: np.ndarray, positions: np.ndarray) -> np.ndarray:
    edge_weight = np.linalg.norm(
        positions[edge_index[0]] - positions[edge_index[1]],
        axis=-1,
    ).astype(np.float32)
    return edge_weight[:, None]


def compute_window_sensor_graphs(
    sensor_positions: np.ndarray,
    window_time: float,
    interpolating_freq: int,
    window_overlap: float,
    graph_type: str,
    graph_params: dict,
    edge_attr_mode: str,
    topology_mode: str = "per_window",
):
    num_frames_per_window = int(round(window_time * interpolating_freq))
    shift_per_window = max(1, int(round(num_frames_per_window * (1.0 - window_overlap))))
    data_idxs = compute_window_data_idxs(len(sensor_positions), num_frames_per_window, shift_per_window)
    return compute_indexed_window_sensor_graphs(
        sensor_positions=sensor_positions,
        window_starts=data_idxs,
        num_frames_per_window=num_frames_per_window,
        graph_type=graph_type,
        graph_params=graph_params,
        edge_attr_mode=edge_attr_mode,
        topology_mode=topology_mode,
    )


def compute_indexed_window_sensor_graphs(
    sensor_positions: np.ndarray,
    window_starts: np.ndarray,
    num_frames_per_window: int,
    graph_type: str,
    graph_params: dict,
    edge_attr_mode: str,
    topology_mode: str = "per_window",
):
    data_idxs = np.asarray(window_starts, dtype=np.int64)
    edge_indices = []
    edge_attrs = []
    edge_counts = []
    static_edge_index = None
    if topology_mode in {"static", "static_edges", "constant"}:
        if graph_type != "physical":
            raise ValueError(f"topology_mode={topology_mode!r} is only supported for graph_type='physical'")
        if len(data_idxs) > 0:
            first_index = int(data_idxs[0])
            first_positions = sensor_positions[first_index : first_index + num_frames_per_window].mean(axis=0)
            first_graph = build_sensor_graph(first_positions, graph_type=graph_type, **graph_params)
            static_edge_index, _ = _directed_edge_index_and_attr(first_graph.edge_index, first_graph.edge_weight)

    for index in data_idxs:
        window_positions = sensor_positions[index : index + num_frames_per_window].mean(axis=0)
        if static_edge_index is None:
            graph = build_sensor_graph(window_positions, graph_type=graph_type, **graph_params)
            directed_edge_index, directed_edge_attr = _directed_edge_index_and_attr(graph.edge_index, graph.edge_weight)
        else:
            directed_edge_index = static_edge_index
            directed_edge_attr = _edge_attr_for_positions(directed_edge_index, window_positions)
        edge_indices.append(directed_edge_index)
        edge_attrs.append(directed_edge_attr)
        edge_counts.append(directed_edge_index.shape[1])

    max_edges = max(edge_counts, default=0)
    graph_edge_index = np.zeros((len(data_idxs), 2, max_edges), dtype=np.int64)
    graph_edge_attr = np.zeros((len(data_idxs), max_edges, 1), dtype=np.float32)
    graph_edge_count = np.asarray(edge_counts, dtype=np.int64)
    for i, edge_count in enumerate(edge_counts):
        graph_edge_index[i, :, :edge_count] = edge_indices[i]
        if edge_attr_mode == "distance":
            graph_edge_attr[i, :edge_count] = edge_attrs[i]

    return {
        "graph_edge_index": graph_edge_index,
        "graph_edge_attr": graph_edge_attr,
        "graph_edge_count": graph_edge_count,
        "graph_window_start": data_idxs,
    }


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

    result = {
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

    graph_cfg = config.get("graph", None)
    if graph_cfg is not None and bool(graph_cfg.get("enabled", False)):
        graph_type = str(graph_cfg.get("type", "physical"))
        if graph_type == "distance":
            graph_type = "distance_threshold"
        edge_attr_mode = str(graph_cfg.get("edge_attr_mode", "distance"))
        graph_params = dict(graph_cfg.get("params", {}))
        if graph_type == "physical" and "bridge_k" not in graph_params:
            graph_params["bridge_k"] = int(graph_cfg.get("bridge_k", 4))
        graph_spec = CacheSpec(
            artifact="window_sensor_graphs",
            schema_version=1,
            semantic_params={
                "window_time": float(config.window_time),
                "window_overlap": float(config.window_overlap),
                "interpolating_freq": int(config.interpolating_freq),
                "graph_type": graph_type,
                "graph_params": graph_params,
                "edge_attr_mode": edge_attr_mode,
                "topology_mode": str(graph_cfg.get("topology_mode", "per_window")),
            },
            producer_functions=(compute_window_sensor_graphs, build_sensor_graph),
            upstream_keys={"sensor_positions": sensor_positions_key},
        )
        graph_artifact, graph_key = cache.get_or_compute(
            graph_spec,
            lambda: compute_window_sensor_graphs(
                sensor_positions_artifact["sensor_positions"],
                window_time=float(config.window_time),
                interpolating_freq=int(config.interpolating_freq),
                window_overlap=float(config.window_overlap),
                graph_type=graph_type,
                graph_params=graph_params,
                edge_attr_mode=edge_attr_mode,
                topology_mode=str(graph_cfg.get("topology_mode", "per_window")),
            ),
        )
        result["window_sensor_graphs"] = graph_artifact
        result["artifact_keys"]["window_sensor_graphs"] = graph_key

    return result

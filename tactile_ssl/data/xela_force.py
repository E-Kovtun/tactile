from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import einops
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.xela.utils import (
    XELA_FLATTEN_ORDER,
    compute_interp_timestamps,
    read_xela_data,
    read_allegro_joint_data,
    read_force_data,
    joint_angles_to_poses,
    xela_flat_to_grid,
)
from tactile_ssl.data.cache import ArtifactCache, CacheSpec
from tactile_ssl.data.cache.fingerprint import file_fingerprint, stable_hash
from tactile_ssl.data.xela.preprocessing import compute_indexed_window_sensor_graphs
from tactile_ssl.graph.builders import build_sensor_graph

from torchvision import transforms

logger = get_pylogger(__name__)


def _ensure_force_cache_config(config: DictConfig) -> None:
    if config.get("cache") is None:
        config.cache = {}
    if config.cache.get("enabled") is None:
        config.cache.enabled = True
    if config.cache.get("root") is None:
        config.cache.root = ".cache/xela_artifacts"
    if config.cache.get("force_recompute") is None:
        config.cache.force_recompute = False
    if config.cache.get("log_hits") is None:
        config.cache.log_hits = True
    if config.cache.get("num_workers") is None:
        config.cache.num_workers = 0
    if config.cache.get("lock_timeout_s") is None:
        config.cache.lock_timeout_s = 1800.0
    if config.cache.get("stale_lock_s") is None:
        config.cache.stale_lock_s = 3600.0
    if config.cache.get("lock_log_interval_s") is None:
        config.cache.lock_log_interval_s = 30.0


def _force_cache_config(config: DictConfig) -> dict:
    return {
        "root": str(config.cache.root),
        "enabled": bool(config.cache.enabled),
        "force_recompute": bool(config.cache.force_recompute),
        "log_hits": bool(config.cache.log_hits),
        "num_workers": int(config.cache.num_workers),
        "lock_timeout_s": float(config.cache.lock_timeout_s),
        "stale_lock_s": float(config.cache.stale_lock_s),
        "lock_log_interval_s": float(config.cache.lock_log_interval_s),
    }


def _force_cache_lock_kwargs(cache_config: dict) -> dict:
    return {
        "lock_timeout_s": cache_config["lock_timeout_s"],
        "stale_lock_s": cache_config["stale_lock_s"],
        "lock_log_interval_s": cache_config["lock_log_interval_s"],
    }


def _force_params_from_config(config: DictConfig) -> dict:
    nominal_freq = int(config.interpolating_freq)
    return {
        "normal_force_contact_threshold": float(config.normal_force_contact_threshold),
        "nominal_freq": nominal_freq,
        "force_nominal_freq": nominal_freq,
        "max_normal_force": list(config.max_normal_force),
        "subtract_baseline": bool(config.subtract_baseline),
        "use_spatial_coords": bool(config.features.use_spatial_coords),
        "window_time": float(config.window_time),
    }


def _graph_params_from_config(config: DictConfig) -> Optional[dict]:
    graph_cfg = config.get("graph", None)
    if graph_cfg is None or not bool(graph_cfg.get("enabled", False)):
        return None
    graph_type = str(graph_cfg.get("type", "physical"))
    if graph_type == "distance":
        graph_type = "distance_threshold"
    graph_params = dict(graph_cfg.get("params", {}))
    if graph_type == "physical" and "bridge_k" not in graph_params:
        graph_params["bridge_k"] = int(graph_cfg.get("bridge_k", 4))
    return {
        "graph_type": graph_type,
        "graph_params": graph_params,
        "edge_attr_mode": str(graph_cfg.get("edge_attr_mode", "distance")),
        "topology_mode": str(graph_cfg.get("topology_mode", "per_window")),
        "window_frames": graph_cfg.get("window_frames", None),
    }


def _force_episode_fingerprints(data_path: Path, baseline_signal_path: Optional[str], urdf_path: str) -> dict:
    return {
        "xela": file_fingerprint(str(data_path / "xela/data.pkl")),
        "xela_force": file_fingerprint(str(data_path / "xela/forces.pkl")),
        "allegro": file_fingerprint(str(data_path / "allegro/data.pkl")),
        "force": file_fingerprint(str(data_path / "data.pkl")),
        "baseline": file_fingerprint(baseline_signal_path),
        "urdf": file_fingerprint(urdf_path),
    }


def _load_force_episode_uncached(
    data_path: str,
    urdf_path: str,
    baseline_signal_path: Optional[str],
    params: dict,
) -> dict[str, np.ndarray]:
    data_path = Path(data_path)
    xela_baseline = None
    if baseline_signal_path is not None:
        with open(baseline_signal_path, "rb") as f:
            baseline_signal = np.asarray(pickle.load(f))
        xela_baseline = np.mean(baseline_signal[:, :, 1:], axis=0)
    xela_kinematic_chain = pk.build_chain_from_urdf(open(urdf_path).read())
    xela_array, xela_force_array, force_data, timestamps, sensor_positions, num_frames = _compute_force_episode(
        data_path=data_path,
        xela_baseline=xela_baseline,
        xela_kinematic_chain=xela_kinematic_chain,
        params=params,
    )
    return {
        "xela_array": xela_array,
        "xela_force_array": xela_force_array,
        "force_data": force_data,
        "timestamps": timestamps,
        "sensor_positions": sensor_positions,
        "num_frames": np.asarray(num_frames, dtype=np.int64),
    }


def _load_cached_force_episode(args: tuple[str, str, Optional[str], dict, dict]) -> dict[str, np.ndarray]:
    data_path, urdf_path, baseline_signal_path, params, cache_config = args
    if not cache_config["enabled"]:
        return _load_force_episode_uncached(data_path, urdf_path, baseline_signal_path, params)

    data_path_obj = Path(data_path)
    cache = ArtifactCache(
        root=cache_config["root"],
        enabled=cache_config["enabled"],
        force_recompute=cache_config["force_recompute"],
        log_hits=cache_config["log_hits"],
        **_force_cache_lock_kwargs(cache_config),
    )
    spec = CacheSpec(
        artifact="xela_force_episode",
        schema_version=1,
        semantic_params={
            "data_path": str(data_path_obj),
            "files": _force_episode_fingerprints(data_path_obj, baseline_signal_path, urdf_path),
            "preprocessing": params,
        },
        producer_functions=(
            _load_force_episode_uncached,
            _compute_force_episode,
            read_xela_data,
            read_allegro_joint_data,
            read_force_data,
            joint_angles_to_poses,
            compute_interp_timestamps,
        ),
        producer_constants={"XELA_FLATTEN_ORDER": XELA_FLATTEN_ORDER},
    )
    arrays, _ = cache.get_or_compute(
        spec,
        lambda: _load_force_episode_uncached(data_path, urdf_path, baseline_signal_path, params),
        metadata={"data_path": str(data_path_obj)},
    )
    return arrays


def _force_arrays_to_tuple(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    return (
        arrays["xela_array"],
        arrays["xela_force_array"],
        arrays["force_data"],
        arrays["timestamps"],
        arrays["sensor_positions"],
        int(np.asarray(arrays["num_frames"]).item()),
    )


def _compute_force_episode(data_path: Path, xela_baseline, xela_kinematic_chain, params: dict):
    skin_pkl_path = data_path / "xela/data.pkl"
    skin_pkl_force_path = data_path / "xela/forces.pkl"
    allegro_pkl_path = data_path / "allegro/data.pkl"
    force_pkl_path = data_path / "data.pkl"

    assert skin_pkl_path.exists(), f"Xela skin data not found at {skin_pkl_path}"
    assert force_pkl_path.exists(), f"Force data not found at {force_pkl_path}"

    with open(skin_pkl_path, "rb") as f:
        skin_data = pickle.load(f)
    skin_data = np.asarray(skin_data)

    with open(skin_pkl_force_path, "rb") as f:
        skin_force_data = pickle.load(f)
    skin_force_data = np.asarray(skin_force_data)

    if not allegro_pkl_path.exists():
        # allegro is kept flat. Setting a fix allegro joint state
        allegro_joint_state = np.array(
            [
                [
                    3.76105724e-01,
                    -2.54875702e-01,
                    -2.64896910e-01,
                    -1.27969950e-01,
                    5.00173611e-02,
                    -2.60374064e-01,
                    -2.85205378e-01,
                    -2.55141751e-01,
                    -1.27792584e-01,
                    -2.41839262e-01,
                    -2.57092783e-01,
                    4.34902728e-01,
                    2.60994847e-01,
                    -3.90028996e-01,
                    1.50069820e00,
                    3.44889215e-01,
                    -8.17324002e-04,
                    -1.05262663e-03,
                    4.13282548e-03,
                    -8.95508305e-03,
                    6.96885109e-02,
                    1.30731142e-03,
                    2.55328768e-03,
                    5.62274925e-02,
                    5.00000000e-01,
                    4.12439429e-03,
                    -1.48203024e-02,
                    -5.00000000e-01,
                    5.62450390e-04,
                    2.74799718e-07,
                    -1.74512554e-02,
                    -1.01071438e-03,
                ]
            ]
        )
        allegro_data = np.repeat(allegro_joint_state, len(skin_data), axis=0)
        allegro_data = np.hstack((skin_data[:, 0, 0].reshape(-1, 1), allegro_data))
    else:
        with open(allegro_pkl_path, "rb") as f:
            allegro_data = pickle.load(f)
        allegro_data = np.array(allegro_data["joint_states"])

    with open(force_pkl_path, "rb") as f:
        force_data = pickle.load(f)
    force_data = np.array(force_data["force"])

    nominal_freq = int(params["nominal_freq"])
    force_nominal_freq = int(params["force_nominal_freq"])
    subsampling_ratio = nominal_freq // force_nominal_freq

    timestamps, num_frames = compute_interp_timestamps(
        [skin_data[:, 0, 0], allegro_data[:, 0], force_data[:, 0]], nominal_freq
    )

    # force_timestamps = timestamps[::subsampling_ratio]
    force_timestamps = timestamps

    xela_array, xela_force_array = read_xela_data(skin_data, timestamps, nominal_freq, False, skin_force_data)
    joint_angles, _ = read_allegro_joint_data(allegro_data, timestamps, nominal_freq, False)
    sensor_positions = joint_angles_to_poses(xela_kinematic_chain, joint_angles)

    mask = xela_array[:, ..., 1] != 0
    if params["subtract_baseline"] and xela_baseline is not None:
        baseline = einops.repeat(xela_baseline, "k c -> b k c", b=xela_array.shape[0])
        xela_array[mask, 1:] = xela_array[mask, 1:] - baseline[mask, :]

    xela_array = xela_array[..., 1:]
    if params["use_spatial_coords"]:
        xela_array = np.concatenate([xela_array, sensor_positions], axis=-1)

    _, gt_force_data = read_force_data(
        force_data, force_timestamps, max_abs_forceXYZ=[1.0, 1.0, 1.0], nominal_freq=force_nominal_freq
    )

    # Clip the length of the data to match the pose data
    # max_force_length = min(len(gt_force_data), len(xela_array) // 10)
    # xela_array = xela_array[: max_force_length * subsampling_ratio]
    # timestamps = timestamps[: max_force_length * subsampling_ratio]
    max_force_length = min(len(gt_force_data), len(xela_array))
    xela_array = xela_array[:max_force_length]
    if xela_force_array is not None:
        xela_force_array = xela_force_array[:max_force_length]
    sensor_positions = sensor_positions[:max_force_length]
    timestamps = timestamps[:max_force_length]
    gt_force_data = gt_force_data[:max_force_length, 1:]
    num_frames = max_force_length

    return xela_array, xela_force_array, gt_force_data, timestamps, sensor_positions, num_frames


class ForceDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        data_list: List[str],
        urdf_path: str,
        baseline_signal_path: Optional[str] = None,
    ):
        if config.get("subtract_baseline") is None:
            config.subtract_baseline = False
        if config.get("normalize") is None:
            config.normalize = False
        if config.get("features") is None:
            config.features = {}
        if config.features.get("use_spatial_coords") is None:
            config.features.use_spatial_coords = False
        _ensure_force_cache_config(config)

        self.datapath_list = data_list
        self.window_time = config.window_time
        self.target_normalize = config.normalize
        self.normal_force_contact_threshold = config.normal_force_contact_threshold
        self.nominal_freq = config.interpolating_freq
        self.target_max = config.max_normal_force
        # self.force_nominal_freq = self.nominal_freq // 10
        self.force_nominal_freq = self.nominal_freq
        self.baseline_signal_path = baseline_signal_path
        self.subtract_baseline = config.subtract_baseline
        self.use_spatial_coords = bool(config.features.use_spatial_coords)
        self.num_xela_taxels = len(XELA_FLATTEN_ORDER.keys())
        self.max_sensors_per_taxel = 30
        self.num_frames_per_window = int(round(self.window_time * self.nominal_freq))
        self.target_frames_per_window = int(round(self.window_time * self.force_nominal_freq))
        self.cache_config = _force_cache_config(config)
        self.cache_params = _force_params_from_config(config)
        self.graph_params = _graph_params_from_config(config)
        self.static_graph_edges = (
            self.graph_params is not None
            and self.graph_params["topology_mode"] in {"static", "static_edges", "constant"}
        )

        self.xela_baseline = None
        if self.baseline_signal_path is not None:
            with open(self.baseline_signal_path, "rb") as f:
                baseline_signal = np.asarray(pickle.load(f))
            self.xela_baseline = np.mean(baseline_signal[:, :, 1:], axis=0)

        self.xela_kinematic_chain = pk.build_chain_from_urdf(open(urdf_path).read())

        xela_array = []
        xela_force_array = []
        force_data = []
        timestamps = []
        sensor_positions = []
        num_frames = []

        for i, data in enumerate(self.load_episode_data(urdf_path)):
            data_path = self.datapath_list[i]
            print(f"{i}, Loading data from {data_path}")
            xela_array.append(data[0])
            xela_force_array.append(data[1])
            force_data.append(data[2])
            timestamps.append(data[3])
            sensor_positions.append(data[4])
            num_frames.append(data[5])

        self.timestamps = np.concatenate(timestamps)
        self.xela_array = np.concatenate(xela_array, axis=0)
        self.xela_force_array = np.concatenate(xela_force_array, axis=0)
        self.force_data = np.concatenate(force_data, axis=0)
        self.sensor_positions = np.concatenate(sensor_positions, axis=0)

        self.target_mean, self.target_std = self.compute_target_stats(force_data)
        print(f"Target mean: {self.target_mean}, Target std: {self.target_std}")

        print(
            f"Timestamps: {self.timestamps.shape}, Xela array: {self.xela_array.shape}, Force: {self.force_data.shape}"
        )

        self.idx_to_episode_idx = self.get_idx_to_episode_idx(force_data)
        self.window_sensor_graphs = self.load_window_sensor_graphs()

        if self.target_normalize:
            self.target_transform = transforms.Lambda(lambda x: (x - self.target_mean) / self.target_std)

    def load_episode_data(self, urdf_path):
        if self.cache_config["enabled"]:
            args = [
                (str(data_path), urdf_path, self.baseline_signal_path, self.cache_params, self.cache_config)
                for data_path in self.datapath_list
            ]
            if self.cache_config["num_workers"] > 0 and len(args) > 1:
                with ProcessPoolExecutor(max_workers=self.cache_config["num_workers"]) as pool:
                    return [_force_arrays_to_tuple(arrays) for arrays in pool.map(_load_cached_force_episode, args)]
            return [_force_arrays_to_tuple(_load_cached_force_episode(arg)) for arg in args]

        return [self.load_data(data_path) for data_path in self.datapath_list]

    def load_data(self, data_path):
        return _compute_force_episode(
            data_path=data_path,
            xela_baseline=self.xela_baseline,
            xela_kinematic_chain=self.xela_kinematic_chain,
            params=self.cache_params,
        )

    def load_window_sensor_graphs(self):
        if self.graph_params is None:
            return None
        graph_window_frames = int(self.graph_params["window_frames"] or self.num_frames_per_window)
        if self.num_frames_per_window % graph_window_frames != 0:
            raise ValueError(
                f"Graph window_frames={graph_window_frames} must divide "
                f"num_frames_per_window={self.num_frames_per_window}"
            )
        graph_chunks_per_sample = self.num_frames_per_window // graph_window_frames
        sample_window_starts = np.asarray(
            [
                item["input_episode_offset"] + item["input_offset"] - self.num_frames_per_window
                for item in self.idx_to_episode_idx
            ],
            dtype=np.int64,
        )
        window_starts = (
            sample_window_starts[:, None] + np.arange(graph_chunks_per_sample, dtype=np.int64)[None, :] * graph_window_frames
        ).reshape(-1)
        cache = ArtifactCache(
            root=self.cache_config["root"],
            enabled=self.cache_config["enabled"],
            force_recompute=self.cache_config["force_recompute"],
            log_hits=self.cache_config["log_hits"],
            **_force_cache_lock_kwargs(self.cache_config),
        )
        spec = CacheSpec(
            artifact="force_window_sensor_graphs",
            schema_version=1,
            semantic_params={
                "data_paths": [str(path) for path in self.datapath_list],
                "num_frames_per_window": int(graph_window_frames),
                "graph_chunks_per_sample": int(graph_chunks_per_sample),
                "window_starts_hash": stable_hash(window_starts.tolist()),
                **self.graph_params,
            },
            producer_functions=(compute_indexed_window_sensor_graphs, build_sensor_graph),
        )
        artifact, _ = cache.get_or_compute(
            spec,
            lambda: compute_indexed_window_sensor_graphs(
                sensor_positions=self.sensor_positions,
                window_starts=window_starts,
                num_frames_per_window=graph_window_frames,
                graph_type=self.graph_params["graph_type"],
                graph_params=self.graph_params["graph_params"],
                edge_attr_mode=self.graph_params["edge_attr_mode"],
                topology_mode=self.graph_params["topology_mode"],
            ),
        )
        artifact["graph_chunks_per_sample"] = np.asarray(graph_chunks_per_sample, dtype=np.int64)
        return artifact

    def compute_target_stats(self, force_data):
        force_data = np.concatenate(force_data, axis=0)
        target_mean = np.mean(force_data, axis=0)
        target_std = np.std(force_data, axis=0)
        return target_mean, target_std

    def get_idx_to_episode_idx(self, force_data):
        idx_to_episode_idx = []
        episode_offset = 0

        for _, target_data in enumerate(force_data):
            in_contact = np.zeros(target_data.shape[0], dtype=bool)
            in_contact[target_data[:, -1] > np.float64(self.normal_force_contact_threshold)] = True
            in_contact = savgol_filter(in_contact, 5, 3) > 0.5

            idx_to_episode_idx.extend(
                [
                    {
                        "input_episode_offset": episode_offset * self.nominal_freq // self.force_nominal_freq,
                        "episode_offset": episode_offset,
                        "input_offset": int(i * self.nominal_freq // self.force_nominal_freq),
                        "target_offset": i,
                    }
                    for i in range(self.target_frames_per_window, len(target_data))
                    if in_contact[i]
                ]
            )
            episode_offset += len(target_data)
        return idx_to_episode_idx

    @staticmethod
    def create_from_files(data_path, urdf_path, baseline_signal_path, config):
        dataset_ = []
        for stage in ["train", "val"]:
            if not (Path(data_path) / stage).exists():
                continue
            dataset_list = [p for p in (Path(data_path) / stage).iterdir() if p.is_dir()]
            dataset_.append(
                ForceDataset(
                    config=config,
                    data_list=dataset_list,
                    urdf_path=urdf_path,
                    baseline_signal_path=baseline_signal_path,
                )
            )
        assert len(dataset_) > 0, "No datasets found in the specified path."
        train_dset = dataset_[0]
        val_dset = dataset_[1] if len(dataset_) > 1 else None
        return train_dset, val_dset

    @staticmethod
    def get_single_sequence(config, data_path, urdf_path, baseline_signal_path, dataset_name):
        data_path = Path(data_path) / dataset_name
        dset = ForceDataset(
            config=config,
            data_list=[data_path],
            urdf_path=urdf_path,
            baseline_signal_path=baseline_signal_path,
        )
        return dset

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def __getitem__(self, idx):
        input_episode_offset, episode_offset, input_offset, target_offset = (
            self.idx_to_episode_idx[idx]["input_episode_offset"],
            self.idx_to_episode_idx[idx]["episode_offset"],
            self.idx_to_episode_idx[idx]["input_offset"],
            self.idx_to_episode_idx[idx]["target_offset"],
        )

        timestamp = self.timestamps[
            input_episode_offset + input_offset - self.num_frames_per_window : input_episode_offset + input_offset
        ]
        sensor_data = self.xela_array[
            input_episode_offset + input_offset - self.num_frames_per_window : input_episode_offset + input_offset
        ]
        sensor_force_data = self.xela_force_array[
            input_episode_offset + input_offset - self.num_frames_per_window : input_episode_offset + input_offset
        ]

        force_data = self.force_data[episode_offset + target_offset]

        if self.target_normalize:
            force_data = self.target_transform(force_data)

        sample = {}

        sample["timestamp"] = torch.tensor(timestamp).float()
        sample["sensor"] = torch.tensor(sensor_data).float()
        sample["sensor_force"] = torch.tensor(sensor_force_data).float()
        sample["force"] = torch.tensor(force_data).float()
        if self.window_sensor_graphs is not None:
            graph_chunks = int(np.asarray(self.window_sensor_graphs["graph_chunks_per_sample"]).item())
            graph_start = idx * graph_chunks
            graph_end = graph_start + graph_chunks
            graph = {
                "edge_attr": torch.from_numpy(self.window_sensor_graphs["graph_edge_attr"][graph_start:graph_end]).float(),
                "edge_count": torch.from_numpy(self.window_sensor_graphs["graph_edge_count"][graph_start:graph_end]).long(),
            }
            if not self.static_graph_edges:
                graph["edge_index"] = torch.from_numpy(
                    self.window_sensor_graphs["graph_edge_index"][graph_start:graph_end]
                ).long()
            sample["graph"] = graph

        return sample

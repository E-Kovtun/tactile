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
from tactile_ssl.evaluation.ids import stable_int64_id
from tactile_ssl.data.xela.utils import (
    XELA_FLATTEN_ORDER,
    compute_interp_timestamps,
    read_xela_data,
    read_allegro_joint_data,
    read_pose_data,
    joint_angles_to_poses,
    xela_flat_to_grid,
)
from tactile_ssl.data.cache import ArtifactCache, CacheSpec
from tactile_ssl.data.cache.fingerprint import file_fingerprint, stable_hash
from tactile_ssl.data.xela.preprocessing import compute_indexed_window_sensor_graphs
from tactile_ssl.graph.builders import build_sensor_graph

from torchvision import transforms

logger = get_pylogger(__name__)

USE_RELATIVE_POSES = False


def _ensure_relative_pose_cache_config(config: DictConfig) -> None:
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


def _relative_pose_cache_config(config: DictConfig) -> dict:
    return {
        "root": str(config.cache.root),
        "enabled": bool(config.cache.enabled),
        "force_recompute": bool(config.cache.force_recompute),
        "log_hits": bool(config.cache.log_hits),
        "num_workers": int(config.cache.num_workers),
    }


def _relative_pose_params_from_config(config: DictConfig) -> dict:
    nominal_freq = int(config.interpolating_freq)
    return {
        "nominal_freq": nominal_freq,
        "pose_nominal_freq": nominal_freq // 10,
        "subtract_baseline": bool(config.subtract_baseline),
        "use_spatial_coords": bool(config.features.use_spatial_coords),
        "window_time": float(config.window_time),
        "use_relative_poses": bool(USE_RELATIVE_POSES),
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


def _relative_pose_episode_fingerprints(data_path: Path, baseline_signal_path: Optional[str], urdf_path: str) -> dict:
    return {
        "xela": file_fingerprint(str(data_path / "xela/data.pkl")),
        "allegro": file_fingerprint(str(data_path / "allegro/data.pkl")),
        "object_pose": file_fingerprint(str(data_path / "object_pose.pkl")),
        "baseline": file_fingerprint(baseline_signal_path),
        "urdf": file_fingerprint(urdf_path),
    }


def _load_relative_pose_episode_uncached(
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
    xela_array, relative_pose_data, relative_pose_planar, timestamps, sensor_positions, num_frames = _compute_relative_pose_episode(
        data_path=data_path,
        xela_baseline=xela_baseline,
        xela_kinematic_chain=xela_kinematic_chain,
        params=params,
    )
    return {
        "xela_array": xela_array,
        "relative_pose_data": relative_pose_data,
        "relative_pose_planar": relative_pose_planar,
        "timestamps": timestamps,
        "sensor_positions": sensor_positions,
        "num_frames": np.asarray(num_frames, dtype=np.int64),
    }


def _load_cached_relative_pose_episode(args: tuple[str, str, Optional[str], dict, dict]) -> dict[str, np.ndarray]:
    data_path, urdf_path, baseline_signal_path, params, cache_config = args
    if not cache_config["enabled"]:
        return _load_relative_pose_episode_uncached(data_path, urdf_path, baseline_signal_path, params)

    data_path_obj = Path(data_path)
    cache = ArtifactCache(
        root=cache_config["root"],
        enabled=cache_config["enabled"],
        force_recompute=cache_config["force_recompute"],
        log_hits=cache_config["log_hits"],
    )
    spec = CacheSpec(
        artifact="xela_relative_pose_episode",
        schema_version=1,
        semantic_params={
            "data_path": str(data_path_obj),
            "files": _relative_pose_episode_fingerprints(data_path_obj, baseline_signal_path, urdf_path),
            "preprocessing": params,
        },
        producer_functions=(
            _load_relative_pose_episode_uncached,
            _compute_relative_pose_episode,
            read_xela_data,
            read_allegro_joint_data,
            read_pose_data,
            joint_angles_to_poses,
            compute_interp_timestamps,
        ),
        producer_constants={"XELA_FLATTEN_ORDER": XELA_FLATTEN_ORDER},
    )
    arrays, _ = cache.get_or_compute(
        spec,
        lambda: _load_relative_pose_episode_uncached(data_path, urdf_path, baseline_signal_path, params),
        metadata={"data_path": str(data_path_obj)},
    )
    return arrays


def _relative_pose_arrays_to_tuple(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    return (
        arrays["xela_array"],
        arrays["relative_pose_data"],
        arrays["relative_pose_planar"],
        arrays["timestamps"],
        arrays["sensor_positions"],
        int(np.asarray(arrays["num_frames"]).item()),
    )


def _compute_relative_pose_episode(data_path: Path, xela_baseline, xela_kinematic_chain, params: dict):
    skin_pkl_path = data_path / "xela/data.pkl"
    allegro_pkl_path = data_path / "allegro/data.pkl"
    object_pose_pkl_path = data_path / "object_pose.pkl"

    assert skin_pkl_path.exists(), f"Xela skin data not found at {skin_pkl_path}"
    assert object_pose_pkl_path.exists(), f"Object pose data not found at {object_pose_pkl_path}"

    with open(allegro_pkl_path, "rb") as f:
        allegro_data = pickle.load(f)
    allegro_data = np.array(allegro_data["joint_states"])

    with open(skin_pkl_path, "rb") as f:
        skin_data = pickle.load(f)
    skin_data = np.asarray(skin_data)

    with open(object_pose_pkl_path, "rb") as f:
        base_T_object = pickle.load(f)
    base_T_object = np.asarray(base_T_object)

    nominal_freq = int(params["nominal_freq"])
    pose_nominal_freq = int(params["pose_nominal_freq"])
    subsampling_ratio = nominal_freq // pose_nominal_freq

    timestamps, num_frames = compute_interp_timestamps(
        [skin_data[:, 0, 0], allegro_data[:, 0], base_T_object[:, 0]], nominal_freq
    )
    pose_timestamps = timestamps[::subsampling_ratio]

    xela_array = read_xela_data(skin_data, timestamps, nominal_freq, False)
    joint_angles, _ = read_allegro_joint_data(allegro_data, timestamps, nominal_freq, False)
    sensor_positions = joint_angles_to_poses(xela_kinematic_chain, joint_angles)

    mask = xela_array[:, ..., 1] != 0
    if params["subtract_baseline"] and xela_baseline is not None:
        baseline = einops.repeat(xela_baseline, "k c -> b k c", b=xela_array.shape[0])
        xela_array[mask, 1:] = xela_array[mask, 1:] - baseline[mask, :]

    xela_array = xela_array[..., 1:]
    if params["use_spatial_coords"]:
        xela_array = np.concatenate([xela_array, sensor_positions], axis=-1)

    base_T_object = read_pose_data(base_T_object, pose_timestamps, pose_nominal_freq)
    base_T_object = base_T_object[..., 1:]  # Strip the timestamps

    if params["use_relative_poses"]:
        base_T_object_1 = base_T_object[1:]
        base_T_object_0 = base_T_object[:-1]

        base_R_object_1 = Rotation.from_quat(base_T_object_1[:, 3:])
        base_R_object_0 = Rotation.from_quat(base_T_object_0[:, 3:])

        object_0_R_object_1 = base_R_object_0.inv() * base_R_object_1
        object_0_R_object_1_quat = object_0_R_object_1.as_quat(canonical=True)

        object_0_t_object_1 = object_0_R_object_1.inv().apply(base_T_object_1[:, :3] - base_T_object_0[:, :3])
    else:
        object_0_R_object_1 = Rotation.from_quat(base_T_object[:, 3:])
        object_0_R_object_1_quat = object_0_R_object_1.as_quat(canonical=True)
        object_0_t_object_1 = base_T_object[:, :3]

    relative_pose_data = np.concatenate([object_0_t_object_1, object_0_R_object_1_quat], axis=1)

    relative_pose_rot = object_0_R_object_1.as_euler("zxy", degrees=True)
    # x, y in m and theta in degrees
    relative_pose_planar = np.concatenate([object_0_t_object_1[:, :2], relative_pose_rot[:, 1:2]], axis=1)
    pose_data_smooth = savgol_filter(relative_pose_planar, 10, 3, axis=0)

    pose_R_smooth = np.eye(3)
    pose_R_smooth = einops.repeat(pose_R_smooth, "i j -> b i j", b=pose_data_smooth.shape[0])
    pose_R_smooth[:, :2, 2] = pose_data_smooth[:, :2]
    pose_R_smooth[:, :2, :2] = Rotation.from_euler("z", pose_data_smooth[:, 2], degrees=True).as_matrix()[:, :2, :2]

    # Clip the length of the data to match the pose data
    max_pose_length = min(len(pose_data_smooth), len(xela_array) // 10)
    xela_array = xela_array[: max_pose_length * subsampling_ratio]
    sensor_positions = sensor_positions[: max_pose_length * subsampling_ratio]
    timestamps = timestamps[: max_pose_length * subsampling_ratio]
    pose_data_smooth = pose_data_smooth[:max_pose_length]
    relative_pose_data = relative_pose_data[:max_pose_length]
    num_frames = max_pose_length * subsampling_ratio

    return xela_array, relative_pose_data, pose_data_smooth, timestamps, sensor_positions, num_frames


class RelativePoseDataset(data.Dataset):
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
        _ensure_relative_pose_cache_config(config)

        self.datapath_list = data_list
        self.window_time = config.window_time
        self.target_normalize = config.normalize
        self.nominal_freq = config.interpolating_freq
        self.pose_nominal_freq = self.nominal_freq // 10
        self.baseline_signal_path = baseline_signal_path
        self.subtract_baseline = config.subtract_baseline
        self.use_spatial_coords = bool(config.features.use_spatial_coords)
        self.num_xela_taxels = len(XELA_FLATTEN_ORDER.keys())
        self.max_sensors_per_taxel = 30
        self.num_frames_per_window = int(round(self.window_time * self.nominal_freq))
        self.target_frames_per_window = int(round(self.window_time * self.pose_nominal_freq))
        self.discretize = config.discretize
        self.cache_config = _relative_pose_cache_config(config)
        self.cache_params = _relative_pose_params_from_config(config)
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
        relative_pose_data = []
        relative_pose_planar = []
        timestamps = []
        sensor_positions = []
        num_frames = []
        for i, data in enumerate(self.load_episode_data(urdf_path)):
            data_path = self.datapath_list[i]
            print(f"{i}, Loading data from {data_path}")
            xela_array.append(data[0])
            relative_pose_data.append(data[1])
            relative_pose_planar.append(data[2])
            timestamps.append(data[3])
            sensor_positions.append(data[4])
            num_frames.append(data[5])

        self.timestamps = np.concatenate(timestamps)
        self.xela_array = np.concatenate(xela_array, axis=0)
        self.sensor_positions = np.concatenate(sensor_positions, axis=0)
        self.relative_pose_data = np.concatenate(relative_pose_data, axis=0)
        relative_pose_planar_ = np.concatenate(relative_pose_planar, axis=0)

        self.target_mean, self.target_std = self.compute_target_stats(relative_pose_planar)
        print(f"Target mean: {self.target_mean}, Target std: {self.target_std}")
        if config.discretize is not None:
            grid_size = int(config.discretize)
            self.target_normalize = False

            upper_bound = self.target_mean + 2 * self.target_std
            lower_bound = self.target_mean - 2 * self.target_std
            print(f"Upper bound: {upper_bound}, Lower bound: {lower_bound}")

            def discretize(x):
                normalized_x = (x - lower_bound) / (upper_bound - lower_bound)
                indices = (normalized_x * grid_size).astype(int)
                indices = np.clip(indices, 0, grid_size - 1)
                return indices

            relative_pose_planar_ = discretize(relative_pose_planar_)
        self.relative_pose_planar = relative_pose_planar_

        print(
            f"Timestamps: {self.timestamps.shape}, Xela array: {self.xela_array.shape}, Relative pose: {self.relative_pose_planar.shape}"
        )
        self.idx_to_episode_idx = self.get_idx_to_episode_idx(relative_pose_planar)
        group_ids = [stable_int64_id("pose-recording", Path(path)) for path in self.datapath_list]
        for item in self.idx_to_episode_idx:
            group_id = group_ids[item["episode_index"]]
            item["group_id"] = group_id
            item["sample_id"] = stable_int64_id("pose-window", group_id, item["target_offset"])
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
                    return [
                        _relative_pose_arrays_to_tuple(arrays)
                        for arrays in pool.map(_load_cached_relative_pose_episode, args)
                    ]
            return [_relative_pose_arrays_to_tuple(_load_cached_relative_pose_episode(arg)) for arg in args]

        return [self.load_data(data_path) for data_path in self.datapath_list]

    def get_idx_to_episode_idx(self, relative_pose_planar):
        idx_to_episode_idx = []
        episode_offset = 0
        for episode_index, target_data in enumerate(relative_pose_planar):
            allowed_offset = len(target_data) - self.target_frames_per_window
            idx_to_episode_idx.extend(
                [
                    {
                        "input_episode_offset": episode_offset * self.nominal_freq // self.pose_nominal_freq,
                        "episode_offset": episode_offset,
                        "input_offset": int(i * self.nominal_freq // self.pose_nominal_freq),
                        "target_offset": i,
                        "episode_index": episode_index,
                    }
                    for i in range(0, allowed_offset)
                ]
            )
            episode_offset += len(target_data)
        return idx_to_episode_idx

    def load_data(self, data_path):
        return _compute_relative_pose_episode(
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
            [item["input_episode_offset"] + item["input_offset"] for item in self.idx_to_episode_idx],
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
        )
        spec = CacheSpec(
            artifact="relative_pose_window_sensor_graphs",
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

    def compute_target_stats(self, relative_pose_planar):
        relative_pose_planar = np.concatenate(relative_pose_planar, axis=0)
        target_mean = np.mean(relative_pose_planar, axis=0)
        target_std = np.std(relative_pose_planar, axis=0)
        return target_mean, target_std

    @staticmethod
    def create_from_files(data_path, urdf_path, baseline_signal_path, config):
        dataset_ = []
        for stage in ["train", "val"]:
            object_paths = [p for p in (Path(data_path) / stage).iterdir() if p.is_dir()]
            dataset_list = []
            for object_path in object_paths:
                dataset_list.extend([p for p in object_path.iterdir() if p.is_dir()])

            data_budget = config.train_data_budget if stage == "train" else config.val_data_budget
            data_budget = round(data_budget * len(dataset_list))
            dataset_list = list(np.random.choice(dataset_list, data_budget, replace=False))

            dataset_.append(
                RelativePoseDataset(
                    config=config,
                    data_list=dataset_list,
                    urdf_path=urdf_path,
                    baseline_signal_path=baseline_signal_path,
                )
            )
        train_dset = dataset_[0]
        val_dset = dataset_[1]
        return train_dset, val_dset

    @staticmethod
    def create_train_val_test_from_files(data_path, urdf_path, baseline_signal_path, config):
        full_train_object_paths = [p for p in (Path(data_path) / "train").iterdir() if p.is_dir()]
        full_train_dataset_list = []
        for object_path in full_train_object_paths:
            full_train_dataset_list.extend([p for p in object_path.iterdir() if p.is_dir()])

        full_train_dataset_list_shuffled = np.random.permutation(full_train_dataset_list)
        full_train_len = len(full_train_dataset_list)
        num_train_files = int(full_train_len * (1 - config.val_ratio))

        train_dataset_list = full_train_dataset_list_shuffled[:num_train_files]
        val_dataset_list = full_train_dataset_list_shuffled[num_train_files:]

        train_data_budget = config.train_data_budget 
        train_data_budget = round(train_data_budget * len(train_dataset_list))
        train_dataset_list = list(np.random.choice(train_dataset_list, train_data_budget, replace=False))

        val_data_budget = config.val_data_budget
        val_data_budget = round(val_data_budget * len(val_dataset_list))
        val_dataset_list = list(np.random.choice(val_dataset_list, val_data_budget, replace=False))

        train_dset = RelativePoseDataset(
                        config=config,
                        data_list=train_dataset_list,
                        urdf_path=urdf_path,
                        baseline_signal_path=baseline_signal_path,
                    )

        val_dset = RelativePoseDataset(
                        config=config,
                        data_list=val_dataset_list,
                        urdf_path=urdf_path,
                        baseline_signal_path=baseline_signal_path,
                    )

        test_object_paths = [p for p in (Path(data_path) / "val").iterdir() if p.is_dir()]
        test_dataset_list = []
        for object_path in test_object_paths:
            test_dataset_list.extend([p for p in object_path.iterdir() if p.is_dir()])
        
        test_dset = RelativePoseDataset(
                        config=config,
                        data_list=test_dataset_list,
                        urdf_path=urdf_path,
                        baseline_signal_path=baseline_signal_path,
                    )
        return train_dset, val_dset, test_dset

    @staticmethod
    def get_single_sequence(config, data_path, urdf_path, baseline_signal_path, dataset_name):
        data_path = Path(data_path) / dataset_name
        dset = RelativePoseDataset(
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
            input_episode_offset + input_offset : input_episode_offset + input_offset + self.num_frames_per_window
        ]
        sensor_data = self.xela_array[
            input_episode_offset + input_offset : input_episode_offset + input_offset + self.num_frames_per_window
        ]
        pose_data = self.relative_pose_planar[
            episode_offset + target_offset : episode_offset + target_offset + self.target_frames_per_window
        ]

        if self.target_normalize:
            pose_data = self.target_transform(pose_data)

        sample = {}

        # if self.xela_image_output:
        #     xela_data = sensor_data[..., 0:3]
        #     tactile_image = []
        #     for i in range(0, xela_data.shape[0], 10):
        #         tactile_value = xela_flat_to_grid(xela_data[i])
        #         img = self.tactile_img.get(type="whole_hand", tactile_values=tactile_value)
        #         img = self.tactile_img_tf(img)
        #         tactile_image.append(img)
        #     tactile_image = torch.stack(tactile_image, dim=0)
        #     sample["image"] = tactile_image.float()

        sample["timestamp"] = torch.tensor(timestamp).float()
        sample["sensor"] = torch.tensor(sensor_data).float()
        if self.discretize:
            sample["relative_object_pose"] = torch.tensor(pose_data).long()
        else:
            sample["relative_object_pose"] = torch.tensor(pose_data).float()

        sample["target_mean"] = torch.tensor(self.target_mean).float()
        sample["target_std"] = torch.tensor(self.target_std).float()
        sample["group_id"] = torch.tensor(self.idx_to_episode_idx[idx]["group_id"], dtype=torch.long)
        sample["sample_id"] = torch.tensor(self.idx_to_episode_idx[idx]["sample_id"], dtype=torch.long)
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


if __name__ == "__main__":
    import hydra
    import matplotlib.pyplot as plt

    with hydra.initialize(version_base="1.3", config_path="../../config"):
        config = hydra.compose(
            config_name="experiment/xela/task/relative_pose_estimation/dinov2.yaml",
            overrides=[
                "paths=default",
                "hydra.job.num=1",
                "data.dataset.config.subtract_baseline=True",
                "data.dataset.config.normalize=True",
                "data.dataset.config.window_time=0.1",
                "data.train_dataloader.shuffle=False",
                "paths.output_dir='outputs/.'",
                "hydra.runtime.output_dir='outputs/.'",
                "paths.work_dir='outputs/.'",
            ],
            return_hydra_config=True,
        )

    print(OmegaConf.to_yaml(config, resolve=True))

    train_dset, val_dset = hydra.utils.instantiate(config.data.dataset)

    train_xela_array, val_xela_array = train_dset.xela_array, val_dset.xela_array
    train_xela_array = einops.rearrange(train_xela_array, "b k c -> (b k) c")
    val_xela_array = einops.rearrange(val_xela_array, "b k c -> (b k) c")
    print(f"xela_array.shape: {train_xela_array.shape}, {val_xela_array.shape}")

    # ax0 = plt.subplot(1, 3, 1)
    # ax0.hist(train_xela_array[:, 1], bins=100, range=(-1000, 1000))
    # ax0.set_title("X distribution")
    # ax1 = plt.subplot(1, 3, 2)
    # ax1.hist(train_xela_array[:, 2], bins=100, range=(-1000, 1000))
    # ax1.set_title("Y distribution")
    # ax2 = plt.subplot(1, 3, 3)
    # ax2.hist(train_xela_array[:, 3], bins=100, range=(-1000, 1000))
    # ax2.set_title(r"Z distribution")

    # plt.suptitle("(Train) Xela array distribution @ 100Hz")
    # plt.show()

    # ax0 = plt.subplot(1, 3, 1)
    # ax0.hist(val_xela_array[:, 1], bins=100, range=(-1000, 1000))
    # ax0.set_title("X distribution")
    # ax1 = plt.subplot(1, 3, 2)
    # ax1.hist(val_xela_array[:, 2], bins=100, range=(-1000, 1000))
    # ax1.set_title("Y distribution")
    # ax2 = plt.subplot(1, 3, 3)
    # ax2.hist(val_xela_array[:, 3], bins=100, range=(-1000, 1000))
    # ax2.set_title(r"Z distribution")

    # plt.suptitle("(Validation) Xela array distribution @ 100Hz")
    # plt.show()

    object_poses = []
    for i in range(len(train_dset)):
        data = train_dset[i]
        object_pose = data["relative_object_pose"][:, :3]
        object_poses.append(object_pose)

    object_poses = torch.cat(object_poses, dim=0)
    object_poses = object_poses.numpy()
    # ax0 = plt.subplot(3, 1, 1)
    # ax0.plot(object_poses[:, 0], label="x")
    # ax1 = plt.subplot(3, 1, 2)
    # ax1.plot(object_poses[:, 1], label="y")
    # ax2 = plt.subplot(3, 1, 3)
    # ax2.plot(object_poses[:, 2], label="z")

    # plt.show()

    ax0 = plt.subplot(1, 3, 1)
    ax0.hist(object_poses[:, 0], bins=100)
    ax0.set_title("X distribution")
    ax1 = plt.subplot(1, 3, 2)
    ax1.hist(object_poses[:, 1], bins=100)
    ax1.set_title("Y distribution")
    ax2 = plt.subplot(1, 3, 3)
    ax2.hist(object_poses[:, 2], bins=100)
    ax2.set_title(r"$\theta$ distribution")

    plt.suptitle("(Train) Relative pose distribution @ 10Hz")
    # plt.show()
    plt.savefig("train_relative_pose_distribution.png")
    plt.close()

    # object_poses = []
    # for i in range(len(val_dset)):
    #     data = val_dset[i]
    #     object_pose = data["relative_object_pose"][:, :3]
    #     object_poses.append(object_pose)

    # object_poses = torch.cat(object_poses, dim=0)
    # object_poses = object_poses.numpy()
    # ax0 = plt.subplot(3, 1, 1)
    # ax0.plot(object_poses[:, 0], label="x")
    # ax1 = plt.subplot(3, 1, 2)
    # ax1.plot(object_poses[:, 1], label="y")
    # ax2 = plt.subplot(3, 1, 3)
    # ax2.plot(object_poses[:, 2], label="z")

    # plt.show()

    # ax0 = plt.subplot(1, 3, 1)
    # ax0.hist(object_poses[:, 0], bins=100)
    # ax0.set_title("X distribution")
    # ax1 = plt.subplot(1, 3, 2)
    # ax1.hist(object_poses[:, 1], bins=100)
    # ax1.set_title("Y distribution")
    # ax2 = plt.subplot(1, 3, 3)
    # ax2.hist(object_poses[:, 2], bins=100)
    # ax2.set_title(r"$\theta$ distribution")

    # plt.suptitle("(Validation) Relative pose distribution @ 10Hz")
    # plt.show()

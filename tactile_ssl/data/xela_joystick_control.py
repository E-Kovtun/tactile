import time
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Union, Tuple
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import einops

import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
import numpy as np
import h5py
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.data.cache import ArtifactCache, CacheSpec
from tactile_ssl.data.cache.fingerprint import file_fingerprint, stable_hash
from tactile_ssl.data.xela.preprocessing import compute_indexed_window_sensor_graphs
from tactile_ssl.graph.builders import build_sensor_graph
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class XelaJoystickDataset(data.Dataset):
    """
    Dataset of a list of torch.Tensor sequences with corresponding targets that
    may themselves be sequences or individual labels
    """

    def __init__(
        self,
        config: DictConfig,
        input_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
    ) -> None:
        super().__init__()
        self.window_time = config.window_time
        # Normalize only the target data
        self.pretrain = config.get("pretrain", False)
        self.object_label = config.get("object_label", None)
        self.with_sensor_pose = config.get("with_sensor_pose", False)
        self.graph_params = self._graph_params_from_config(config)
        assert not (self.pretrain and self.object_label is None), "Must provide object label for pretraining"
        self.output_normalize = config.get("output_normalize", False)
        self.input_nominal_freq = 100
        self.target_nominal_freq = 10
        self.num_xela_sensors = 368

        self.input_frames_per_window = int(round(self.window_time * self.input_nominal_freq))
        self.target_frames_per_window = int(round(self.window_time * self.target_nominal_freq))

        self.input_lens = torch.tensor([len(x) for x in input_list])
        self.target_lens = torch.tensor([len(x) for x in target_list])

        self.get_idx_to_episode_idx(input_list, target_list)

        input_data = pad_sequence(input_list, batch_first=True)
        e, t = input_data.shape[:2]
        xela_data = input_data[..., : self.num_xela_sensors * 3]
        self.sensor_positions = None
        if self.with_sensor_pose:
            joint_data = input_data[..., self.num_xela_sensors * 3 :]
            sensor_positions = einops.rearrange(joint_data, "e t (n c) -> (e t) n c", c=3)
            self.sensor_positions = einops.rearrange(sensor_positions, "(e t) n c -> e t n c", e=e, t=t)
        xela_data = einops.rearrange(xela_data, "e t (n c) -> (e t) n c", c=3)
        self.xela_array = xela_data.detach().cpu().numpy()
        self.input_data = einops.rearrange(xela_data, "(e t) n c -> e t n c", e=e, t=t)

        target_data_unpadded = torch.cat(target_list, dim=0)
        print(
            f"torch.max(target_data_unpadded): {torch.max(target_data_unpadded[..., 2])}, {torch.min(target_data_unpadded[..., 2])}"
        )
        target_data = pad_sequence(target_list, batch_first=True)

        if config.discretize is not None:
            grid_size = int(config.discretize.num_bins)
            self.num_bins = grid_size
            upper_bound = config.discretize.upper_bound
            lower_bound = config.discretize.lower_bound
            assert lower_bound < upper_bound, "Lower bound must be less than upper bound"

            voxel_size = (upper_bound - lower_bound) / grid_size
            print(f"Voxel size: {voxel_size}, grid size: {grid_size}")

            def voxelize(data, voxel_size, grid_size):
                normalized_data = (data - (-1.0)) / 2
                voxel_indices = (normalized_data // voxel_size).long()
                voxel_indices = torch.clamp(voxel_indices, 0, grid_size - 1)
                return voxel_indices

            voxel_indices = voxelize(target_data, voxel_size, grid_size)
            indices = (
                voxel_indices[..., 0] * grid_size * grid_size
                + voxel_indices[..., 1] * grid_size
                + voxel_indices[..., 2]
            )
            self.target_data = indices
            self.output_normalize = False
        else:
            self.target_data = target_data
            self.target_weights = torch.ones(3)

        self.target_mean, self.target_std = self.compute_mean_std(target_list)
        self.transform = {"target": None}
        if self.output_normalize:
            self.transform["target"] = transforms.Lambda(lambda x: (x - self.target_mean) / self.target_std)
        self.window_sensor_graphs = self.load_window_sensor_graphs(config)

    def _graph_params_from_config(self, config: DictConfig):
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

    def _cache_config_from_config(self, config: DictConfig):
        cache_cfg = config.get("cache", {})
        return {
            "root": str(cache_cfg.get("root", ".cache/xela_artifacts")),
            "enabled": bool(cache_cfg.get("enabled", True)),
            "force_recompute": bool(cache_cfg.get("force_recompute", False)),
            "log_hits": bool(cache_cfg.get("log_hits", True)),
        }

    def load_window_sensor_graphs(self, config: DictConfig):
        if self.graph_params is None:
            return None
        if not self.with_sensor_pose or self.sensor_positions is None:
            raise ValueError("Graph cache for joystick dataset requires with_sensor_pose=True")

        graph_window_frames = int(self.graph_params["window_frames"] or self.input_frames_per_window)
        if self.input_frames_per_window % graph_window_frames != 0:
            raise ValueError(
                f"Graph window_frames={graph_window_frames} must divide "
                f"input_frames_per_window={self.input_frames_per_window}"
            )
        graph_chunks_per_sample = self.input_frames_per_window // graph_window_frames
        episode_len = self.sensor_positions.shape[1]
        flat_positions = einops.rearrange(self.sensor_positions, "e t n c -> (e t) n c").detach().cpu().numpy()
        sample_window_starts = np.asarray(
            [
                item["episode_idx"] * episode_len + item["input_offset"]
                for item in self.idx_to_episode_idx
            ],
            dtype=np.int64,
        )
        window_starts = (
            sample_window_starts[:, None] + np.arange(graph_chunks_per_sample, dtype=np.int64)[None, :] * graph_window_frames
        ).reshape(-1)
        cache_config = self._cache_config_from_config(config)
        cache = ArtifactCache(
            root=cache_config["root"],
            enabled=cache_config["enabled"],
            force_recompute=cache_config["force_recompute"],
            log_hits=cache_config["log_hits"],
        )
        spec = CacheSpec(
            artifact="joystick_window_sensor_graphs",
            schema_version=1,
            semantic_params={
                "input_lens": self.input_lens.tolist(),
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
                sensor_positions=flat_positions,
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

    def get_idx_to_episode_idx(self, input_list, target_list):
        idx_to_episode_idx = []
        for idx, target_data in enumerate(target_list):
            allowed_offset = len(target_data) - self.target_frames_per_window
            idx_to_episode_idx.extend(
                [
                    {
                        "episode_idx": idx,
                        "input_offset": int(k * self.input_nominal_freq // self.target_nominal_freq),
                        "target_offset": k,
                    }
                    for k in range(allowed_offset)
                ]
            )

        self.idx_to_episode_idx = idx_to_episode_idx

    def compute_mean_std(self, data_list):
        data = torch.cat(data_list, dim=0)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return mean, std

    def update_normalization(self, mean, std):
        self.xela_mean = mean
        self.xela_std = std

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def get_sample(self, episode_idx, input_offset, target_offset):
        xela_data = self.input_data[episode_idx][input_offset : input_offset + self.input_frames_per_window]
        if self.with_sensor_pose and self.sensor_positions is not None:
            sensor_pose_data = self.sensor_positions[episode_idx][
                input_offset : input_offset + self.input_frames_per_window
            ]
            xela_data = torch.cat([xela_data, sensor_pose_data[..., :3]], dim=-1)
        episode_target_data = self.target_data[episode_idx][
            target_offset : target_offset + self.target_frames_per_window
        ]
        return xela_data, episode_target_data

    def __getitem__(self, index):
        episode_idx, input_offset, target_offset = (
            self.idx_to_episode_idx[index]["episode_idx"],
            self.idx_to_episode_idx[index]["input_offset"],
            self.idx_to_episode_idx[index]["target_offset"],
        )
        input_data, target_data = self.get_sample(episode_idx, input_offset, target_offset)
        if self.transform["target"] is not None:
            target_data = self.transform["target"](target_data)
        sample_dict = {
            "sensor": input_data,
        }
        if self.window_sensor_graphs is not None:
            graph_chunks = int(np.asarray(self.window_sensor_graphs["graph_chunks_per_sample"]).item())
            graph_start = index * graph_chunks
            graph_end = graph_start + graph_chunks
            sample_dict["graph"] = {
                "edge_index": torch.from_numpy(self.window_sensor_graphs["graph_edge_index"][graph_start:graph_end]).long(),
                "edge_attr": torch.from_numpy(self.window_sensor_graphs["graph_edge_attr"][graph_start:graph_end]).float(),
                "edge_count": torch.from_numpy(self.window_sensor_graphs["graph_edge_count"][graph_start:graph_end]).long(),
            }
        if self.pretrain:
            sample_dict.update({"object_classification": torch.tensor(self.object_label)})
        else:
            sample_dict.update(
                {
                    "joystick_dir": target_data,
                    "len": self.target_lens[episode_idx],
                    "mean": self.target_mean,
                    "std": self.target_std,
                }
            )
        return sample_dict

    @staticmethod
    def create_from_files(
        config: DictConfig,
        data_root: str,
        file_paths: Union[str, List[str]],
    ) -> Tuple["XelaJoystickDataset", "XelaJoystickDataset"]:

        if isinstance(file_paths, str):
            file_paths = [file_paths]
        input_list = []
        target_list = []

        cache_cfg = config.get("cache", {})
        io_num_workers = int(cache_cfg.get("num_workers", 0)) if cache_cfg is not None else 0
        h5_paths = [os.path.join(data_root, file_path) for file_path in file_paths]
        if io_num_workers > 0 and len(h5_paths) > 1:
            with ProcessPoolExecutor(max_workers=io_num_workers) as pool:
                h5_data = list(
                    pool.map(
                        _load_cached_input_target_lists,
                        [(path, dict(cache_cfg or {})) for path in h5_paths],
                    )
                )
        else:
            h5_data = [_load_cached_input_target_lists((curr_path, dict(cache_cfg or {}))) for curr_path in h5_paths]

        for curr_in, curr_tgt in h5_data:
            input_list.extend(curr_in)
            target_list.extend(curr_tgt)

        num_sequences = len(input_list)
        assert num_sequences == len(target_list)

        print(f"Number of episodes: {num_sequences}")
        shuffled_seq_idxs = np.random.permutation(num_sequences)

        # train_seq_idxs = shuffled_seq_idxs[: int(0.8 * num_sequences)]
        # val_seq_idxs = shuffled_seq_idxs[int(0.8 * num_sequences) :]

        train_seq_idxs = np.load(f"{data_root}/train_seq_idxs.npy")
        val_seq_idxs = np.load(f"{data_root}/val_seq_idxs.npy")

        data_train_budget = round( config.train_data_budget * len(train_seq_idxs))
        train_seq_idxs = train_seq_idxs[:data_train_budget]

        val_seq_idxs = val_seq_idxs[:10] # REMOVE THIS LINE

        print(f"Train episodes: {len(train_seq_idxs)}")
        print(f"Val episodes: {len(val_seq_idxs)}")

        train_input_list = [input_list[i] for i in train_seq_idxs]
        train_target_list = [target_list[i] for i in train_seq_idxs]

        val_input_list = [input_list[i] for i in val_seq_idxs]
        val_target_list = [target_list[i] for i in val_seq_idxs]

        train_dataset = XelaJoystickDataset(config, train_input_list, train_target_list)
        val_dataset = XelaJoystickDataset(config, val_input_list, val_target_list)
        return train_dataset, val_dataset


def _joystick_cache_config(cache_cfg: dict) -> dict:
    return {
        "root": str(cache_cfg.get("root", ".cache/xela_artifacts")),
        "enabled": bool(cache_cfg.get("enabled", True)),
        "force_recompute": bool(cache_cfg.get("force_recompute", False)),
        "log_hits": bool(cache_cfg.get("log_hits", True)),
    }


def _load_joystick_h5_arrays(data_file: str) -> dict[str, np.ndarray]:
    with h5py.File(data_file) as obs_dict:
        return {
            "input_data": np.asarray(obs_dict["xela"], dtype=np.float32),
            "joint_data": np.asarray(obs_dict["xela_sensor_pos"], dtype=np.float32),
            "target_data": np.asarray(obs_dict["extreme3d"], dtype=np.float32)[:, :3],
            "input_episode_ids": np.asarray(obs_dict["xela_episode_ids"], dtype=np.int64),
            "target_episode_ids": np.asarray(obs_dict["extreme3d_episode_ids"], dtype=np.int64),
        }


def _joystick_arrays_to_input_target_lists(
    arrays: dict[str, np.ndarray],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    input_data = torch.from_numpy(arrays["input_data"]).float()
    joint_data = torch.from_numpy(arrays["joint_data"]).float()
    target_data = torch.from_numpy(arrays["target_data"]).float()
    input_eids = arrays["input_episode_ids"]
    target_eids = arrays["target_episode_ids"]

    input_list, target_list = [], []
    for i in range(len(input_eids) - 1):
        input_id_start, input_id_end = input_eids[i], input_eids[i + 1]
        target_id_start, target_id_end = target_eids[i], target_eids[i + 1]
        curr_input = input_data[input_id_start:input_id_end]
        curr_jointstate = joint_data[input_id_start:input_id_end]
        curr_input = torch.cat((curr_input, curr_jointstate), dim=-1)
        target_curr = target_data[target_id_start:target_id_end] - target_data[target_id_start : target_id_start + 1]

        input_list.append(curr_input)
        target_list.append(target_curr)
    return input_list, target_list


def _load_cached_input_target_lists(args: tuple[str, dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data_file, cache_cfg = args
    cache_config = _joystick_cache_config(cache_cfg)
    if not cache_config["enabled"]:
        return _joystick_arrays_to_input_target_lists(_load_joystick_h5_arrays(data_file))

    cache = ArtifactCache(
        root=cache_config["root"],
        enabled=cache_config["enabled"],
        force_recompute=cache_config["force_recompute"],
        log_hits=cache_config["log_hits"],
    )
    spec = CacheSpec(
        artifact="joystick_h5_arrays",
        schema_version=1,
        semantic_params={
            "data_file": data_file,
            "file": file_fingerprint(data_file),
        },
        producer_functions=(_load_joystick_h5_arrays, _joystick_arrays_to_input_target_lists),
    )
    arrays, _ = cache.get_or_compute(
        spec,
        lambda: _load_joystick_h5_arrays(data_file),
        metadata={"data_file": data_file},
    )
    return _joystick_arrays_to_input_target_lists(arrays)


def get_input_target_lists(
    data_file: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    return _joystick_arrays_to_input_target_lists(_load_joystick_h5_arrays(data_file))


if __name__ == "__main__":
    import random

    data_root = "/home/akashsharma/workspace/datasets/joystick_control_hiss_dataset"
    np.random.seed(0)
    random.seed(0)

    config = OmegaConf.create({"window_time": 30.0, "normalize": False, "discretize": None})
    train_dataset, val_dataset = XelaJoystickDataset.create_from_files(
        config,
        data_root,
        "joystick_control_hiss_dataset_x100e10_raw.h5",
    )
    print(len(train_dataset))
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        input = sample["sensor"]
        target = sample["joystick_dir"]

        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(target[:, 0], label="Force X", color="blue")
        ax1.set_ylabel("Force X (N)")
        ax1.legend(loc="upper right")

        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(target[:, 1], label="Force Y", color="green")
        ax2.set_ylabel("Force Y (N)")
        ax2.legend(loc="upper right")

        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(target[:, 2], label="Force Z", color="red")
        ax3.set_ylabel("Force Z (N)")

        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(target[:, 0], target[:, 1])  # , target[:, 2])
        plt.show()  # print("==" * 20)
        plt.close()

        if i > 100:
            break

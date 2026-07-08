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
                h5_data = list(pool.map(get_input_target_lists, h5_paths))
        else:
            h5_data = [get_input_target_lists(curr_path) for curr_path in h5_paths]

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


def get_input_target_lists(
    data_file: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extracts input and target lists from a .h5 file
    Args:
        file_path:
            path to .h5 file containing data
    Returns:
        Tuple of lists of input and target torch.Tensor sequences
    """
    # Extract input and target data using corresponding functions
    with h5py.File(data_file) as obs_dict:
        input_data = torch.Tensor(np.array(obs_dict["xela"]))
        joint_data = torch.Tensor(np.array(obs_dict["xela_sensor_pos"]))
        target_data = torch.Tensor(np.array(obs_dict["extreme3d"])[:, :3])
        input_episode_ids = np.array(obs_dict["xela_episode_ids"])
        target_episode_ids = np.array(obs_dict["extreme3d_episode_ids"])

    def get_data(input_eids, target_eids):
        input_list, target_list = [], []
        for i in range(len(input_eids) - 1):
            input_id_start, input_id_end = (
                input_eids[i],
                input_eids[i + 1],
            )
            target_id_start, target_id_end = (
                target_eids[i],
                target_eids[i + 1],
            )
            curr_input = input_data[input_id_start:input_id_end]
            curr_jointstate = joint_data[input_id_start:input_id_end]
            curr_input = torch.cat((curr_input, curr_jointstate), dim=-1)
            target_curr = (
                target_data[target_id_start:target_id_end] - target_data[target_id_start : target_id_start + 1]
            )

            input_list.append(curr_input)
            target_list.append(target_curr)
        return input_list, target_list

    return get_data(input_episode_ids, target_episode_ids)


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

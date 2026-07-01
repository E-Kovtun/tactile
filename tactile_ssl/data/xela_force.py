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

from torchvision import transforms

logger = get_pylogger(__name__)


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
        self.num_xela_taxels = len(XELA_FLATTEN_ORDER.keys())
        self.max_sensors_per_taxel = 30
        self.num_frames_per_window = int(round(self.window_time * self.nominal_freq))
        self.target_frames_per_window = int(round(self.window_time * self.force_nominal_freq))

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
        num_frames = []

        for i, data_path in enumerate(self.datapath_list):
            print(f"{i}, Loading data from {data_path}")
            data = self.load_data(data_path)
            xela_array.append(data[0])
            xela_force_array.append(data[1])
            force_data.append(data[2])
            timestamps.append(data[3])
            num_frames.append(data[4])

        self.timestamps = np.concatenate(timestamps)
        self.xela_array = np.concatenate(xela_array, axis=0)
        self.xela_force_array = np.concatenate(xela_force_array, axis=0)
        self.force_data = np.concatenate(force_data, axis=0)

        self.target_mean, self.target_std = self.compute_target_stats(force_data)
        print(f"Target mean: {self.target_mean}, Target std: {self.target_std}")

        print(
            f"Timestamps: {self.timestamps.shape}, Xela array: {self.xela_array.shape}, Force: {self.force_data.shape}"
        )

        self.idx_to_episode_idx = self.get_idx_to_episode_idx(force_data)

        if self.target_normalize:
            self.target_transform = transforms.Lambda(lambda x: (x - self.target_mean) / self.target_std)

    def load_data(self, data_path):
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

        subsampling_ratio = self.nominal_freq // self.force_nominal_freq

        timestamps, num_frames = compute_interp_timestamps(
            [skin_data[:, 0, 0], allegro_data[:, 0], force_data[:, 0]], self.nominal_freq
        )

        # force_timestamps = timestamps[::subsampling_ratio]
        force_timestamps = timestamps

        xela_array, xela_force_array = read_xela_data(skin_data, timestamps, self.nominal_freq, False, skin_force_data)
        joint_angles, _ = read_allegro_joint_data(allegro_data, timestamps, self.nominal_freq, False)
        sensor_positions = joint_angles_to_poses(self.xela_kinematic_chain, joint_angles)

        mask = xela_array[:, ..., 1] != 0
        if self.subtract_baseline and self.xela_baseline is not None:
            baseline = einops.repeat(self.xela_baseline, "k c -> b k c", b=xela_array.shape[0])
            xela_array[mask, 1:] = xela_array[mask, 1:] - baseline[mask, :]

        # xela_array = np.concatenate([xela_array[..., 1:], sensor_positions], axis=-1)
        xela_array = xela_array[..., 1:]

        _, gt_force_data = read_force_data(
            force_data, force_timestamps, max_abs_forceXYZ=[1.0, 1.0, 1.0], nominal_freq=self.force_nominal_freq
        )

        # Clip the length of the data to match the pose data
        # max_force_length = min(len(gt_force_data), len(xela_array) // 10)
        # xela_array = xela_array[: max_force_length * subsampling_ratio]
        # timestamps = timestamps[: max_force_length * subsampling_ratio]
        max_force_length = min(len(gt_force_data), len(xela_array))
        xela_array = xela_array[:max_force_length]
        if xela_force_array is not None:
            xela_force_array = xela_force_array[:max_force_length]
        timestamps = timestamps[:max_force_length]
        gt_force_data = gt_force_data[:max_force_length, 1:]
        num_frames = max_force_length

        return xela_array, xela_force_array, gt_force_data, timestamps, num_frames

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

        return sample

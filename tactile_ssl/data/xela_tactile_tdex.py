from typing import Optional, List
import pickle
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from omegaconf import DictConfig
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

from tactile_ssl.data.xela.utils import (
    read_xela_data,
    compute_interp_timestamps,
    load_data_dict,
    pad_xela_sample,
    xela_flat_to_grid,
    XELA_FLATTEN_ORDER,
)
from tactile_ssl.data.xela_tdex.utils import TactileImage, get_tactile_augmentations

from tactile_ssl.utils.logging import get_pylogger

torch.set_printoptions(precision=4, sci_mode=False)


log = get_pylogger(__name__)

VIS_POSES = False


class XelaBYOLDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        xela_urdf_path: str,
        baseline_signal_path: Optional[str] = None,
        object_class: Optional[int] = None,
        load_images: bool = False,
    ):
        if config.get("window_overlap") is None:
            config.window_overlap = 0.0
        if config.get("subtract_baseline") is None:
            config.subtract_baseline = False
        if config.get("smooth_data") is None:
            config.smooth_data = False
        if config.get("bias_noise_std") is None:
            config.bias_noise_std = 0.0
        if config.get("bias_range") is None:
            config.bias_range = 0.0
        if baseline_signal_path is None:
            config.subtract_baseline = False

        self.window_time = config.window_time
        assert 0 <= config.window_overlap < 1, "Window overlap should be between 0 and 1"
        self.window_overlap = config.window_overlap
        # The original baseline emits one instantaneous sample at 10 Hz.  Keep
        # the 100 Hz grid explicit so the phase within each ten-frame block is
        # reproducible instead of being implicit in a direct 10 Hz resample.
        self.interpolating_freq = int(config.get("interpolating_freq", 100))
        self.frame_stride = int(config.get("frame_stride", 10))
        self.frame_offset = int(config.get("frame_offset", 0))
        if self.frame_stride <= 0 or not 0 <= self.frame_offset < self.frame_stride:
            raise ValueError("frame_offset must be in [0, frame_stride)")
        self.tactile_img_size = 224
        shuffle_type = None
        # self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        # self.shift_per_window = int(round(self.num_frames_per_window * (1.0 - self.window_overlap)))

        self.subtract_baseline = config.subtract_baseline
        self.smooth_data = config.smooth_data
        self.load_images = load_images
        self.bias_noise_std = config.bias_noise_std
        self.bias_range = config.bias_range
        # self.augment = False if self.bias_noise_std == 0.0 and self.bias_range == 0.0 else True
        self.augment = False
        self.with_object_classes = True
        self.object_label = object_class

        # self.num_xela_taxels = len(XELA_FLATTEN_ORDER.keys())
        # self.max_sensors_per_taxel = 30

        # assert Path(xela_urdf_path).exists(), f"{xela_urdf_path} does not exist"
        # self.xela_kinematic_chain = pk.build_chain_from_urdf(open(xela_urdf_path).read())

        self.data_path = data_path
        xela_dict, allegro_dict = load_data_dict(self.data_path)
        self.baseline_signal_path = baseline_signal_path
        if self.baseline_signal_path is not None:
            with open(self.baseline_signal_path, "rb") as f:
                baseline_signal = np.asarray(pickle.load(f))
            self.xela_baseline = np.mean(baseline_signal[:, :, 1:], axis=0)
        self.xela_mean, self.xela_std = None, None

        xela_array = np.array(xela_dict, copy=True)
        full_timestamps, _ = compute_interp_timestamps(
            [xela_array[:, 0, 0]], self.interpolating_freq
        )
        
        full_xela_array = read_xela_data(
            xela_array,
            full_timestamps,
            self.interpolating_freq,
            self.smooth_data,
        )
        selected = full_xela_array[self.frame_offset :: self.frame_stride]
        self.timestamps = selected[:, 0, 0]
        # Match the current common Xela contract: xela_array contains magnetic
        # channels only; timestamps are stored separately.
        self.xela_array = selected[..., 1:]
        self.num_frames = len(self.xela_array)
        self.data_idxs = np.arange(0, self.num_frames)

        # Remove outliers
        self.xela_array = np.where(self.xela_array < 20000, 0, self.xela_array)
        self.xela_array = np.where(self.xela_array > 60000, 0, self.xela_array)

        # NOTE: There were some bad sensors during pilot pretraining data collection (Sensor IDX: 104, 145)
        mask = self.xela_array[..., 0] != 0
        if self.subtract_baseline and self.xela_baseline is not None:
            baseline = einops.repeat(self.xela_baseline, "k c -> b k c", b=self.xela_array.shape[0])
            self.xela_array[mask, :] = self.xela_array[mask, :] - baseline[mask, :]
        
        self.tactile_img = TactileImage(tactile_image_size=self.tactile_img_size, shuffle_type=shuffle_type)
        self.augmentations = get_tactile_augmentations(self.tactile_img_size)

        # sensor_imgs = xela_flat_to_grid(self.xela_array[0][:,1:])
        # img = self.tactile_img.get(type="whole_hand", tactile_values=sensor_imgs)

    def __len__(self):
        return len(self.data_idxs)

    def update_normalization(self, xela_mean, xela_std):
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def _get_tactile_image(self, tactile_values):
        return self.tactile_img.get(type="whole_hand", tactile_values=tactile_values)


    def __getitem__(self, idx):
        # num_frames_per_window = 1
        index = self.data_idxs[idx]
        # timestamp = self.timestamps[index : index + num_frames_per_window]
        sensor_data_flat = self.xela_array[index]
        tactile_value = xela_flat_to_grid(sensor_data_flat)

        tactile_image = self._get_tactile_image(tactile_value)

        # Generate BYOL views per sample in DataLoader workers. Applying the
        # torchvision transform later to a collated BCHW tensor would share a
        # single random crop/blur draw across the whole batch.
        aug_tactile_image = self.augmentations(tactile_image)
        aug_tactile_image2 = self.augmentations(tactile_image)
        sensor_data_flat = torch.from_numpy(sensor_data_flat).float()

        sample_dict =  {
            "image": tactile_image.float(),
            "aug_image": aug_tactile_image.float(),
            "aug_image2": aug_tactile_image2.float(),
            "tactile_values": sensor_data_flat,
        }
        if self.with_object_classes:
            sample_dict.update({"object_classification": torch.tensor(self.object_label)})
        return sample_dict

from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from omegaconf import DictConfig

from tactile_ssl.data.cache import ArtifactCache
from tactile_ssl.data.xela.preprocessing import load_cached_xela_sequence
from tactile_ssl.data.xela.utils import (
    XELA_FLATTEN_ORDER,
)
from tactile_ssl.evaluation.ids import stable_int64_id
from tactile_ssl.utils.logging import get_pylogger

torch.set_printoptions(precision=4, sci_mode=False)


log = get_pylogger(__name__)

VIS_POSES = False


class XelaSSLDataset(data.Dataset):
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
        if config.get("features") is None:
            config.features = {}
        if config.features.get("use_spatial_coords") is None:
            config.features.use_spatial_coords = False
        if config.get("cache") is None:
            config.cache = {}
        if config.cache.get("enabled") is None:
            config.cache.enabled = False
        if config.cache.get("root") is None:
            config.cache.root = ".cache/xela_artifacts"
        if config.cache.get("force_recompute") is None:
            config.cache.force_recompute = False
        if config.cache.get("log_hits") is None:
            config.cache.log_hits = True
        if baseline_signal_path is None:
            config.subtract_baseline = False

        self.window_time = config.window_time
        assert 0 <= config.window_overlap < 1, "Window overlap should be between 0 and 1"
        self.window_overlap = config.window_overlap
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.shift_per_window = int(round(self.num_frames_per_window * (1.0 - self.window_overlap)))
        self.shift_per_window = max(1, self.shift_per_window)

        self.subtract_baseline = config.subtract_baseline
        self.smooth_data = config.smooth_data
        self.load_images = load_images
        self.bias_noise_std = config.bias_noise_std
        self.bias_range = config.bias_range
        self.augment = False if self.bias_noise_std == 0.0 and self.bias_range == 0.0 else True
        self.object_label = object_class
        self.use_spatial_coords = bool(config.features.use_spatial_coords)

        self.num_xela_taxels = len(XELA_FLATTEN_ORDER.keys())
        self.max_sensors_per_taxel = 30

        assert Path(xela_urdf_path).exists(), f"{xela_urdf_path} does not exist"

        self.data_path = data_path
        self.evaluation_group_id = stable_int64_id("xela-recording", Path(data_path))
        self.baseline_signal_path = baseline_signal_path
        self.xela_mean, self.xela_std = None, None

        self.cache = ArtifactCache(
            root=config.cache.root,
            enabled=bool(config.cache.enabled),
            force_recompute=bool(config.cache.force_recompute),
            log_hits=bool(config.cache.log_hits),
        )
        cached = load_cached_xela_sequence(
            cache=self.cache,
            config=config,
            data_path=self.data_path,
            xela_urdf_path=xela_urdf_path,
            baseline_signal_path=self.baseline_signal_path,
        )
        self.timestamps = cached["timestamps"]
        self.num_frames = len(self.timestamps)
        self.xela_array = cached["xela_array"]
        self.joint_angles = cached["joint_angles"]
        self.joint_effort = cached["joint_effort"]
        self.joint_poses = cached["joint_poses"]
        self.sensor_positions = cached["sensor_positions"]
        self.artifact_keys = cached["artifact_keys"]
        self.window_sensor_graphs = cached.get("window_sensor_graphs")
        graph_cfg = config.get("graph", None)
        self.static_graph_edges = (
            graph_cfg is not None and str(graph_cfg.get("topology_mode", "per_window")) in {"static", "static_edges", "constant"}
        )

        if VIS_POSES:
            import matplotlib.pyplot as plt
            from tactile_ssl.utils.plotting_utils import draw_3d_axes, set_equal_aspect_ratio_3D

            sensor_pose = self.read_joint_pose_sample(0)

            ax = plt.subplot(111, projection="3d")
            transform = np.repeat(np.eye(4)[None], 368, axis=0)
            transform[:, :3, 3] = sensor_pose[0, :, :3]
            draw_3d_axes(ax, transform, axis_length=0.005)
            # Get rid of colored axes planes
            # First remove fill
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Now set color to white (or whatever is "invisible")
            ax.xaxis.pane.set_edgecolor("w")
            ax.yaxis.pane.set_edgecolor("w")
            ax.zaxis.pane.set_edgecolor("w")

            # Bonus: To get rid of the grid as well:
            ax.grid(False)
            plt.axis("off")
            set_equal_aspect_ratio_3D(ax, sensor_pose[:, :, 0], sensor_pose[:, :, 1], sensor_pose[:, :, 2], alpha=1.5)
            plt.show()

        max_length = self.num_frames - (self.num_frames % self.num_frames_per_window)
        max_length = max_length - self.num_frames_per_window

        self.data_idxs = np.arange(0, max_length, self.shift_per_window)
        self.evaluation_sample_ids = np.asarray(
            [stable_int64_id("xela-window", self.evaluation_group_id, int(index)) for index in self.data_idxs],
            dtype=np.int64,
        )
        if self.window_sensor_graphs is not None:
            graph_window_start = self.window_sensor_graphs["graph_window_start"]
            if graph_window_start.shape != self.data_idxs.shape or not np.array_equal(graph_window_start, self.data_idxs):
                raise ValueError("Cached graph windows do not match dataset windows")

        if self.load_images:
            self.color_image_path = Path(self.data_path + "/top_camera/color")
            self.depth_image_path = Path(self.data_path + "/top_camera/depth")
            color_timestamps_ = np.loadtxt(self.color_image_path / "timestamps.txt")
            color_filenames = sorted(list(self.color_image_path.glob("*.jpg")))

            start_idx = np.abs(color_timestamps_ - self.timestamps[0]).argmin()
            end_idx = np.abs(color_timestamps_ - self.timestamps[-1]).argmin()
            self.color_filenames = color_filenames[start_idx:end_idx]
            color_timestamps_ = color_timestamps_[start_idx:end_idx]
            self.color_freq = int(round(1 / ((color_timestamps_[-1] - 0.0) / len(color_timestamps_))))
            # At minimum each window should have 1 image
            self.images_per_window = max(1, int(self.color_freq * self.window_time))
            self.color_timestamps = color_timestamps_
            self.image_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_idxs)

    def joint_angles_to_poses(self):
        return self.joint_poses

    def read_images(self, index):
        color_images = []
        for idx in range(index, index + self.images_per_window):
            # image_id = f"{image_id + self.color_start_idx:06d}.jpg"
            image_id = self.color_filenames[idx]
            color_image = cv2.cvtColor(
                cv2.imread(str(image_id), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            color_images.append(color_image)
        return color_images

    def read_joint_sample(self, index):
        joint_angles = self.joint_angles[index : index + self.num_frames_per_window]
        joint_effort = self.joint_effort[index : index + self.num_frames_per_window]

        return joint_angles, joint_effort

    def read_joint_pose_sample(self, index):
        return self.sensor_positions[index : index + self.num_frames_per_window]

    def update_normalization(self, xela_mean, xela_std):
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def __getitem__(self, idx):
        sample_dict = {}
        sample_idx = idx
        index = self.data_idxs[idx]
        timestamp = self.timestamps[index : index + self.num_frames_per_window]

        sensor_data = self.xela_array[index : index + self.num_frames_per_window]

        # sensor_data = pad_xela_sample(sensor_data, self.num_xela_taxels, self.max_sensors_per_taxel)
        joint_angles, _ = self.read_joint_sample(index)
        joint_poses = self.read_joint_pose_sample(index)

        if self.load_images:
            idx = np.abs(self.color_timestamps - timestamp[0]).argmin()
            color_images = self.read_images(idx)
            images = []
            for color_image in color_images:
                image = self.image_transform(color_image)
                images.append(image)
            sample_dict.update({"color_images": torch.stack(images, dim=0)})

        sensor_data = torch.from_numpy(sensor_data).float()
        joint_angles = torch.from_numpy(joint_angles).float()
        sensor_poses = torch.from_numpy(joint_poses).float()
        if self.use_spatial_coords:
            sensor_data = torch.cat([sensor_data, sensor_poses[..., :3]], dim=-1)
        sample_dict.update({"sensor": sensor_data})
        sample_dict.update({"joint_angles": joint_angles})
        sample_dict.update({"sensor_poses": sensor_poses})
        sample_dict.update(
            {
                "group_id": torch.tensor(self.evaluation_group_id, dtype=torch.long),
                "sample_id": torch.tensor(self.evaluation_sample_ids[sample_idx], dtype=torch.long),
            }
        )
        if self.window_sensor_graphs is not None:
            edge_count = int(self.window_sensor_graphs["graph_edge_count"][sample_idx])
            graph = {
                "edge_attr": torch.from_numpy(self.window_sensor_graphs["graph_edge_attr"][sample_idx]).float(),
                "edge_count": torch.tensor(edge_count, dtype=torch.long),
            }
            if not self.static_graph_edges:
                graph["edge_index"] = torch.from_numpy(self.window_sensor_graphs["graph_edge_index"][sample_idx]).long()
            sample_dict.update({"graph": graph})
        if self.object_label is not None:
            sample_dict.update({"object_classification": torch.tensor(self.object_label)})
        return sample_dict


class XelaGraphSSLDataset(XelaSSLDataset):
    def __init__(self, config: DictConfig, *args, **kwargs):
        if config.get("graph") is None:
            config.graph = {}
        if config.graph.get("enabled") is None:
            config.graph.enabled = True
        if config.graph.get("type") is None:
            config.graph.type = "physical"
        if config.graph.get("topology_mode") is None:
            config.graph.topology_mode = "per_window"
        if config.graph.get("edge_attr_mode") is None:
            config.graph.edge_attr_mode = "distance"
        if config.graph.get("bridge_k") is None:
            config.graph.bridge_k = 4
        super().__init__(config=config, *args, **kwargs)


if __name__ == "__main__":
    import os
    import hydra
    import einops
    from omegaconf import OmegaConf, DictConfig
    from tactile_ssl.data.xela.utils import compute_xela_normalization

    def get_xela_datasets(data_cfg: DictConfig):
        def get_xela_dataset(dataset_cfg: DictConfig, dataset_name: str, d_id: int):
            data_path = f"{dataset_cfg.data_path}"
            data_files = os.listdir(data_path)
            dataset_name_exists = True in [f in f"{dataset_name}" for f in data_files]
            if not dataset_name_exists:
                print(f"Dataset {dataset_name} not found")
                return None

            dataset = hydra.utils.instantiate(
                dataset_cfg,
                data_path=f"{data_path}/{dataset_name}/{d_id}",
            )
            return dataset

        dataset_list: List[str] = data_cfg.dataset_list
        train_dataset_ids, val_dataset_ids = (
            data_cfg.train_dataset_ids,
            data_cfg.val_dataset_ids,
        )

        train_datasets, val_datasets = [], []
        for obj in dataset_list:
            for d_id in train_dataset_ids:
                train_datasets.append(get_xela_dataset(data_cfg.dataset, dataset_name=obj, d_id=d_id))
            for d_id in val_dataset_ids:
                val_datasets.append(get_xela_dataset(data_cfg.dataset, dataset_name=obj, d_id=d_id))
        return train_datasets, val_datasets

    with hydra.initialize(version_base="1.3", config_path="../../config"):
        config = hydra.compose(
            config_name="default.yaml",
            overrides=[
                "+experiment=xela/dinov2",
                "paths=default",
                "hydra.job.num=1",
                "paths.output_dir='outputs/.'",
                "paths.work_dir='outputs/.'",
            ],
            return_hydra_config=True,
        )

    print(OmegaConf.to_yaml(config, resolve=True))

    train_dsets, val_dsets = get_xela_datasets(config.data)
    xela_train_array = []
    xela_val_array = []
    xela_mean, xela_std = compute_xela_normalization(train_dsets)
    val_xela_mean, val_xela_std = compute_xela_normalization(val_dsets)
    print(f"xela_mean: {xela_mean}, xela_std: {xela_std}")
    print(f"val_xela_mean: {val_xela_mean}, val_xela_std: {val_xela_std}")

    for dset in train_dsets:
        xela_train_array.append(dset.xela_array)
    xela_train_array = np.concatenate(xela_train_array, axis=0)
    xela_array = einops.rearrange(xela_train_array, "b k c -> (b k) c")
    # xela_array = np.where(xela_array == 0, np.nan, xela_array)

    import matplotlib.pyplot as plt

    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(xela_array[:, 0], bins=100)
    ax1.set_yscale("log")

    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(xela_array[:, 1], bins=100)
    ax2.set_yscale("log")

    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(xela_array[:, 2], bins=100)
    ax3.set_yscale("log")

    plt.show()

    datasets = train_dsets + val_dsets
    object_classes = config.data.dataset_list
    num_samples_per_class = np.zeros(len(object_classes))
    for dataset in datasets:
        num_samples_per_class[dataset.object_label] += len(dataset)

    print(num_samples_per_class)
    object_class_ratios = num_samples_per_class / np.sum(num_samples_per_class)
    print(object_class_ratios)
    object_class_weights = 1 / object_class_ratios
    print(f"Object class weights: {object_class_weights / np.sum(object_class_weights)}")

    dataloader = data.DataLoader(dataset, shuffle=True, num_workers=0, batch_size=1)

    for i, data in enumerate(dataloader, 0):
        for key, value in data.items():
            print(f"{key}: {value.shape}")
        if i > 5:
            break

from typing import List, Optional, Union
from pathlib import Path
import einops
import pickle
import numpy as np
from numpy._typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import io
from scipy.spatial.transform import RotationSpline, Rotation
import torch

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

XELA_FLATTEN_ORDER = {
    "3aftc_palm_link": 30,
    "link_15_4x4_palm_link": 16,
    "link_14_4x4_palm_link": 16,
    "0aftc_palm_link": 30,
    "link_2_4x4_palm_link": 16,
    "link_1A_4x4_palm_link": 16,
    "link_1B_4x4_palm_link": 16,
    "1aftc_palm_link": 30,
    "link_6_4x4_palm_link": 16,
    "link_5A_4x4_palm_link": 16,
    "link_5B_4x4_palm_link": 16,
    "2aftc_palm_link": 30,
    "link_10_4x4_palm_link": 16,
    "link_9A_4x4_palm_link": 16,
    "link_9B_4x4_palm_link": 16,
    "ahr_palm_2_4x6_palm_link": 24,
    "ahr_palm_1_4x6_palm_link": 24,
    "ahr_palm_3_4x6_palm_link": 24,
}


def load_data_dict(data_path: str):
    assert Path(data_path).exists(), f"{data_path} does not exist"
    xela_pkl_path = Path(data_path + "/xela/data.pkl")
    allegro_joint_state_pkl_path = Path(data_path + "/allegro/data.pkl")

    assert xela_pkl_path.exists(), f"{str(xela_pkl_path)} does not exist"
    # assert allegro_joint_state_pkl_path.exists(), f"{str(allegro_joint_state_pkl_path)} does not exist"

    log.info(f"Loading data from {data_path}")
    with open(xela_pkl_path, "rb") as f:
        xela_dict = pickle.load(f)
    if allegro_joint_state_pkl_path.exists():
        with open(allegro_joint_state_pkl_path, "rb") as f:
            allegro_joint_state_dict = pickle.load(f)
    else:
        allegro_joint_state_dict = None

    return xela_dict, allegro_joint_state_dict


def compute_interp_timestamps(modality_timestamps, nominal_freq):
    start_timestamps = []
    end_timestamps = []
    for modality_timestamp in modality_timestamps:
        start_timestamp = modality_timestamp[0]
        end_timestamp = modality_timestamp[-1]
        log.info(f"Start timestamp: {start_timestamp}, End timestamp: {end_timestamp}")
        start_timestamps.append(start_timestamp)
        end_timestamps.append(end_timestamp)
    start_timestamp = max(start_timestamps)
    end_timestamp = min(end_timestamps)
    num_frames = int((end_timestamp - start_timestamp) * nominal_freq)
    nominal_timestamps = np.linspace(start_timestamp, end_timestamp, num_frames)
    return nominal_timestamps, num_frames


def read_xela_data(
    xela_array: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
    xela_force_data: Optional[NDArray] = None,
):
    num_frames = int((resampling_timestamps[-1] - resampling_timestamps[0]) * nominal_freq)
    num_of_sensors = xela_array.shape[1]
    xela_processed_array = np.zeros((num_frames, num_of_sensors, 4))
    xela_timestamps = xela_array[:, 0, 0]
    xela_mag = xela_array[:, :, 1:4]
    xela_mag_interpolated = CubicSpline(xela_timestamps, xela_mag, axis=0)(resampling_timestamps)
    if smooth_data:
        xela_mag_interpolated = savgol_filter(xela_mag_interpolated, nominal_freq // 3, 3, axis=0)
    xela_processed_array[:, :, 0] = resampling_timestamps[:, None]
    xela_processed_array[:, :, 1:4] = xela_mag_interpolated

    if xela_force_data is not None:
        xela_force_array = np.zeros((num_frames, xela_force_data.shape[-2], 7))
        xela_force_mag = xela_force_data[:, :, 1:4]
        xela_force_mag_interpolated = CubicSpline(xela_timestamps, xela_force_mag, axis=0)(resampling_timestamps)
        if smooth_data:
            xela_force_mag_interpolated = savgol_filter(xela_force_mag_interpolated, nominal_freq // 3, 3, axis=0)
        xela_force_array[:, :, 0] = resampling_timestamps[:, None]
        xela_force_array[:, :, 1:4] = xela_force_mag_interpolated
        return xela_processed_array, xela_force_array

    return xela_processed_array


def read_allegro_joint_data(
    joint_data: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
):
    joint_timestamps = joint_data[:, 0]
    joint_angles = joint_data[:, 1:17]
    joint_effort = joint_data[:, 17:]

    joint_angles_interpolated = CubicSpline(joint_timestamps, joint_angles, axis=0)(resampling_timestamps)
    joint_effort_interpolated = CubicSpline(joint_timestamps, joint_effort, axis=0)(resampling_timestamps)

    if smooth_data:
        joint_angles_interpolated = savgol_filter(joint_angles_interpolated, nominal_freq // 3, 3, axis=0)
        joint_effort_interpolated = savgol_filter(joint_effort_interpolated, nominal_freq // 3, 3, axis=0)

    return joint_angles_interpolated, joint_effort_interpolated


def read_force_data(force, resampling_timestamps, max_abs_forceXYZ, nominal_freq, smooth_data=False):
    force_timestamps = force[:, 0]
    unnormalized_force_xyz = force[:, 1:4] / 1.0
    unnormalized_force_xyz[:, -1] *= -1.0
    unnormalized_force_xyz[:, -1] = np.clip(unnormalized_force_xyz[:, -1], 0, None)
    force_xyz = unnormalized_force_xyz / max_abs_forceXYZ
    force_interpolated = CubicSpline(force_timestamps, force_xyz, axis=0)(resampling_timestamps)
    unnormalized_force_interpolated = CubicSpline(force_timestamps, unnormalized_force_xyz, axis=0)(
        resampling_timestamps
    )
    if smooth_data:
        force_interpolated = savgol_filter(force_interpolated, nominal_freq // 3, 3, axis=0)
        unnormalized_force_interpolated = savgol_filter(unnormalized_force_interpolated, nominal_freq // 3, 3, axis=0)
    
    force_interpolated = np.concatenate([resampling_timestamps[:, None], force_interpolated], axis=1)
    unnormalized_force_interpolated = np.concatenate([resampling_timestamps[:, None], unnormalized_force_interpolated], axis=1)
    return force_interpolated, unnormalized_force_interpolated

# def read_force_data(force_data, resampling_timestamps, nominal_freq, smooth_data=False):
#     force_timestamps = force_data[:, 0]
#     force_xyz = force_data[:, 1:4] / 1.0
#     force_xyz[:, -1] *= -1.0
#     force_xyz_interpolated = CubicSpline(force_timestamps, force_xyz, axis=0)(resampling_timestamps)
#     force = np.concatenate([resampling_timestamps[:, None], force_xyz_interpolated], axis=1)
#     return force

def read_pose_data(pose_data, resampling_timestamps, nominal_freq, smooth_data=False):
    pose_timestamps = pose_data[:, 0]
    pose_xyz = pose_data[:, 1:4]
    pose_rot = Rotation.from_quat(pose_data[:, 4:])
    pose_xyz_interpolated = CubicSpline(pose_timestamps, pose_xyz, axis=0)(resampling_timestamps)
    pose_rot_interpolated = RotationSpline(pose_timestamps, pose_rot)(resampling_timestamps)
    pose_rot_interpolated = pose_rot_interpolated.as_quat(canonical=True)
    pose = np.concatenate([resampling_timestamps[:, None], pose_xyz_interpolated, pose_rot_interpolated], axis=1)
    return pose


def pad_xela_sample(
    xela_array: Union[NDArray, torch.Tensor],
    num_taxels: int = 18,
    max_sensors_per_taxel: int = 30,
):
    t, _, c = xela_array.shape

    if isinstance(xela_array, torch.Tensor):
        xela_array_padded = torch.zeros(
            (t, num_taxels, max_sensors_per_taxel, c),
            dtype=xela_array.dtype,
            device=xela_array.device,
        )
    else:
        assert isinstance(xela_array, np.ndarray), "xela_array should be a numpy array"
        xela_array_padded = np.zeros((t, num_taxels, max_sensors_per_taxel, c))
    prev_idx = 0
    for i, (_, v) in enumerate(XELA_FLATTEN_ORDER.items()):
        array = xela_array[:, prev_idx : prev_idx + v]
        xela_array_padded[:, i, : array.shape[-2], :] = array
        prev_idx = prev_idx + v
    xela_array_padded = einops.rearrange(xela_array_padded, "t n k c -> t n (k c)")

    return xela_array_padded


def get_pad_xela_indexes(pad_id: int):
    prev_idx = 0
    next_idx = 0
    for i, (_, v) in enumerate(XELA_FLATTEN_ORDER.items()):
        next_idx += v
        if i == pad_id:
            break
        prev_idx += v
    return prev_idx, next_idx


def pad_xela_sample_torch(
    xela_array: torch.Tensor,
    num_taxels: int = 18,
    max_sensors_per_taxel: int = 30,
):
    t, _, c = xela_array.shape

    xela_array_padded = torch.zeros(
        (t, num_taxels, max_sensors_per_taxel, c),
        dtype=xela_array.dtype,
        device=xela_array.device,
    )
    prev_idx = 0
    for i, (_, v) in enumerate(XELA_FLATTEN_ORDER.items()):
        array = xela_array[:, prev_idx : prev_idx + v]
        xela_array_padded[:, i, : array.shape[-2], :] = array
        prev_idx = prev_idx + v
    xela_array_padded = einops.rearrange(xela_array_padded, "t n k c -> t n (k c)")
    return xela_array_padded


def read_xelaforce_sample(xela_array: NDArray, num_taxels: int = 18, max_sensors_per_taxel: int = 30):
    t, _, c = xela_array.shape
    xela_array_padded = np.zeros((t, num_taxels, max_sensors_per_taxel, c))
    prev_idx = 0
    for i, (k, v) in enumerate(XELA_FLATTEN_ORDER.items()):
        if "aftc" in k:
            continue
        array = xela_array[:, prev_idx : prev_idx + v]
        xela_array_padded[:, i, : array.shape[-2], :] = array
        prev_idx = prev_idx + v
    xela_array_padded = xela_array_padded[..., 1:]
    xela_array_padded = einops.rearrange(xela_array_padded, "t n k c -> t n (k c)")
    return xela_array_padded


def create_sensor_image(k, sensor_grid, resolution=128, aspect_ratio=1.0):
    h, w = sensor_grid.shape[:2]
    w_resolution = int(resolution * aspect_ratio)
    h_resolution = resolution
    x, y = np.linspace(0, w, w_resolution), np.linspace(0, h, h_resolution)

    x_coords = np.linspace(0.5, w + 0.5, w, endpoint=False) * w_resolution
    y_coords = np.linspace(0.5, h + 0.5, h, endpoint=False) * h_resolution
    x_coords /= w
    y_coords /= h

    image = np.zeros((h_resolution, w_resolution, 3), dtype=np.uint8)
    base_resolution = 128
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            overlay = np.zeros((h_resolution, w_resolution, 3), dtype=np.uint8)
            x_offset, y_offset = (
                int(sensor_grid[j, i, 0] / 10),
                int(sensor_grid[j, i, 1] / 10),
            )
            grid_position = (int(x), int(y))
            position = (int(x + x_offset), int(y + y_offset))
            # Note: We're obviously clipping negative values in normal forces that may show up due to inaccurate baseline subtraction
            # radius = int(sensor_grid[j, i, 2])

            radius = max(int(sensor_grid[j, i, 2] / 10), 2)
            radius = int(resolution * radius / base_resolution)
            cv2.circle(
                image,
                grid_position,
                int(4 * resolution / base_resolution),
                color=(255, 0, 0),
                thickness=-1,
            )
            cv2.line(
                image,
                grid_position,
                position,
                color=(255, 0, 0),
                thickness=(2 * resolution // base_resolution),
            )
            # if radius > 0:
            cv2.circle(overlay, position, radius, color=(0, 255, 0), thickness=-1)
            # else:
            #     cv2.circle(overlay, position, -radius, color=(0, 0, 255), thickness=-1)
            image = cv2.addWeighted(overlay, 0.55, image, 1.0, 0)

    if "4x6" in k:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image


def xela_plot_layout(fig):
    # Create a GridSpec with 4 rows and 4 columns
    gs = gridspec.GridSpec(7, 4, height_ratios=[1.2, 1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1])

    axes = []

    for i in range(3):
        ax = fig.add_subplot(gs[i, 3])
        axes.append(ax)

    # 9 square subplots (3x3 grid)
    for i in range(3):
        for j in range(4):
            ax = fig.add_subplot(gs[j, 2 - i])
            axes.append(ax)

    # 3 rectangular subplots below the squares
    gs_palm = gs[4:6, :3].subgridspec(2, 2)

    ax = fig.add_subplot(gs_palm[0, 0])
    axes.append(ax)
    ax = fig.add_subplot(gs_palm[0, 1])
    axes.append(ax)
    ax = fig.add_subplot(gs_palm[1, 0])
    axes.append(ax)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return axes


def xela_sensor_layout_plt(x, xela_mean, xela_std):
    t = x.shape[0]
    sensor_plots = []
    fig = plt.figure(figsize=(10, 15))
    for i in range(t):
        frame = x[i]
        sensor_images = []
        for j, sensor_type in enumerate(XELA_FLATTEN_ORDER.keys()):
            sensor_ = frame[j]
            aspect_ratio = 1.0
            if "aftc" in sensor_type:
                sensor_grid = np.zeros((6, 6, 3))
                sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
                sensor_grid[0, 2:-2] = sensor_[0:2]
                sensor_grid[1, 1:-1] = sensor_[2:6]
            elif "4x4" in sensor_type:
                sensor_ = sensor_[:16, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=4)
            elif "4x6" in sensor_type:
                sensor_ = sensor_[:24, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=6)
                aspect_ratio = 1.5
            else:
                raise ValueError(f"Sensor type {sensor_type} not supported")
            sensor_grid = (sensor_grid * xela_std[None]) + xela_mean[None]
            sensor_image = create_sensor_image(sensor_type, sensor_grid, resolution=128, aspect_ratio=aspect_ratio)
            sensor_images.append(sensor_image)
        axes = xela_plot_layout(fig)
        for j, ax in enumerate(axes):
            ax.imshow(sensor_images[j])
        with io.BytesIO() as buf:
            fig.savefig(buf, format="raw")
            buf.seek(0)
            data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        sensor_plot = data.reshape((h, w, -1))[:, :, :3]
        sensor_plots.append(sensor_plot)
        fig.clf()
    sensor_plots = np.stack(sensor_plots, axis=0)
    sensor_plots = einops.rearrange(sensor_plots, "n h w c -> n c h w")
    plt.close(fig)
    return sensor_plots


XELA_LAYOUT_ORDER = {
    "2aftc_palm_link": 30,
    "1aftc_palm_link": 30,
    "0aftc_palm_link": 30,
    "3aftc_palm_link": 30,
    "link_10_4x4_palm_link": 16,
    "link_6_4x4_palm_link": 16,
    "link_2_4x4_palm_link": 16,
    "link_15_4x4_palm_link": 16,
    "link_9A_4x4_palm_link": 16,
    "link_5A_4x4_palm_link": 16,
    "link_1A_4x4_palm_link": 16,
    "link_14_4x4_palm_link": 16,
    "link_9B_4x4_palm_link": 16,
    "link_5B_4x4_palm_link": 16,
    "link_1B_4x4_palm_link": 16,
    "ahr_palm_2_4x6_palm_link": 24,
    "ahr_palm_1_4x6_palm_link": 24,
    "ahr_palm_3_4x6_palm_link": 24,
}


def xela_sensor_layout(x, xela_mean=None, xela_std=None, mask=None):
    t = x.shape[0]
    sensor_plots = []
    for i in range(t):
        frame = x[i]
        sensor_images = []

        prev_idx = 0
        for j, (sensor_type, num_taxels) in enumerate(XELA_FLATTEN_ORDER.items()):
            sensor_ = frame[prev_idx : prev_idx + num_taxels]
            aspect_ratio = 1.0
            if "aftc" in sensor_type:
                sensor_grid = np.zeros((6, 6, 3))
                sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
                sensor_grid[0, 2:-2] = sensor_[0:2]
                sensor_grid[1, 1:-1] = sensor_[2:6]
            elif "4x4" in sensor_type:
                sensor_ = sensor_[:16, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=4)
            elif "4x6" in sensor_type:
                sensor_ = sensor_[:24, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=6)
                aspect_ratio = 1.5
            else:
                raise ValueError(f"Sensor type {sensor_type} not supported")
            if xela_mean is not None and xela_std is not None:
                sensor_grid = (sensor_grid * xela_std[None]) + xela_mean[None]
            sensor_image = create_sensor_image(sensor_type, sensor_grid, resolution=128, aspect_ratio=aspect_ratio)
            if mask is not None:
                sensor_image = sensor_image * mask[i, j]
            sensor_images.append(sensor_image)
            prev_idx += num_taxels

        idx = []
        canvas_image = np.ones(((128 + 10) * 6, (128 + 10) * 4, 3)) * 255
        flatten_keys = list(XELA_FLATTEN_ORDER.keys())
        layout_keys = list(XELA_LAYOUT_ORDER.keys())
        for i, k in enumerate(layout_keys[:15]):
            idx = flatten_keys.index(k)
            pos_x = (i % 4) * (128 + 5)
            pos_y = (i // 4) * (128 + 5)
            canvas_image[pos_y : pos_y + 128, pos_x : pos_x + 128] = sensor_images[idx]
        for i, k in enumerate(layout_keys[15:]):
            idx = flatten_keys.index(k)
            pos_x = (i % 2) * (192 + 5)
            pos_y = (128 + 10) * 4 + (i // 2) * (128 + 5)
            canvas_image[pos_y : pos_y + 128, pos_x : pos_x + 192] = sensor_images[idx]
        sensor_plots.append(canvas_image)
    sensor_plots = np.stack(sensor_plots, axis=0)
    sensor_plots = einops.rearrange(sensor_plots, "n h w c -> n c h w")

    sensor_plots = sensor_plots.astype(np.uint8)
    return sensor_plots


def xela_patch_layout(x, xela_mean=None, xela_std=None, mask=None):
    t = x.shape[0]
    sensor_plots = []
    for i in range(t):
        frame = x[i]
        sensor_images = []
        for j, sensor_type in enumerate(XELA_FLATTEN_ORDER.keys()):
            sensor_ = frame[j]
            aspect_ratio = 1.0
            if "aftc" in sensor_type:
                sensor_grid = np.zeros((6, 6, 3))
                sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
                sensor_grid[0, 2:-2] = sensor_[0:2]
                sensor_grid[1, 1:-1] = sensor_[2:6]
            elif "4x4" in sensor_type:
                sensor_ = sensor_[:16, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=4)
            elif "4x6" in sensor_type:
                sensor_ = sensor_[:24, :]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=6)
                aspect_ratio = 1.5
            else:
                raise ValueError(f"Sensor type {sensor_type} not supported")
            if xela_mean is not None and xela_std is not None:
                sensor_grid = (sensor_grid * xela_std[None]) + xela_mean[None]
            sensor_image = create_sensor_image(sensor_type, sensor_grid, resolution=128, aspect_ratio=aspect_ratio)
            if mask is not None:
                sensor_image = sensor_image * mask[i, j]
            sensor_images.append(sensor_image)

        idx = []
        canvas_image = np.ones(((128 + 10) * 6, (128 + 10) * 4, 3)) * 255
        flatten_keys = list(XELA_FLATTEN_ORDER.keys())
        layout_keys = list(XELA_LAYOUT_ORDER.keys())
        for i, k in enumerate(layout_keys[:15]):
            idx = flatten_keys.index(k)
            pos_x = (i % 4) * (128 + 5)
            pos_y = (i // 4) * (128 + 5)
            canvas_image[pos_y : pos_y + 128, pos_x : pos_x + 128] = sensor_images[idx]
        for i, k in enumerate(layout_keys[15:]):
            idx = flatten_keys.index(k)
            pos_x = (i % 2) * (192 + 5)
            pos_y = (128 + 10) * 4 + (i // 2) * (128 + 5)
            canvas_image[pos_y : pos_y + 128, pos_x : pos_x + 192] = sensor_images[idx]
        sensor_plots.append(canvas_image)
    sensor_plots = np.stack(sensor_plots, axis=0)
    sensor_plots = einops.rearrange(sensor_plots, "n h w c -> n c h w")

    sensor_plots = sensor_plots.astype(np.uint8)
    return sensor_plots


def xela_flat_to_grid(xela_array):
    # xela_array = xela_array[0][:, 1:]
    sensor_images = []
    for j, sensor_type in enumerate(XELA_FLATTEN_ORDER.keys()):
        start_idx, end_idx = get_pad_xela_indexes(j)
        sensor_ = xela_array[start_idx:end_idx]
        aspect_ratio = 1.0
        if "aftc" in sensor_type:
            sensor_grid = np.zeros((6, 6, 3))
            sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
            sensor_grid[0, 2:-2] = sensor_[0:2]
            sensor_grid[1, 1:-1] = sensor_[2:6]
        elif "4x4" in sensor_type:
            sensor_ = sensor_[:16, :]
            sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=4)
        elif "4x6" in sensor_type:
            sensor_ = sensor_[:24, :]
            sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=6)
        sensor_images.append(sensor_grid)
    return sensor_images

def compute_xela_normalization(xela_datasets: List, per_sensor: bool = False):
    total = None
    total_sq = None
    count = None
    nan_count = 0
    total_frames = 0
    for xela_dataset in xela_datasets:
        xela_array_ = xela_dataset.xela_array
        if xela_array_.shape[-1] == 4:
            xela_array_ = xela_array_[..., 1:]
        assert xela_array_.shape[-1] == 3, "Expected 3 channels"
        assert xela_array_.shape[-2] == 368, "Expected 368 sensors"

        if per_sensor:
            out_shape = xela_array_.shape[-2:]
            reduce_axis = 0
        else:
            out_shape = (xela_array_.shape[-1],)
            reduce_axis = (0, 1)

        if total is None:
            total = np.zeros(out_shape, dtype=np.float64)
            total_sq = np.zeros(out_shape, dtype=np.float64)
            count = np.zeros(out_shape, dtype=np.int64)

        invalid = (xela_array_ == 0) | np.isnan(xela_array_)
        valid = ~invalid
        xela_array_ = xela_array_.astype(np.float64, copy=False)
        total += np.where(valid, xela_array_, 0.0).sum(axis=reduce_axis)
        total_sq += np.where(valid, xela_array_ * xela_array_, 0.0).sum(axis=reduce_axis)
        count += valid.sum(axis=reduce_axis)
        nan_count += int(invalid.sum())
        total_frames += int(xela_array_.shape[0])

    if total is None or total_sq is None or count is None:
        raise ValueError("No Xela datasets provided for normalization")

    print(f"Xela array shape: {(total_frames, 368, 3)}")
    log.warning(f"Number of NaNs in xela_array: {nan_count}")

    xela_mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    variance = np.divide(total_sq, count, out=np.full_like(total_sq, np.nan), where=count > 0) - xela_mean * xela_mean
    variance = np.maximum(variance, 0.0)
    xela_std = np.sqrt(variance)

    if per_sensor:
        assert xela_mean.shape == (368, 3), f"Expected per-sensor normalization shape (368, 3), got {xela_mean.shape}"

    return xela_mean, xela_std


def get_sensor_grid(patch_name):
    if "aftc" in patch_name:
        h, w, d = 0.031, 0.039, 0.029  # numbers taken from mesh boundingbox
        h_res, w_res = 6, 6
        x = np.linspace(0.5 - h_res / 2, h_res / 2 + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx_, yy_ = np.meshgrid(x, y)
        xx = np.concatenate([xx_[:4, :].flatten(), xx_[-2, 1:-1], xx_[-1, 2:-2]], axis=0)
        yy = np.concatenate([yy_[:4, :].flatten(), yy_[-2, 1:-1], yy_[-1, 2:-2]], axis=0)
    elif "4x4" in patch_name:
        h, w, d = 0.026, 0.024, 0.0044  # numbers taken from mesh boundingbox
        h_res, w_res = 4, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    elif "4x6" in patch_name:
        h, w, d = 0.052, 0.032, 0.0  # numbers taken from mesh boundingbox
        h_res, w_res = 6, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    return xx, yy, d


def joint_angles_to_poses(xela_kinematic_chain, joint_angles: np.ndarray):
    joint_angles = torch.tensor(joint_angles).float()
    joint_poses = xela_kinematic_chain.forward_kinematics(joint_angles)
    poses = []
    for k, num_sensors in XELA_FLATTEN_ORDER.items():
        joint_pose = joint_poses[k].get_matrix().numpy()
        xx, yy, d = get_sensor_grid(k)
        sensor_positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        sensor_positions = np.concatenate([sensor_positions, np.zeros_like(sensor_positions)], axis=-1)
        sensor_positions[..., -2] = d
        sensor_positions[..., -1] = 1
        t = joint_pose.shape[0]
        joint_pose = einops.repeat(joint_pose, "t i j -> t s i j", s=num_sensors)
        joint_pose = einops.rearrange(joint_pose, "t s i j -> (t s) i j")
        sensor_positions = einops.repeat(sensor_positions, "s c -> (t s) c", t=t)
        sensor_pose = np.einsum("m i j, m j -> m i", joint_pose, sensor_positions)
        joint_pose[..., :, 3] = sensor_pose
        joint_pose = einops.rearrange(joint_pose, "(t s) i j -> t s i j", s=num_sensors)
        poses.append(joint_pose)
    poses = np.concatenate(poses, axis=1)
    positions = poses[..., :3, 3]
    return positions

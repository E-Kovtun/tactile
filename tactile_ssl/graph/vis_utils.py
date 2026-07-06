from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tactile_ssl.graph.utils import HAND_PART_LABELS_RU, SENSOR_TO_HAND_PART


HAND_PART_COLORS = {
    "thumb": "#8c564b",
    "index_finger": "#1f77b4",
    "middle_finger": "#2ca02c",
    "ring_finger": "#d62728",
    "palm": "#9467bd",
}


def _to_numpy(sensor_positions) -> np.ndarray:
    if hasattr(sensor_positions, "detach"):
        sensor_positions = sensor_positions.detach().cpu().numpy()
    sensor_positions = np.asarray(sensor_positions, dtype=np.float32)
    if sensor_positions.ndim == 2:
        sensor_positions = sensor_positions[None]
    if sensor_positions.ndim != 3 or sensor_positions.shape[1:] != (368, 3):
        raise ValueError(
            "sensor_positions must have shape (368, 3) or (T, 368, 3); "
            f"got {sensor_positions.shape}"
        )
    return sensor_positions


def _normalize_frame_indices(frame_indices: Optional[Iterable[int]], num_frames: int) -> list[int]:
    if frame_indices is None:
        if num_frames == 1:
            return [0]
        return sorted({0, num_frames // 2, num_frames - 1})

    normalized = []
    for frame_idx in frame_indices:
        idx = int(frame_idx)
        if idx < 0:
            idx = num_frames + idx
        if idx < 0 or idx >= num_frames:
            raise IndexError(f"Frame index {frame_idx} is outside 0..{num_frames - 1}")
        normalized.append(idx)
    return normalized


def _set_equal_3d_limits(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _highlight_sensor_ids(highlight_sensor_ids: Optional[Iterable[int]]) -> list[int]:
    if highlight_sensor_ids is None:
        return []
    ids = sorted({int(sensor_id) for sensor_id in highlight_sensor_ids})
    invalid = [sensor_id for sensor_id in ids if sensor_id < 0 or sensor_id >= 368]
    if invalid:
        raise ValueError(f"highlight_sensor_ids contains invalid sensor ids: {invalid}")
    return ids


def plot_sensor_positions(
    sensor_positions,
    frame_indices: Optional[Iterable[int]] = None,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    highlight_sensor_ids: Optional[Iterable[int]] = None,
    show_sensor_ids: bool = False,
    show_legend: bool = True,
    figsize: Optional[Sequence[float]] = None,
    point_size: float = 18.0,
):
    """Plot 3D Xela sensor positions for one or more time frames.

    Args:
        sensor_positions: Array-like with shape (368, 3) or (T, 368, 3).
        frame_indices: Frames to draw. Defaults to first/middle/last for a sequence.
        sensor_to_part: Mapping from flattened sensor id to hand part.
        labels: Display labels for hand parts.
        colors: Matplotlib colors per hand part.
        highlight_sensor_ids: Optional sensor ids to outline in black.
        show_sensor_ids: Annotate every plotted sensor id.
        show_legend: Add a legend to the first subplot.
        figsize: Optional figure size.
        point_size: Scatter marker size.

    Returns:
        matplotlib.figure.Figure
    """
    positions = _to_numpy(sensor_positions)
    selected_frames = _normalize_frame_indices(frame_indices, positions.shape[0])
    highlighted = _highlight_sensor_ids(highlight_sensor_ids)

    if set(sensor_to_part.keys()) != set(range(368)):
        raise ValueError("sensor_to_part must contain exactly sensor ids 0..367")

    if figsize is None:
        figsize = (5.0 * len(selected_frames), 5.0)

    fig = plt.figure(figsize=figsize)
    all_points = positions[selected_frames].reshape(-1, 3)
    hand_parts = list(dict.fromkeys(sensor_to_part.values()))

    for subplot_id, frame_idx in enumerate(selected_frames, start=1):
        ax = fig.add_subplot(1, len(selected_frames), subplot_id, projection="3d")
        frame_points = positions[frame_idx]

        for hand_part in hand_parts:
            sensor_ids = [sensor_id for sensor_id, part in sensor_to_part.items() if part == hand_part]
            pts = frame_points[sensor_ids]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=point_size,
                color=colors.get(hand_part, "tab:gray"),
                label=labels.get(hand_part, hand_part),
                alpha=0.9,
            )

        if highlighted:
            pts = frame_points[highlighted]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=point_size * 3.0,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
            )

        if show_sensor_ids:
            for sensor_id, point in enumerate(frame_points):
                ax.text(point[0], point[1], point[2], str(sensor_id), fontsize=6)

        ax.set_title(f"frame {frame_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_equal_3d_limits(ax, all_points)
        ax.view_init(elev=24, azim=-58)

        if show_legend and subplot_id == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout()
    return fig

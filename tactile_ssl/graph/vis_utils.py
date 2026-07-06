from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tactile_ssl.graph.types import WeightedSensorGraph
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


def _expand_points_from_center(points: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return points
    center = points.mean(axis=0, keepdims=True)
    return center + (points - center) * float(scale)


def _highlight_sensor_ids(highlight_sensor_ids: Optional[Iterable[int]]) -> list[int]:
    if highlight_sensor_ids is None:
        return []
    ids = sorted({int(sensor_id) for sensor_id in highlight_sensor_ids})
    invalid = [sensor_id for sensor_id in ids if sensor_id < 0 or sensor_id >= 368]
    if invalid:
        raise ValueError(f"highlight_sensor_ids contains invalid sensor ids: {invalid}")
    return ids


def _single_frame_positions(sensor_positions) -> np.ndarray:
    positions = _to_numpy(sensor_positions)
    if positions.shape[0] != 1:
        raise ValueError(
            "Graph visualization expects one frame. "
            f"Got {positions.shape[0]} frames; select one frame first."
        )
    return positions[0]


def _hand_parts_in_order(sensor_to_part: Mapping[int, str]) -> list[str]:
    return list(dict.fromkeys(sensor_to_part.values()))


def _draw_sensor_scatter_3d(
    ax,
    points: np.ndarray,
    sensor_to_part: Mapping[int, str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    point_size: float,
    show_legend: bool,
) -> None:
    for hand_part in _hand_parts_in_order(sensor_to_part):
        sensor_ids = [sensor_id for sensor_id, part in sensor_to_part.items() if part == hand_part]
        pts = points[sensor_ids]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=point_size,
            color=colors.get(hand_part, "tab:gray"),
            label=labels.get(hand_part, hand_part),
            alpha=0.9,
        )
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))


def _draw_graph_edges_3d(
    ax,
    points: np.ndarray,
    graph: WeightedSensorGraph,
    edge_color: str = "#333333",
    edge_alpha: float = 0.28,
    edge_linewidth: float = 0.7,
) -> None:
    if graph.edge_index.shape[1] == 0:
        return
    segments = points[graph.edge_index.T]
    collection = Line3DCollection(
        segments,
        colors=edge_color,
        linewidths=edge_linewidth,
        alpha=edge_alpha,
    )
    ax.add_collection3d(collection)


def _project_positions_2d(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _draw_sensor_scatter_2d(
    ax,
    points_2d: np.ndarray,
    sensor_to_part: Mapping[int, str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    point_size: float,
    show_legend: bool,
) -> None:
    for hand_part in _hand_parts_in_order(sensor_to_part):
        sensor_ids = [sensor_id for sensor_id, part in sensor_to_part.items() if part == hand_part]
        pts = points_2d[sensor_ids]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            color=colors.get(hand_part, "tab:gray"),
            label=labels.get(hand_part, hand_part),
            alpha=0.9,
        )
    if show_legend:
        ax.legend(loc="best")


def _draw_graph_edges_2d(
    ax,
    points_2d: np.ndarray,
    graph: WeightedSensorGraph,
    edge_color: str = "#333333",
    edge_alpha: float = 0.24,
    edge_linewidth: float = 0.6,
) -> None:
    if graph.edge_index.shape[1] == 0:
        return
    segments = points_2d[graph.edge_index.T]
    collection = LineCollection(
        segments,
        colors=edge_color,
        linewidths=edge_linewidth,
        alpha=edge_alpha,
        zorder=1,
    )
    ax.add_collection(collection)


def plot_sensor_positions(
    sensor_positions,
    frame_indices: Optional[Iterable[int]] = None,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    highlight_sensor_ids: Optional[Iterable[int]] = None,
    show_sensor_ids: bool = False,
    show_points: bool = True,
    show_legend: bool = True,
    figsize: Optional[Sequence[float]] = None,
    point_size: float = 18.0,
    sensor_id_position_scale: float = 1.0,
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
        show_points: Draw sensor markers. Disable this to show only sensor ids.
        show_legend: Add a legend to the first subplot.
        figsize: Optional figure size.
        point_size: Scatter marker size.
        sensor_id_position_scale: Scale sensor id text positions away from each frame center.

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
    if show_sensor_ids and sensor_id_position_scale != 1.0:
        all_text_points = [
            _expand_points_from_center(positions[frame_idx], sensor_id_position_scale)
            for frame_idx in selected_frames
        ]
        all_points = np.concatenate([all_points, np.concatenate(all_text_points, axis=0)], axis=0)
    hand_parts = list(dict.fromkeys(sensor_to_part.values()))

    for subplot_id, frame_idx in enumerate(selected_frames, start=1):
        ax = fig.add_subplot(1, len(selected_frames), subplot_id, projection="3d")
        frame_points = positions[frame_idx]

        for hand_part in hand_parts:
            sensor_ids = [sensor_id for sensor_id, part in sensor_to_part.items() if part == hand_part]
            pts = frame_points[sensor_ids]
            if show_points:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=point_size,
                    color=colors.get(hand_part, "tab:gray"),
                    label=labels.get(hand_part, hand_part),
                    alpha=0.9,
                )

        if highlighted and show_points:
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
            text_points = _expand_points_from_center(frame_points, sensor_id_position_scale)
            for sensor_id, point in enumerate(text_points):
                ax.text(point[0], point[1], point[2], str(sensor_id), fontsize=6)

        ax.set_title(f"frame {frame_idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_equal_3d_limits(ax, all_points)
        ax.view_init(elev=24, azim=-58)

        if show_legend and show_points and subplot_id == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout()
    return fig


def plot_sensor_positions_2d(
    sensor_positions,
    frame_indices: Optional[Iterable[int]] = None,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    show_sensor_ids: bool = False,
    show_points: bool = True,
    show_legend: bool = True,
    figsize: Optional[Sequence[float]] = None,
    point_size: float = 18.0,
    sensor_id_position_scale: float = 1.0,
):
    """Plot Xela sensor positions in a 2D PCA projection."""
    positions = _to_numpy(sensor_positions)
    selected_frames = _normalize_frame_indices(frame_indices, positions.shape[0])

    if set(sensor_to_part.keys()) != set(range(368)):
        raise ValueError("sensor_to_part must contain exactly sensor ids 0..367")

    if figsize is None:
        figsize = (5.0 * len(selected_frames), 5.0)

    fig, axes = plt.subplots(1, len(selected_frames), figsize=figsize, squeeze=False)
    hand_parts = list(dict.fromkeys(sensor_to_part.values()))

    for ax, frame_idx in zip(axes[0], selected_frames):
        frame_points = positions[frame_idx]
        points_2d = _project_positions_2d(frame_points)

        if show_sensor_ids and sensor_id_position_scale != 1.0:
            text_points_2d = _expand_points_from_center(points_2d, sensor_id_position_scale)
            all_points_2d = np.concatenate([points_2d, text_points_2d], axis=0)
        else:
            text_points_2d = points_2d
            all_points_2d = points_2d

        for hand_part in hand_parts:
            sensor_ids = [sensor_id for sensor_id, part in sensor_to_part.items() if part == hand_part]
            pts = points_2d[sensor_ids]
            if show_points:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=point_size,
                    color=colors.get(hand_part, "tab:gray"),
                    label=labels.get(hand_part, hand_part),
                    alpha=0.9,
                )

        if show_sensor_ids:
            for sensor_id, point in enumerate(text_points_2d):
                ax.text(point[0], point[1], str(sensor_id), fontsize=6, ha="center", va="center")

        ax.set_title(f"frame {frame_idx}: PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(all_points_2d[:, 0].min(), all_points_2d[:, 0].max())
        ax.set_ylim(all_points_2d[:, 1].min(), all_points_2d[:, 1].max())
        ax.margins(0.05)

        if show_legend and show_points and frame_idx == selected_frames[0]:
            ax.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_graph_on_sensor_positions(
    sensor_positions,
    graph: WeightedSensorGraph,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    show_legend: bool = True,
    figsize: Sequence[float] = (6.0, 6.0),
    point_size: float = 18.0,
    edge_color: str = "#333333",
    edge_alpha: float = 0.28,
    edge_linewidth: float = 0.7,
):
    points = _single_frame_positions(sensor_positions)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_graph_edges_3d(ax, points, graph, edge_color, edge_alpha, edge_linewidth)
    _draw_sensor_scatter_3d(ax, points, sensor_to_part, labels, colors, point_size, show_legend)
    ax.set_title(graph.metadata.get("graph_type", "sensor graph"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d_limits(ax, points)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    return fig


def plot_sensor_graph_2d(
    sensor_positions,
    graph: WeightedSensorGraph,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    show_legend: bool = True,
    figsize: Sequence[float] = (6.0, 6.0),
    point_size: float = 18.0,
    edge_color: str = "#333333",
    edge_alpha: float = 0.24,
    edge_linewidth: float = 0.6,
):
    points = _single_frame_positions(sensor_positions)
    points_2d = _project_positions_2d(points)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_graph_edges_2d(ax, points_2d, graph, edge_color, edge_alpha, edge_linewidth)
    _draw_sensor_scatter_2d(ax, points_2d, sensor_to_part, labels, colors, point_size, show_legend)
    ax.set_title(f"{graph.metadata.get('graph_type', 'sensor graph')} PCA projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_sensor_graph_comparison(
    sensor_positions,
    graph: WeightedSensorGraph,
    sensor_to_part: Mapping[int, str] = SENSOR_TO_HAND_PART,
    labels: Mapping[str, str] = HAND_PART_LABELS_RU,
    colors: Mapping[str, str] = HAND_PART_COLORS,
    figsize: Sequence[float] = (16.0, 5.0),
    point_size: float = 18.0,
    edge_color: str = "#333333",
    edge_alpha_3d: float = 0.28,
    edge_alpha_2d: float = 0.24,
    edge_linewidth: float = 0.7,
):
    points = _single_frame_positions(sensor_positions)
    points_2d = _project_positions_2d(points)
    graph_type = graph.metadata.get("graph_type", "sensor graph")

    fig = plt.figure(figsize=figsize)
    ax_sensor = fig.add_subplot(1, 3, 1, projection="3d")
    ax_graph_3d = fig.add_subplot(1, 3, 2, projection="3d")
    ax_graph_2d = fig.add_subplot(1, 3, 3)

    _draw_sensor_scatter_3d(
        ax_sensor,
        points,
        sensor_to_part,
        labels,
        colors,
        point_size,
        show_legend=True,
    )
    ax_sensor.set_title("sensor positions")
    ax_sensor.set_xlabel("x")
    ax_sensor.set_ylabel("y")
    ax_sensor.set_zlabel("z")
    _set_equal_3d_limits(ax_sensor, points)
    ax_sensor.view_init(elev=24, azim=-58)

    _draw_graph_edges_3d(
        ax_graph_3d,
        points,
        graph,
        edge_color=edge_color,
        edge_alpha=edge_alpha_3d,
        edge_linewidth=edge_linewidth,
    )
    _draw_sensor_scatter_3d(
        ax_graph_3d,
        points,
        sensor_to_part,
        labels,
        colors,
        point_size,
        show_legend=False,
    )
    ax_graph_3d.set_title(f"{graph_type}: 3D edges")
    ax_graph_3d.set_xlabel("x")
    ax_graph_3d.set_ylabel("y")
    ax_graph_3d.set_zlabel("z")
    _set_equal_3d_limits(ax_graph_3d, points)
    ax_graph_3d.view_init(elev=24, azim=-58)

    _draw_graph_edges_2d(
        ax_graph_2d,
        points_2d,
        graph,
        edge_color=edge_color,
        edge_alpha=edge_alpha_2d,
        edge_linewidth=edge_linewidth,
    )
    _draw_sensor_scatter_2d(
        ax_graph_2d,
        points_2d,
        sensor_to_part,
        labels,
        colors,
        point_size,
        show_legend=False,
    )
    ax_graph_2d.set_title(f"{graph_type}: 2D PCA")
    ax_graph_2d.set_xlabel("PC1")
    ax_graph_2d.set_ylabel("PC2")
    ax_graph_2d.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig

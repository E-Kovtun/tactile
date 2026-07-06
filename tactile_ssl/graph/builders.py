from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import numpy as np

from tactile_ssl.graph.types import WeightedSensorGraph
from tactile_ssl.graph.utils import PHYSICAL_BRIDGE_LINK_PAIRS, SensorRange, iter_sensor_ranges


def as_single_frame_positions(sensor_positions) -> np.ndarray:
    if hasattr(sensor_positions, "detach"):
        sensor_positions = sensor_positions.detach().cpu().numpy()
    positions = np.asarray(sensor_positions, dtype=np.float32)
    if positions.ndim == 3:
        if positions.shape[0] != 1:
            raise ValueError(
                "Graph builders expect one frame with shape (368, 3). "
                f"Got a sequence with shape {positions.shape}; select one frame first."
            )
        positions = positions[0]
    if positions.shape != (368, 3):
        raise ValueError(f"sensor_positions must have shape (368, 3); got {positions.shape}")
    return positions


def pairwise_sensor_distances(sensor_positions) -> np.ndarray:
    positions = as_single_frame_positions(sensor_positions)
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)


def _sensor_ranges_by_link() -> Dict[str, SensorRange]:
    return {sensor_range.link_name: sensor_range for sensor_range in iter_sensor_ranges()}


def _add_edge(edge_pairs: set[tuple[int, int]], left: int, right: int) -> None:
    if left == right:
        return
    edge_pairs.add((left, right) if left < right else (right, left))


def _grid_cells_for_link(link_name: str) -> Dict[int, tuple[int, int]]:
    if "aftc" in link_name:
        cells = {}
        for row in range(4):
            for col in range(6):
                cells[row * 6 + col] = (row, col)
        for col in range(1, 5):
            cells[24 + col - 1] = (4, col)
        for col in range(2, 4):
            cells[28 + col - 2] = (5, col)
        return cells
    if "4x4" in link_name:
        return {row * 4 + col: (row, col) for row in range(4) for col in range(4)}
    if "4x6" in link_name:
        return {row * 6 + col: (row, col) for row in range(4) for col in range(6)}
    raise ValueError(f"Unsupported Xela link type: {link_name}")


def _local_grid_edges(link_name: str, start: int) -> set[tuple[int, int]]:
    cells = _grid_cells_for_link(link_name)
    by_cell = {cell: local_id for local_id, cell in cells.items()}
    edges: set[tuple[int, int]] = set()
    for local_id, (row, col) in cells.items():
        for neighbor in ((row + 1, col), (row, col + 1)):
            neighbor_id = by_cell.get(neighbor)
            if neighbor_id is not None:
                _add_edge(edges, start + local_id, start + neighbor_id)
    return edges


def _nearest_cross_link_edges(
    positions: np.ndarray,
    left_ids: Sequence[int],
    right_ids: Sequence[int],
    k: int,
) -> set[tuple[int, int]]:
    k = max(0, int(k))
    if k == 0:
        return set()
    left = np.asarray(left_ids, dtype=np.int64)
    right = np.asarray(right_ids, dtype=np.int64)
    dist = np.linalg.norm(positions[left, None, :] - positions[None, right, :], axis=-1)
    flat_order = np.argsort(dist, axis=None)
    edges: set[tuple[int, int]] = set()
    for flat_id in flat_order:
        i, j = np.unravel_index(flat_id, dist.shape)
        _add_edge(edges, int(left[i]), int(right[j]))
        if len(edges) >= k:
            break
    return edges


def _graph_from_edges(
    positions: np.ndarray,
    edge_pairs: Iterable[tuple[int, int]],
    metadata: Optional[Dict[str, Any]] = None,
) -> WeightedSensorGraph:
    unique_edges = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in edge_pairs if i != j})
    if unique_edges:
        edge_index = np.asarray(unique_edges, dtype=np.int64).T
        edge_weight = np.linalg.norm(
            positions[edge_index[0]] - positions[edge_index[1]],
            axis=-1,
        ).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)
    return WeightedSensorGraph(edge_index=edge_index, edge_weight=edge_weight, metadata=metadata)


def build_physical_graph(sensor_positions, bridge_k: int = 1) -> WeightedSensorGraph:
    positions = as_single_frame_positions(sensor_positions)
    ranges_by_link = _sensor_ranges_by_link()
    edge_pairs: set[tuple[int, int]] = set()

    for sensor_range in iter_sensor_ranges():
        edge_pairs.update(_local_grid_edges(sensor_range.link_name, sensor_range.start))

    for left_link, right_link in PHYSICAL_BRIDGE_LINK_PAIRS:
        left_range = ranges_by_link[left_link]
        right_range = ranges_by_link[right_link]
        edge_pairs.update(
            _nearest_cross_link_edges(
                positions,
                list(left_range.sensor_ids),
                list(right_range.sensor_ids),
                k=bridge_k,
            )
        )

    return _graph_from_edges(
        positions,
        edge_pairs,
        metadata={
            "graph_type": "physical",
            "bridge_k": int(bridge_k),
            "bridge_link_pairs": PHYSICAL_BRIDGE_LINK_PAIRS,
        },
    )


def build_distance_threshold_graph(sensor_positions, threshold: float = 0.02) -> WeightedSensorGraph:
    positions = as_single_frame_positions(sensor_positions)
    dist = pairwise_sensor_distances(positions)
    row, col = np.where(np.triu((dist <= float(threshold)) & (dist > 0), k=1))
    edge_pairs = zip(row.tolist(), col.tolist())
    return _graph_from_edges(
        positions,
        edge_pairs,
        metadata={"graph_type": "distance_threshold", "threshold": float(threshold)},
    )


def build_knn_graph(sensor_positions, k: int = 6, symmetrize: bool = True) -> WeightedSensorGraph:
    positions = as_single_frame_positions(sensor_positions)
    k = min(max(0, int(k)), positions.shape[0] - 1)
    dist = pairwise_sensor_distances(positions)
    np.fill_diagonal(dist, np.inf)
    edge_pairs: set[tuple[int, int]] = set()
    directed_edges: list[tuple[int, int]] = []
    if k > 0:
        nearest = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        for src in range(positions.shape[0]):
            for dst in nearest[src]:
                if symmetrize:
                    _add_edge(edge_pairs, src, int(dst))
                elif src != int(dst):
                    directed_edges.append((src, int(dst)))

    if symmetrize:
        return _graph_from_edges(
            positions,
            edge_pairs,
            metadata={"graph_type": "knn", "k": k, "symmetrize": True},
        )

    if directed_edges:
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        edge_weight = np.linalg.norm(
            positions[edge_index[0]] - positions[edge_index[1]],
            axis=-1,
        ).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)
    return WeightedSensorGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        metadata={"graph_type": "knn", "k": k, "symmetrize": False},
    )


def build_sensor_graph(
    sensor_positions,
    graph_type: Literal["physical", "distance_threshold", "knn"],
    **kwargs,
) -> WeightedSensorGraph:
    if graph_type == "physical":
        return build_physical_graph(sensor_positions, **kwargs)
    if graph_type == "distance_threshold":
        return build_distance_threshold_graph(sensor_positions, **kwargs)
    if graph_type == "knn":
        return build_knn_graph(sensor_positions, **kwargs)
    raise ValueError(f"Unsupported graph_type={graph_type!r}")

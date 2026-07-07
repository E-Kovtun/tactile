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


def _sensor_link_ids() -> np.ndarray:
    link_ids = np.empty(368, dtype=object)
    for sensor_range in iter_sensor_ranges():
        link_ids[list(sensor_range.sensor_ids)] = sensor_range.link_name
    return link_ids


def _physical_neighbor_links() -> Dict[str, set[str]]:
    neighbors = {sensor_range.link_name: set() for sensor_range in iter_sensor_ranges()}
    for left_link, right_link in PHYSICAL_BRIDGE_LINK_PAIRS:
        neighbors[left_link].add(right_link)
        neighbors[right_link].add(left_link)
    return neighbors


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


def _dense_pad_edges(sensor_range: SensorRange) -> set[tuple[int, int]]:
    sensor_ids = list(sensor_range.sensor_ids)
    return {
        (sensor_ids[i], sensor_ids[j])
        for i in range(len(sensor_ids))
        for j in range(i + 1, len(sensor_ids))
    }


def _pad_edges(link_pads: Literal["none", "sparse", "dense"]) -> set[tuple[int, int]]:
    edge_pairs: set[tuple[int, int]] = set()
    for sensor_range in iter_sensor_ranges():
        if link_pads == "none":
            continue
        if link_pads == "sparse":
            edge_pairs.update(_local_grid_edges(sensor_range.link_name, sensor_range.start))
        elif link_pads == "dense":
            edge_pairs.update(_dense_pad_edges(sensor_range))
        else:
            raise ValueError(f"Unsupported link_pads={link_pads!r}")
    return edge_pairs


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
    used_left: set[int] = set()
    used_right: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for flat_id in flat_order:
        i, j = np.unravel_index(flat_id, dist.shape)
        if int(i) in used_left or int(j) in used_right:
            continue
        _add_edge(edges, int(left[i]), int(right[j]))
        used_left.add(int(i))
        used_right.add(int(j))
        if len(edges) >= min(k, len(left), len(right)):
            break
    return edges


def _physical_bridge_edges(positions: np.ndarray, bridge_k: int) -> set[tuple[int, int]]:
    ranges_by_link = _sensor_ranges_by_link()
    edge_pairs: set[tuple[int, int]] = set()
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
    return edge_pairs


def _distance_threshold_edges(positions: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    dist = pairwise_sensor_distances(positions)
    row, col = np.where(np.triu((dist <= float(threshold)) & (dist > 0), k=1))
    return set(zip(row.tolist(), col.tolist()))


def _knn_edges(positions: np.ndarray, k: int, allowed_mask: Optional[np.ndarray] = None) -> set[tuple[int, int]]:
    k = min(max(0, int(k)), positions.shape[0] - 1)
    edge_pairs: set[tuple[int, int]] = set()
    if k == 0:
        return edge_pairs

    dist = pairwise_sensor_distances(positions)
    np.fill_diagonal(dist, np.inf)
    if allowed_mask is not None:
        dist = np.where(allowed_mask, dist, np.inf)

    for src in range(positions.shape[0]):
        finite = np.isfinite(dist[src])
        if not finite.any():
            continue
        src_k = min(k, int(finite.sum()))
        nearest = np.argpartition(dist[src], kth=src_k - 1)[:src_k]
        for dst in nearest:
            _add_edge(edge_pairs, src, int(dst))
    return edge_pairs


def _directed_knn_edges(
    positions: np.ndarray,
    k: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    k = min(max(0, int(k)), positions.shape[0] - 1)
    if k == 0:
        return []

    dist = pairwise_sensor_distances(positions)
    np.fill_diagonal(dist, np.inf)
    if allowed_mask is not None:
        dist = np.where(allowed_mask, dist, np.inf)

    directed_edges: list[tuple[int, int]] = []
    for src in range(positions.shape[0]):
        finite = np.isfinite(dist[src])
        if not finite.any():
            continue
        src_k = min(k, int(finite.sum()))
        nearest = np.argpartition(dist[src], kth=src_k - 1)[:src_k]
        for dst in nearest:
            if src != int(dst):
                directed_edges.append((src, int(dst)))
    return directed_edges


def _same_pad_allowed_mask() -> np.ndarray:
    link_ids = _sensor_link_ids()
    allowed = link_ids[:, None] == link_ids[None, :]
    np.fill_diagonal(allowed, False)
    return allowed


def _outer_neighbor_allowed_mask() -> np.ndarray:
    link_ids = _sensor_link_ids()
    physical_neighbors = _physical_neighbor_links()
    allowed = np.ones((368, 368), dtype=bool)
    np.fill_diagonal(allowed, False)
    for src in range(368):
        src_link = link_ids[src]
        blocked_links = physical_neighbors[src_link] | {src_link}
        allowed[src] = np.array([dst_link not in blocked_links for dst_link in link_ids], dtype=bool)
        allowed[src, src] = False
    return allowed


def _boundary_sensor_mask() -> np.ndarray:
    boundary = np.zeros(368, dtype=bool)
    for sensor_range in iter_sensor_ranges():
        cells = _grid_cells_for_link(sensor_range.link_name)
        occupied_cells = set(cells.values())
        for local_id, (row, col) in cells.items():
            is_boundary = any(
                neighbor not in occupied_cells
                for neighbor in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
            )
            boundary[sensor_range.start + local_id] = is_boundary
    return boundary


def _outer_edge_distance_edges(positions: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    dist = pairwise_sensor_distances(positions)
    link_ids = _sensor_link_ids()
    boundary = _boundary_sensor_mask()
    boundary_pair = boundary[:, None] & boundary[None, :]
    cross_pad_pair = link_ids[:, None] != link_ids[None, :]
    within_threshold = (dist <= float(threshold)) & (dist > 0)
    row, col = np.where(np.triu(boundary_pair & cross_pad_pair & within_threshold, k=1))
    return set(zip(row.tolist(), col.tolist()))


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
    edge_pairs = _pad_edges("sparse")
    edge_pairs.update(_physical_bridge_edges(positions, bridge_k))

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
    edge_pairs = _distance_threshold_edges(positions, threshold)
    return _graph_from_edges(
        positions,
        edge_pairs,
        metadata={"graph_type": "distance_threshold", "threshold": float(threshold)},
    )


def _resolve_neighbor_count(
    canonical_value: Optional[int],
    alias_value: Optional[int],
    default: int,
    canonical_name: str,
    alias_name: str,
) -> int:
    if (
        canonical_value is not None
        and alias_value is not None
        and int(canonical_value) != int(alias_value)
    ):
        raise ValueError(
            f"Received both {canonical_name}={canonical_value} and "
            f"{alias_name}={alias_value}; pass only one value."
        )
    value = canonical_value if canonical_value is not None else alias_value
    if value is None:
        value = default
    return max(0, int(value))


def build_knn_graph(
    sensor_positions,
    k: Optional[int] = None,
    symmetrize: bool = True,
    k_inner_neighbors: Optional[int] = None,
    k_outer_neighbors: int = 0,
) -> WeightedSensorGraph:
    positions = as_single_frame_positions(sensor_positions)
    k_inner_neighbors = _resolve_neighbor_count(
        k_inner_neighbors,
        k,
        default=6,
        canonical_name="k_inner_neighbors",
        alias_name="k",
    )
    k_outer_neighbors = max(0, int(k_outer_neighbors))
    inner_mask = _same_pad_allowed_mask()
    outer_mask = _outer_neighbor_allowed_mask()

    if symmetrize:
        edge_pairs: set[tuple[int, int]] = set()
        if k_inner_neighbors > 0:
            edge_pairs.update(
                _knn_edges(positions, k_inner_neighbors, allowed_mask=inner_mask)
            )
        if k_outer_neighbors > 0:
            edge_pairs.update(
                _knn_edges(positions, k_outer_neighbors, allowed_mask=outer_mask)
            )
        return _graph_from_edges(
            positions,
            edge_pairs,
            metadata={
                "graph_type": "knn",
                "k_inner_neighbors": k_inner_neighbors,
                "k_outer_neighbors": k_outer_neighbors,
                "symmetrize": True,
            },
        )

    directed_edges: list[tuple[int, int]] = []
    if k_inner_neighbors > 0:
        directed_edges.extend(
            _directed_knn_edges(positions, k_inner_neighbors, allowed_mask=inner_mask)
        )
    if k_outer_neighbors > 0:
        directed_edges.extend(
            _directed_knn_edges(positions, k_outer_neighbors, allowed_mask=outer_mask)
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
        metadata={
            "graph_type": "knn",
            "k_inner_neighbors": k_inner_neighbors,
            "k_outer_neighbors": k_outer_neighbors,
            "symmetrize": False,
        },
    )


def build_custom_graph(
    sensor_positions,
    link_pads: Literal["none", "sparse", "dense"] = "sparse",
    phys_bridge_k: int = 1,
    distance_threshold: Optional[float] = None,
    k_inner_neighbors: Optional[int] = None,
    k_outer_neighbors: Optional[int] = None,
    outer_edge_distance: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    k_extra_neighbors: Optional[int] = None,
) -> WeightedSensorGraph:
    """Build a composable Xela sensor graph from local, physical, and metric edges.

    Args:
        sensor_positions: Sensor coordinates for one frame, shape (368, 3), in meters.
        link_pads: Controls connectivity inside each physical Xela sensor pad. ``"none"``
            adds no intra-pad edges; sensors on the same pad can still be linked by other
            mechanisms below. ``"sparse"`` adds local 4-neighbor taxel-grid edges inside
            each flat or curved pad. ``"dense"`` fully connects all sensors that belong
            to the same pad.
        phys_bridge_k: Number of shortest inter-pad edges to add for every known physically
            adjacent pad pair. ``0`` disables physical pad-to-pad bridges.
        distance_threshold: Optional global radius graph threshold in meters. When provided
            and positive, every sensor pair with Euclidean distance <= threshold is connected.
        k_inner_neighbors: Optional pad-local kNN component. For each sensor, only sensors on
            the same physical pad are considered as candidates; edges are symmetrized. ``0``
            disables this component.
        k_outer_neighbors: Optional long-range kNN component. For each sensor, candidates from
            the same pad and from physically adjacent pads are excluded first; the sensor is
            then connected to the k nearest remaining sensors and edges are symmetrized. ``0``
            disables this component.
        outer_edge_distance: Optional radius graph threshold in meters for boundary sensors
            only. It connects boundary sensors from different physical pads when their
            Euclidean distance is <= threshold.
        k_neighbors: Deprecated alias for ``k_inner_neighbors``.
        k_extra_neighbors: Deprecated alias for ``k_outer_neighbors``.
    """
    positions = as_single_frame_positions(sensor_positions)
    k_inner_neighbors = _resolve_neighbor_count(
        k_inner_neighbors,
        k_neighbors,
        default=0,
        canonical_name="k_inner_neighbors",
        alias_name="k_neighbors",
    )
    k_outer_neighbors = _resolve_neighbor_count(
        k_outer_neighbors,
        k_extra_neighbors,
        default=0,
        canonical_name="k_outer_neighbors",
        alias_name="k_extra_neighbors",
    )
    edge_pairs = _pad_edges(link_pads)

    if phys_bridge_k > 0:
        edge_pairs.update(_physical_bridge_edges(positions, phys_bridge_k))
    if distance_threshold is not None and distance_threshold > 0:
        edge_pairs.update(_distance_threshold_edges(positions, distance_threshold))
    if k_inner_neighbors > 0:
        edge_pairs.update(
            _knn_edges(
                positions,
                k_inner_neighbors,
                allowed_mask=_same_pad_allowed_mask(),
            )
        )
    if k_outer_neighbors > 0:
        edge_pairs.update(
            _knn_edges(
                positions,
                k_outer_neighbors,
                allowed_mask=_outer_neighbor_allowed_mask(),
            )
        )
    if outer_edge_distance is not None and outer_edge_distance > 0:
        edge_pairs.update(_outer_edge_distance_edges(positions, outer_edge_distance))

    return _graph_from_edges(
        positions,
        edge_pairs,
        metadata={
            "graph_type": "custom",
            "link_pads": link_pads,
            "phys_bridge_k": int(phys_bridge_k),
            "distance_threshold": None if distance_threshold is None else float(distance_threshold),
            "k_inner_neighbors": int(k_inner_neighbors),
            "k_outer_neighbors": int(k_outer_neighbors),
            "outer_edge_distance": None if outer_edge_distance is None else float(outer_edge_distance),
        },
    )


def build_sensor_graph(
    sensor_positions,
    graph_type: Literal["physical", "distance_threshold", "knn", "custom"],
    **kwargs,
) -> WeightedSensorGraph:
    if graph_type == "physical":
        return build_physical_graph(sensor_positions, **kwargs)
    if graph_type == "distance_threshold":
        return build_distance_threshold_graph(sensor_positions, **kwargs)
    if graph_type == "knn":
        return build_knn_graph(sensor_positions, **kwargs)
    if graph_type == "custom":
        return build_custom_graph(sensor_positions, **kwargs)
    raise ValueError(f"Unsupported graph_type={graph_type!r}")

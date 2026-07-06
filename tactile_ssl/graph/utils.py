from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SensorRange:
    link_name: str
    hand_part: str
    start: int
    end: int

    @property
    def sensor_ids(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class WeightedSensorGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    num_nodes: int = 368
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_weight = np.asarray(self.edge_weight, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E); got {edge_index.shape}")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_weight must have shape (E,) matching edge_index; "
                f"got {edge_weight.shape} and {edge_index.shape}"
            )
        if edge_index.size:
            if edge_index.min() < 0 or edge_index.max() >= self.num_nodes:
                raise ValueError(f"edge_index values must be within 0..{self.num_nodes - 1}")
            if np.any(edge_index[0] == edge_index[1]):
                raise ValueError("Self-loops are not included in sensor graphs")
        if np.any(edge_weight <= 0):
            raise ValueError("All edge weights must be positive distances")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_weight", edge_weight)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


# This order mirrors tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER, but is kept
# local so the graph utilities can be imported without the heavier data stack.
XELA_LINK_SENSOR_COUNTS = (
    ("3aftc_palm_link", 30),
    ("link_15_4x4_palm_link", 16),
    ("link_14_4x4_palm_link", 16),
    ("0aftc_palm_link", 30),
    ("link_2_4x4_palm_link", 16),
    ("link_1A_4x4_palm_link", 16),
    ("link_1B_4x4_palm_link", 16),
    ("1aftc_palm_link", 30),
    ("link_6_4x4_palm_link", 16),
    ("link_5A_4x4_palm_link", 16),
    ("link_5B_4x4_palm_link", 16),
    ("2aftc_palm_link", 30),
    ("link_10_4x4_palm_link", 16),
    ("link_9A_4x4_palm_link", 16),
    ("link_9B_4x4_palm_link", 16),
    ("ahr_palm_2_4x6_palm_link", 24),
    ("ahr_palm_1_4x6_palm_link", 24),
    ("ahr_palm_3_4x6_palm_link", 24),
)

LINK_TO_HAND_PART: Dict[str, str] = {
    "3aftc_palm_link": "thumb",
    "link_15_4x4_palm_link": "thumb",
    "link_14_4x4_palm_link": "thumb",
    "0aftc_palm_link": "index_finger",
    "link_2_4x4_palm_link": "index_finger",
    "link_1A_4x4_palm_link": "index_finger",
    "link_1B_4x4_palm_link": "index_finger",
    "1aftc_palm_link": "middle_finger",
    "link_6_4x4_palm_link": "middle_finger",
    "link_5A_4x4_palm_link": "middle_finger",
    "link_5B_4x4_palm_link": "middle_finger",
    "2aftc_palm_link": "ring_finger",
    "link_10_4x4_palm_link": "ring_finger",
    "link_9A_4x4_palm_link": "ring_finger",
    "link_9B_4x4_palm_link": "ring_finger",
    "ahr_palm_2_4x6_palm_link": "palm",
    "ahr_palm_1_4x6_palm_link": "palm",
    "ahr_palm_3_4x6_palm_link": "palm",
}

HAND_PART_LABELS_RU: Dict[str, str] = {
    "thumb": "большой палец",
    "index_finger": "указательный палец",
    "middle_finger": "средний палец",
    "ring_finger": "безымянный палец",
    "palm": "ладонь",
}

PHYSICAL_BRIDGE_LINK_PAIRS = (
    ("3aftc_palm_link", "link_15_4x4_palm_link"),
    ("link_15_4x4_palm_link", "link_14_4x4_palm_link"),
    ("link_14_4x4_palm_link", "ahr_palm_1_4x6_palm_link"),
    ("0aftc_palm_link", "link_2_4x4_palm_link"),
    ("link_2_4x4_palm_link", "link_1A_4x4_palm_link"),
    ("link_1A_4x4_palm_link", "link_1B_4x4_palm_link"),
    ("link_1B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("1aftc_palm_link", "link_6_4x4_palm_link"),
    ("link_6_4x4_palm_link", "link_5A_4x4_palm_link"),
    ("link_5A_4x4_palm_link", "link_5B_4x4_palm_link"),
    ("link_5B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("2aftc_palm_link", "link_10_4x4_palm_link"),
    ("link_10_4x4_palm_link", "link_9A_4x4_palm_link"),
    ("link_9A_4x4_palm_link", "link_9B_4x4_palm_link"),
    ("link_9B_4x4_palm_link", "ahr_palm_3_4x6_palm_link"),
    ("ahr_palm_1_4x6_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("ahr_palm_2_4x6_palm_link", "ahr_palm_3_4x6_palm_link"),
)


def iter_sensor_ranges(
    link_sensor_counts: Iterable[tuple[str, int]] = XELA_LINK_SENSOR_COUNTS,
    link_to_part: Mapping[str, str] = LINK_TO_HAND_PART,
) -> Iterator[SensorRange]:
    start = 0
    for link_name, count in link_sensor_counts:
        if link_name not in link_to_part:
            raise KeyError(f"Missing hand part for Xela link {link_name!r}")
        end = start + count
        yield SensorRange(
            link_name=link_name,
            hand_part=link_to_part[link_name],
            start=start,
            end=end,
        )
        start = end


def _as_single_frame_positions(sensor_positions) -> np.ndarray:
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


def pairwise_sensor_distances(sensor_positions) -> np.ndarray:
    positions = _as_single_frame_positions(sensor_positions)
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)


def build_physical_graph(sensor_positions, bridge_k: int = 1) -> WeightedSensorGraph:
    positions = _as_single_frame_positions(sensor_positions)
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
    positions = _as_single_frame_positions(sensor_positions)
    dist = pairwise_sensor_distances(positions)
    row, col = np.where(np.triu((dist <= float(threshold)) & (dist > 0), k=1))
    edge_pairs = zip(row.tolist(), col.tolist())
    return _graph_from_edges(
        positions,
        edge_pairs,
        metadata={"graph_type": "distance_threshold", "threshold": float(threshold)},
    )


def build_knn_graph(sensor_positions, k: int = 6, symmetrize: bool = True) -> WeightedSensorGraph:
    positions = _as_single_frame_positions(sensor_positions)
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


def _connected_components(num_nodes: int, edge_index: np.ndarray) -> List[List[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edge_index.T.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)

    seen = np.zeros(num_nodes, dtype=bool)
    components = []
    for node in range(num_nodes):
        if seen[node]:
            continue
        stack = [node]
        seen[node] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def sensor_graph_stats(graph: WeightedSensorGraph) -> Dict[str, float]:
    degrees = np.zeros(graph.num_nodes, dtype=np.int64)
    if graph.edge_index.shape[1]:
        np.add.at(degrees, graph.edge_index[0], 1)
        np.add.at(degrees, graph.edge_index[1], 1)
    components = _connected_components(graph.num_nodes, graph.edge_index)
    component_sizes = [len(component) for component in components]
    edge_weight = graph.edge_weight
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.shape[1]),
        "avg_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "num_components": int(len(components)),
        "largest_component": int(max(component_sizes) if component_sizes else 0),
        "min_edge_weight": float(edge_weight.min()) if edge_weight.size else 0.0,
        "mean_edge_weight": float(edge_weight.mean()) if edge_weight.size else 0.0,
        "max_edge_weight": float(edge_weight.max()) if edge_weight.size else 0.0,
    }


def _build_sensor_to_part() -> Dict[int, str]:
    sensor_to_part: Dict[int, str] = {}
    for sensor_range in iter_sensor_ranges():
        for sensor_id in sensor_range.sensor_ids:
            if sensor_id in sensor_to_part:
                raise ValueError(f"Sensor {sensor_id} is mapped more than once")
            sensor_to_part[sensor_id] = sensor_range.hand_part
    _validate_complete_sensor_mapping(sensor_to_part)
    return sensor_to_part


def _build_part_to_sensors(sensor_to_part: Mapping[int, str]) -> Dict[str, List[int]]:
    part_to_sensors: defaultdict[str, List[int]] = defaultdict(list)
    for sensor_id, hand_part in sorted(sensor_to_part.items()):
        part_to_sensors[hand_part].append(sensor_id)
    return dict(part_to_sensors)


def _validate_complete_sensor_mapping(sensor_to_part: Mapping[int, str], num_sensors: int = 368) -> None:
    expected_ids = set(range(num_sensors))
    actual_ids = set(sensor_to_part.keys())
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"Expected sensor ids 0..{num_sensors - 1}; missing={missing}, extra={extra}"
        )


def validate_against_xela_flatten_order(
    xela_flatten_order: Optional[Mapping[str, int]] = None,
) -> None:
    """Verify that the local graph order still matches the data preprocessing order.

    Passing `xela_flatten_order` keeps this function dependency-light. If omitted,
    it imports tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER lazily.
    """
    if xela_flatten_order is None:
        from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER

        xela_flatten_order = XELA_FLATTEN_ORDER

    local_order = dict(XELA_LINK_SENSOR_COUNTS)
    if dict(xela_flatten_order) != local_order:
        raise ValueError(
            "tactile_ssl.graph.utils.XELA_LINK_SENSOR_COUNTS does not match "
            "tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER"
        )
    _validate_complete_sensor_mapping(SENSOR_TO_HAND_PART)


SENSOR_TO_HAND_PART: Dict[int, str] = _build_sensor_to_part()
HAND_PART_TO_SENSORS: Dict[str, List[int]] = _build_part_to_sensors(SENSOR_TO_HAND_PART)

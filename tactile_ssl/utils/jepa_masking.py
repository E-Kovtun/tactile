import heapq
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import torch
from torch.utils.data import default_collate, get_worker_info

from tactile_ssl.graph.types import WeightedSensorGraph
from tactile_ssl.utils.masking import sample_block_size_1d


GrowthStrategy = Literal["dijkstra", "bfs"]
RegionStrategy = Literal["connected_region", "random"]
TargetRegionStrategy = Literal[
    "connected_region",
    "random",
    "stratified_random",
    "stratified_connected_lobes",
    "graph_farthest_points",
    "geodesic_corridor",
    "geodesic_endcaps",
    "topological_endcaps",
]


@dataclass(frozen=True)
class _StratifiedConnectedLobePlan:
    """Graph-only work shared by all stratified targets for one sample."""

    group_ids: torch.Tensor
    groups: torch.Tensor
    base_size: int
    remainder: int
    induced_adjacencies: tuple[Sequence[Sequence[tuple[int, float]]], ...]
    eligible_by_quota: tuple[Mapping[int, Sequence[int]], ...]


def _as_cpu_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _build_undirected_adjacency(
    num_nodes: int,
    edge_index,
    edge_weight,
) -> list[list[tuple[int, float]]]:
    edge_index = _as_cpu_tensor(edge_index, dtype=torch.long)
    edge_weight = _as_cpu_tensor(edge_weight, dtype=torch.float32).reshape(-1)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E); got {tuple(edge_index.shape)}")
    if edge_weight.numel() != edge_index.shape[1]:
        raise ValueError(
            "edge_weight must contain one value per edge; "
            f"got {edge_weight.numel()} weights for {edge_index.shape[1]} edges"
        )
    if edge_index.shape[1] == 0:
        return [[] for _ in range(num_nodes)]

    edges = edge_index.numpy()
    weights = edge_weight.numpy()
    if edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError(f"edge_index values must be within 0..{num_nodes - 1}")
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("All graph edge weights must be finite and positive")

    left = np.minimum(edges[0], edges[1])
    right = np.maximum(edges[0], edges[1])
    non_self = left != right
    left, right, weights = left[non_self], right[non_self], weights[non_self]
    if left.size == 0:
        return [[] for _ in range(num_nodes)]

    # Sort canonical undirected pairs by (left, right, weight). The first item
    # of every pair is therefore its minimum weight, so deduplication remains
    # deterministic without per-edge Torch scalar conversions or dictionaries.
    order = np.lexsort((weights, right, left))
    left, right, weights = left[order], right[order], weights[order]
    unique = np.ones(left.shape[0], dtype=bool)
    unique[1:] = (left[1:] != left[:-1]) | (right[1:] != right[:-1])
    left, right, weights = left[unique], right[unique], weights[unique]

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for left_node, right_node, weight in zip(
        left.tolist(),
        right.tolist(),
        weights.tolist(),
    ):
        adjacency[left_node].append((right_node, weight))
        adjacency[right_node].append((left_node, weight))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    return adjacency


def _connected_components(
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> list[list[int]]:
    seen = [False] * len(adjacency)
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def _connected_components_for_nodes(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    nodes: Sequence[int],
) -> list[list[int]]:
    """Find components restricted to ``nodes`` without scanning empty outsiders."""
    allowed = set(nodes)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor in allowed and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _eligible_seed_nodes(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    components: Optional[Sequence[Sequence[int]]] = None,
) -> list[int]:
    components = components if components is not None else _connected_components(adjacency)
    return [node for component in components if len(component) >= size for node in component]


def _sample_seed(eligible: Sequence[int], generator: Optional[torch.Generator]) -> int:
    index = int(torch.randint(len(eligible), (1,), generator=generator).item())
    return int(eligible[index])


def _sample_bfs_region(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    seed: int,
    size: int,
    generator: Optional[torch.Generator],
) -> list[int]:
    selected: list[int] = []
    discovered = {seed}
    frontier = [seed]
    while frontier and len(selected) < size:
        order = torch.randperm(len(frontier), generator=generator).tolist()
        level = [frontier[index] for index in order]
        next_frontier: list[int] = []
        for node in level:
            if len(selected) >= size:
                break
            selected.append(node)
            neighbors = [neighbor for neighbor, _ in adjacency[node] if neighbor not in discovered]
            if neighbors:
                neighbor_order = torch.randperm(len(neighbors), generator=generator).tolist()
                for index in neighbor_order:
                    neighbor = neighbors[index]
                    if neighbor not in discovered:
                        discovered.add(neighbor)
                        next_frontier.append(neighbor)
        frontier = next_frontier
    return selected


def _bfs_distances(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    seed: int,
) -> list[int]:
    """Return unweighted hop distances from ``seed``."""
    unreachable = len(adjacency) + 1
    distances = [unreachable] * len(adjacency)
    distances[seed] = 0
    frontier = deque([seed])
    while frontier:
        node = frontier.popleft()
        next_distance = distances[node] + 1
        for neighbor, _ in adjacency[node]:
            if distances[neighbor] == unreachable:
                distances[neighbor] = next_distance
                frontier.append(neighbor)
    return distances


def _sample_dijkstra_region(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    seed: int,
    size: int,
    generator: Optional[torch.Generator],
) -> list[int]:
    tie_breakers = torch.rand(len(adjacency), generator=generator).tolist()
    distances = [math.inf] * len(adjacency)
    distances[seed] = 0.0
    heap = [(0.0, tie_breakers[seed], seed)]
    settled: set[int] = set()
    selected: list[int] = []

    while heap and len(selected) < size:
        distance, _, node = heapq.heappop(heap)
        if node in settled or distance != distances[node]:
            continue
        settled.add(node)
        selected.append(node)
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, tie_breakers[neighbor], neighbor))
    return selected


def _dijkstra_tree(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    seed: int,
    generator: Optional[torch.Generator],
) -> tuple[list[float], list[Optional[int]], list[int]]:
    """Return weighted distances and a deterministic randomized shortest-path tree."""
    tie_breakers = torch.rand(len(adjacency), generator=generator).tolist()
    distances = [math.inf] * len(adjacency)
    predecessors: list[Optional[int]] = [None] * len(adjacency)
    hop_counts = [len(adjacency) + 1] * len(adjacency)
    distances[seed] = 0.0
    hop_counts[seed] = 0
    heap = [(0.0, tie_breakers[seed], seed)]

    while heap:
        distance, _, node = heapq.heappop(heap)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            candidate_hops = hop_counts[node] + 1
            if candidate < distances[neighbor] or (
                candidate == distances[neighbor] and candidate_hops < hop_counts[neighbor]
            ):
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                hop_counts[neighbor] = candidate_hops
                heapq.heappush(
                    heap,
                    (candidate, tie_breakers[neighbor], neighbor),
                )
    return distances, predecessors, hop_counts


def _weighted_distance_matrix(
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> np.ndarray:
    """Compute all-pairs weighted distances in compiled sparse code."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError as error:
        raise RuntimeError(
            "graph_farthest_points requires scipy for efficient all-pairs Dijkstra"
        ) from error

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for node, neighbors in enumerate(adjacency):
        for neighbor, weight in neighbors:
            rows.append(node)
            columns.append(neighbor)
            weights.append(weight)
    matrix = csr_matrix(
        (
            np.asarray(weights, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(len(adjacency), len(adjacency)),
        dtype=np.float64,
    )
    return np.asarray(dijkstra(matrix, directed=False), dtype=np.float64)


def _sample_graph_farthest_points_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    generator: Optional[torch.Generator],
    eligible: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Greedily spread points by weighted graph distance."""
    eligible = list(eligible) if eligible is not None else _eligible_seed_nodes(adjacency, size)
    if not eligible:
        raise ValueError(f"No connected component contains the requested {size} nodes")

    distances = _weighted_distance_matrix(adjacency)
    seed = _sample_seed(eligible, generator)
    component_mask = np.isfinite(distances[seed])
    if int(component_mask.sum()) < size:
        raise RuntimeError(
            f"Seed component contains {int(component_mask.sum())} nodes, below requested {size}"
        )

    tie_breakers = np.asarray(
        torch.rand(len(adjacency), generator=generator).tolist(),
        dtype=np.float64,
    )
    selected = [seed]
    selected_mask = np.zeros(len(adjacency), dtype=bool)
    selected_mask[seed] = True
    minimum_distances = distances[seed].copy()

    while len(selected) < size:
        candidates = component_mask & ~selected_mask
        farthest_distance = minimum_distances[candidates].max()
        tied = np.flatnonzero(
            candidates & np.isclose(minimum_distances, farthest_distance, rtol=0.0, atol=1e-12)
        )
        node = int(tied[np.argmax(tie_breakers[tied])])
        selected.append(node)
        selected_mask[node] = True
        minimum_distances = np.minimum(minimum_distances, distances[node])

    return torch.tensor(selected, dtype=torch.long)


def _reconstruct_path(
    predecessors: Sequence[Optional[int]],
    seed: int,
    endpoint: int,
) -> list[int]:
    path = [endpoint]
    while path[-1] != seed:
        predecessor = predecessors[path[-1]]
        if predecessor is None:
            raise RuntimeError(f"No shortest path from {seed} to {endpoint}")
        path.append(predecessor)
    path.reverse()
    return path


def _dilate_path_region(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    path: Sequence[int],
    size: int,
    generator: Optional[torch.Generator],
) -> list[int]:
    """Grow a shortest path by weighted distance from the whole path."""
    selected = list(path)
    selected_set = set(selected)
    if len(selected) == size:
        return selected

    tie_breakers = torch.rand(len(adjacency), generator=generator).tolist()
    distances = [math.inf] * len(adjacency)
    heap: list[tuple[float, float, int]] = []
    for node in path:
        distances[node] = 0.0
        heapq.heappush(heap, (0.0, tie_breakers[node], node))

    settled: set[int] = set()
    while heap and len(selected) < size:
        distance, _, node = heapq.heappop(heap)
        if node in settled or distance != distances[node]:
            continue
        settled.add(node)
        if node not in selected_set:
            selected.append(node)
            selected_set.add(node)
            if len(selected) == size:
                break
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(
                    heap,
                    (candidate, tie_breakers[neighbor], neighbor),
                )
    return selected


def _sample_geodesic_corridor_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    farthest_quantile: float,
    generator: Optional[torch.Generator],
    eligible: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if not 0.0 <= farthest_quantile < 1.0:
        raise ValueError("farthest_quantile must be within [0, 1)")
    eligible = list(eligible) if eligible is not None else _eligible_seed_nodes(adjacency, size)
    if not eligible:
        raise ValueError(f"No connected component contains the requested {size} nodes")

    seed = _sample_seed(eligible, generator)
    distances, predecessors, hop_counts = _dijkstra_tree(adjacency, seed, generator)
    endpoint_candidates = [
        node
        for node, distance in enumerate(distances)
        if node != seed and math.isfinite(distance) and hop_counts[node] + 1 <= size
    ]
    if not endpoint_candidates:
        raise ValueError(
            f"No endpoint has a complete shortest path that fits in a {size}-node corridor"
        )

    endpoint_candidates.sort(key=lambda node: (-distances[node], node))
    farthest_count = max(
        1,
        int(math.ceil((1.0 - farthest_quantile) * len(endpoint_candidates))),
    )
    farthest_candidates = endpoint_candidates[:farthest_count]
    endpoint = farthest_candidates[
        int(torch.randint(len(farthest_candidates), (1,), generator=generator).item())
    ]
    path = _reconstruct_path(predecessors, seed, endpoint)
    selected = _dilate_path_region(adjacency, path, size, generator)
    if len(selected) != size:
        raise RuntimeError(f"Corridor dilation returned {len(selected)} nodes instead of {size}")
    return torch.tensor(selected, dtype=torch.long)


def _sample_geodesic_endcaps_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    farthest_quantile: float,
    lobe_size_ratio: float,
    min_lobe_tokens: int,
    generator: Optional[torch.Generator],
    eligible: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if not 0.0 <= farthest_quantile < 1.0:
        raise ValueError("farthest_quantile must be within [0, 1)")
    if not 0.0 < lobe_size_ratio < 1.0:
        raise ValueError("lobe_size_ratio must be within (0, 1)")
    first_lobe_size = int(size * lobe_size_ratio)
    second_lobe_size = size - first_lobe_size
    if min(first_lobe_size, second_lobe_size) < min_lobe_tokens:
        raise ValueError(
            f"A {size}-node target split with lobe_size_ratio={lobe_size_ratio} "
            f"produces lobes of {first_lobe_size} and {second_lobe_size} nodes, "
            f"below min_lobe_tokens={min_lobe_tokens}"
        )

    eligible = list(eligible) if eligible is not None else _eligible_seed_nodes(adjacency, size)
    if not eligible:
        raise ValueError(f"No connected component contains the requested {size} nodes")

    seed = _sample_seed(eligible, generator)
    first_lobe = _sample_dijkstra_region(
        adjacency,
        seed=seed,
        size=first_lobe_size,
        generator=generator,
    )
    first_lobe_set = set(first_lobe)
    distances, _, _ = _dijkstra_tree(adjacency, seed, generator)
    endpoint_candidates = [
        node
        for node, distance in enumerate(distances)
        if node not in first_lobe_set and math.isfinite(distance)
    ]
    if not endpoint_candidates:
        raise ValueError("No endpoint remains outside the first geodesic endcap")

    endpoint_candidates.sort(key=lambda node: (-distances[node], node))
    farthest_count = max(
        1,
        int(math.ceil((1.0 - farthest_quantile) * len(endpoint_candidates))),
    )
    farthest_candidates = endpoint_candidates[:farthest_count]
    endpoint_order = torch.randperm(
        len(farthest_candidates),
        generator=generator,
    ).tolist()
    for endpoint_id in endpoint_order:
        second_lobe = _sample_dijkstra_region(
            adjacency,
            seed=farthest_candidates[endpoint_id],
            size=second_lobe_size,
            generator=generator,
        )
        if first_lobe_set.isdisjoint(second_lobe):
            return torch.tensor(first_lobe + second_lobe, dtype=torch.long)

    raise ValueError(
        "Could not grow two disjoint geodesic endcaps from the configured farthest quantile"
    )


def _sample_topological_endcaps_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    farthest_quantile: float,
    lobe_size_ratio: float,
    min_lobe_tokens: int,
    generator: Optional[torch.Generator],
    eligible: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Sample two distant BFS lobes using hop distance only."""
    if not 0.0 <= farthest_quantile < 1.0:
        raise ValueError("farthest_quantile must be within [0, 1)")
    if not 0.0 < lobe_size_ratio < 1.0:
        raise ValueError("lobe_size_ratio must be within (0, 1)")
    first_lobe_size = int(size * lobe_size_ratio)
    second_lobe_size = size - first_lobe_size
    if min(first_lobe_size, second_lobe_size) < min_lobe_tokens:
        raise ValueError(
            f"A {size}-node target split with lobe_size_ratio={lobe_size_ratio} "
            f"produces lobes of {first_lobe_size} and {second_lobe_size} nodes, "
            f"below min_lobe_tokens={min_lobe_tokens}"
        )

    eligible = list(eligible) if eligible is not None else _eligible_seed_nodes(adjacency, size)
    if not eligible:
        raise ValueError(f"No connected component contains the requested {size} nodes")

    seed = _sample_seed(eligible, generator)
    first_lobe = _sample_bfs_region(
        adjacency,
        seed=seed,
        size=first_lobe_size,
        generator=generator,
    )
    first_lobe_set = set(first_lobe)
    distances = _bfs_distances(adjacency, seed)
    unreachable = len(adjacency) + 1
    endpoint_candidates = [
        node
        for node, distance in enumerate(distances)
        if node not in first_lobe_set and distance != unreachable
    ]
    if not endpoint_candidates:
        raise ValueError("No endpoint remains outside the first topological endcap")

    endpoint_candidates.sort(key=lambda node: (-distances[node], node))
    farthest_count = max(
        1,
        int(math.ceil((1.0 - farthest_quantile) * len(endpoint_candidates))),
    )
    farthest_candidates = endpoint_candidates[:farthest_count]
    endpoint_order = torch.randperm(
        len(farthest_candidates),
        generator=generator,
    ).tolist()
    for endpoint_id in endpoint_order:
        second_lobe = _sample_bfs_region(
            adjacency,
            seed=farthest_candidates[endpoint_id],
            size=second_lobe_size,
            generator=generator,
        )
        if first_lobe_set.isdisjoint(second_lobe):
            return torch.tensor(first_lobe + second_lobe, dtype=torch.long)

    raise ValueError(
        "Could not grow two disjoint topological endcaps from the configured farthest quantile"
    )


def _sample_connected_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    size: int,
    growth: GrowthStrategy,
    generator: Optional[torch.Generator],
    eligible: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if growth not in {"dijkstra", "bfs"}:
        raise ValueError(f"Unsupported growth strategy {growth!r}; expected 'dijkstra' or 'bfs'")
    eligible = list(eligible) if eligible is not None else _eligible_seed_nodes(adjacency, size)
    if not eligible:
        raise ValueError(f"No connected component contains the requested {size} nodes")
    seed = _sample_seed(eligible, generator)
    if growth == "bfs":
        selected = _sample_bfs_region(adjacency, seed, size, generator)
    else:
        selected = _sample_dijkstra_region(adjacency, seed, size, generator)
    if len(selected) != size:
        raise RuntimeError(f"Graph traversal returned {len(selected)} nodes instead of {size}")
    return torch.tensor(selected, dtype=torch.long)


def sample_connected_region(
    graph: WeightedSensorGraph,
    size: int,
    growth: GrowthStrategy = "dijkstra",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample exactly ``size`` nodes forming a connected graph region.

    The returned indices retain traversal order. Callers may sort them before
    passing them to an encoder without changing the set-level connectivity.
    """
    size = int(size)
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if size > graph.num_nodes:
        raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
    adjacency = _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
    return _sample_connected_from_adjacency(adjacency, size, growth, generator)


def sample_graph_farthest_points(
    graph: WeightedSensorGraph,
    size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample an exact, spatially dispersed set using weighted graph FPS."""
    size = int(size)
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if size > graph.num_nodes:
        raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
    adjacency = _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
    return _sample_graph_farthest_points_from_adjacency(
        adjacency,
        size=size,
        generator=generator,
    )


def sample_stratified_random_region(
    node_group_id,
    size: int,
    *,
    min_per_group: int = 1,
    remainder_allocation: str = "proportional",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample without replacement while covering every sensor group."""
    group_ids = _as_cpu_tensor(node_group_id, dtype=torch.long).reshape(-1)
    size = int(size)
    min_per_group = int(min_per_group)
    if not 0 < size <= group_ids.numel():
        raise ValueError(f"Cannot sample {size} nodes from {group_ids.numel()} node groups")
    if min_per_group <= 0:
        raise ValueError("min_per_group must be positive")
    if remainder_allocation != "proportional":
        raise ValueError("stratified_random supports only remainder_allocation='proportional'")

    groups = torch.unique(group_ids, sorted=True)
    required = int(groups.numel()) * min_per_group
    if size < required:
        raise ValueError(
            f"stratified_random size {size} cannot cover {groups.numel()} groups "
            f"with min_per_group={min_per_group}; need at least {required}"
        )

    selected_parts: list[torch.Tensor] = []
    available = torch.ones(group_ids.numel(), dtype=torch.bool)
    for group in groups:
        nodes = torch.nonzero(group_ids == group, as_tuple=False).reshape(-1)
        if nodes.numel() < min_per_group:
            raise ValueError(
                f"Sensor group {int(group)} contains {nodes.numel()} nodes, "
                f"below min_per_group={min_per_group}"
            )
        mandatory = nodes[
            torch.randperm(nodes.numel(), generator=generator)[:min_per_group]
        ]
        selected_parts.append(mandatory)
        available[mandatory] = False

    remaining_size = size - required
    if remaining_size:
        remaining_nodes = torch.nonzero(available, as_tuple=False).reshape(-1)
        selected_parts.append(
            remaining_nodes[
                torch.randperm(remaining_nodes.numel(), generator=generator)[:remaining_size]
            ]
        )
    selected = torch.cat(selected_parts)
    return selected[torch.randperm(selected.numel(), generator=generator)]


def _prepare_stratified_connected_lobe_plan(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    node_group_id,
    size: int,
    *,
    min_per_group: int = 1,
    remainder_allocation: str = "equal",
) -> _StratifiedConnectedLobePlan:
    """Prepare group subgraphs and eligible seeds without consuming RNG."""
    group_ids = _as_cpu_tensor(node_group_id, dtype=torch.long).reshape(-1)
    size = int(size)
    min_per_group = int(min_per_group)
    if group_ids.numel() != len(adjacency):
        raise ValueError("node_group_id length must match graph.num_nodes")
    if not 0 < size <= group_ids.numel():
        raise ValueError(f"Cannot sample {size} nodes from a {group_ids.numel()}-node graph")
    if min_per_group <= 0:
        raise ValueError("min_per_group must be positive")
    if remainder_allocation != "equal":
        raise ValueError(
            "stratified_connected_lobes supports only remainder_allocation='equal'"
        )

    groups = torch.unique(group_ids, sorted=True)
    num_groups = int(groups.numel())
    required = num_groups * min_per_group
    if size < required:
        raise ValueError(
            f"stratified_connected_lobes size {size} cannot cover {num_groups} groups "
            f"with min_per_group={min_per_group}; need at least {required}"
        )

    base_size, remainder = divmod(size, num_groups)
    if base_size < min_per_group:
        raise ValueError(
            f"Equal lobe allocation gives only {base_size} nodes per group, "
            f"below min_per_group={min_per_group}"
        )

    induced_adjacencies = []
    eligible_by_quota = []
    group_id_values = group_ids.tolist()
    possible_quotas = {base_size}
    if remainder:
        possible_quotas.add(base_size + 1)
    for group in groups:
        group_mask = group_ids == group
        group_nodes = torch.nonzero(group_mask, as_tuple=False).reshape(-1)
        group_node_values = group_nodes.tolist()
        group_value = int(group)
        max_quota = max(possible_quotas)
        if group_nodes.numel() < max_quota:
            raise ValueError(
                f"Node group {int(group)} contains {group_nodes.numel()} nodes, "
                f"below its equal lobe quota {max_quota}"
            )

        induced_adjacency: list[list[tuple[int, float]]] = [
            [] for _ in range(len(adjacency))
        ]
        for node in group_node_values:
            induced_adjacency[node] = [
                (neighbor, weight)
                for neighbor, weight in adjacency[node]
                if group_id_values[neighbor] == group_value
            ]
        components = _connected_components_for_nodes(
            induced_adjacency,
            group_node_values,
        )
        group_eligible_by_quota = {}
        for quota in possible_quotas:
            eligible = _eligible_seed_nodes(
                induced_adjacency,
                quota,
                components,
            )
            if not eligible:
                raise ValueError(
                    f"Node group {int(group)} has no connected component "
                    f"large enough for a {quota}-node lobe"
                )
            group_eligible_by_quota[quota] = eligible
        induced_adjacencies.append(induced_adjacency)
        eligible_by_quota.append(group_eligible_by_quota)

    return _StratifiedConnectedLobePlan(
        group_ids=group_ids,
        groups=groups,
        base_size=base_size,
        remainder=remainder,
        induced_adjacencies=tuple(induced_adjacencies),
        eligible_by_quota=tuple(eligible_by_quota),
    )


def _sample_stratified_connected_lobes_from_adjacency(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    node_group_id,
    size: int,
    *,
    growth: GrowthStrategy = "dijkstra",
    min_per_group: int = 1,
    remainder_allocation: str = "equal",
    generator: Optional[torch.Generator] = None,
    prepared_plan: Optional[_StratifiedConnectedLobePlan] = None,
) -> torch.Tensor:
    """Sample one connected lobe inside every node group.

    The overall target is generally disconnected. Its exact token budget is
    divided as evenly as possible between groups; randomized remainder
    assignment prevents the same groups from always receiving the extra token.
    """
    plan = prepared_plan or _prepare_stratified_connected_lobe_plan(
        adjacency,
        node_group_id,
        size,
        min_per_group=min_per_group,
        remainder_allocation=remainder_allocation,
    )
    quotas = torch.full(
        (int(plan.groups.numel()),),
        plan.base_size,
        dtype=torch.long,
    )
    if plan.remainder:
        extra_order = torch.randperm(int(plan.groups.numel()), generator=generator)
        quotas[extra_order[: plan.remainder]] += 1

    lobes: list[torch.Tensor] = []
    for group_index, group in enumerate(plan.groups):
        quota = int(quotas[group_index].item())
        lobes.append(
            _sample_connected_from_adjacency(
                plan.induced_adjacencies[group_index],
                quota,
                growth,
                generator,
                plan.eligible_by_quota[group_index][quota],
            )
        )

    selected = torch.cat(lobes)
    if selected.unique().numel() != int(size):
        raise RuntimeError(
            "stratified_connected_lobes returned duplicate nodes or an incorrect size"
        )
    return selected


def sample_stratified_connected_lobes(
    graph: WeightedSensorGraph,
    node_group_id,
    size: int,
    *,
    growth: GrowthStrategy = "dijkstra",
    min_per_group: int = 1,
    remainder_allocation: str = "equal",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample an exact target containing one connected lobe per node group."""
    adjacency = _build_undirected_adjacency(
        graph.num_nodes,
        graph.edge_index,
        graph.edge_weight,
    )
    return _sample_stratified_connected_lobes_from_adjacency(
        adjacency,
        node_group_id,
        size,
        growth=growth,
        min_per_group=min_per_group,
        remainder_allocation=remainder_allocation,
        generator=generator,
    )


def sample_geodesic_corridor(
    graph: WeightedSensorGraph,
    size: int,
    farthest_quantile: float = 0.8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a long connected target around a weighted shortest path."""
    size = int(size)
    if size <= 1:
        raise ValueError(f"Geodesic corridors require size >= 2; got {size}")
    if size > graph.num_nodes:
        raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
    adjacency = _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
    return _sample_geodesic_corridor_from_adjacency(
        adjacency,
        size=size,
        farthest_quantile=float(farthest_quantile),
        generator=generator,
    )


def sample_geodesic_endcaps(
    graph: WeightedSensorGraph,
    size: int,
    farthest_quantile: float = 0.8,
    lobe_size_ratio: float = 0.5,
    min_lobe_tokens: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one global target as two distant, connected, disjoint lobes."""
    size = int(size)
    min_lobe_tokens = int(min_lobe_tokens)
    if size <= 1:
        raise ValueError(f"Geodesic endcaps require size >= 2; got {size}")
    if size > graph.num_nodes:
        raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
    if min_lobe_tokens <= 0:
        raise ValueError("min_lobe_tokens must be positive")
    adjacency = _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
    return _sample_geodesic_endcaps_from_adjacency(
        adjacency,
        size=size,
        farthest_quantile=float(farthest_quantile),
        lobe_size_ratio=float(lobe_size_ratio),
        min_lobe_tokens=min_lobe_tokens,
        generator=generator,
    )


def sample_topological_endcaps(
    graph: WeightedSensorGraph,
    size: int,
    farthest_quantile: float = 0.8,
    lobe_size_ratio: float = 0.5,
    min_lobe_tokens: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one global target as two distant BFS lobes."""
    size = int(size)
    min_lobe_tokens = int(min_lobe_tokens)
    if size <= 1:
        raise ValueError(f"Topological endcaps require size >= 2; got {size}")
    if size > graph.num_nodes:
        raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
    if min_lobe_tokens <= 0:
        raise ValueError("min_lobe_tokens must be positive")
    adjacency = _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
    return _sample_topological_endcaps_from_adjacency(
        adjacency,
        size=size,
        farthest_quantile=float(farthest_quantile),
        lobe_size_ratio=float(lobe_size_ratio),
        min_lobe_tokens=min_lobe_tokens,
        generator=generator,
    )


def _sample_region(
    graph: WeightedSensorGraph,
    size: int,
    strategy: TargetRegionStrategy,
    growth: GrowthStrategy,
    generator: Optional[torch.Generator],
    adjacency: Optional[Sequence[Sequence[tuple[int, float]]]] = None,
    eligible: Optional[Sequence[int]] = None,
    farthest_quantile: float = 0.8,
    lobe_size_ratio: float = 0.5,
    min_lobe_tokens: int = 1,
    node_group_id=None,
    min_per_group: int = 1,
    remainder_allocation: str = "proportional",
    stratified_connected_plan: Optional[_StratifiedConnectedLobePlan] = None,
) -> torch.Tensor:
    if strategy == "connected_region":
        if adjacency is not None:
            return _sample_connected_from_adjacency(
                adjacency,
                size,
                growth,
                generator,
                eligible,
            )
        return sample_connected_region(graph, size=size, growth=growth, generator=generator)
    if strategy == "random":
        if not 0 < size <= graph.num_nodes:
            raise ValueError(f"Cannot sample {size} nodes from a {graph.num_nodes}-node graph")
        return torch.randperm(graph.num_nodes, generator=generator)[:size]
    if strategy == "stratified_random":
        if node_group_id is None:
            raise ValueError(
                "stratified_random requires graph_info['node_group_id'] "
                "with one sensor-panel ID per node"
            )
        if _as_cpu_tensor(node_group_id, dtype=torch.long).numel() != graph.num_nodes:
            raise ValueError(
                "stratified_random node_group_id length must match graph.num_nodes"
            )
        return sample_stratified_random_region(
            node_group_id,
            size,
            min_per_group=min_per_group,
            remainder_allocation=remainder_allocation,
            generator=generator,
        )
    if strategy == "stratified_connected_lobes":
        if node_group_id is None:
            raise ValueError(
                "stratified_connected_lobes requires graph_info['node_group_id'] "
                "with one group ID per node"
            )
        if _as_cpu_tensor(node_group_id, dtype=torch.long).numel() != graph.num_nodes:
            raise ValueError(
                "stratified_connected_lobes node_group_id length must match graph.num_nodes"
            )
        if adjacency is not None:
            return _sample_stratified_connected_lobes_from_adjacency(
                adjacency,
                node_group_id,
                size,
                growth=growth,
                min_per_group=min_per_group,
                remainder_allocation=remainder_allocation,
                generator=generator,
                prepared_plan=stratified_connected_plan,
            )
        return sample_stratified_connected_lobes(
            graph,
            node_group_id,
            size,
            growth=growth,
            min_per_group=min_per_group,
            remainder_allocation=remainder_allocation,
            generator=generator,
        )
    if strategy == "graph_farthest_points":
        if growth != "dijkstra":
            raise ValueError("graph_farthest_points requires growth='dijkstra'")
        if adjacency is not None:
            return _sample_graph_farthest_points_from_adjacency(
                adjacency,
                size=size,
                generator=generator,
                eligible=eligible,
            )
        return sample_graph_farthest_points(
            graph,
            size=size,
            generator=generator,
        )
    if strategy == "geodesic_corridor":
        if growth != "dijkstra":
            raise ValueError("geodesic_corridor currently requires growth='dijkstra'")
        if adjacency is not None:
            return _sample_geodesic_corridor_from_adjacency(
                adjacency,
                size=size,
                farthest_quantile=farthest_quantile,
                generator=generator,
                eligible=eligible,
            )
        return sample_geodesic_corridor(
            graph,
            size=size,
            farthest_quantile=farthest_quantile,
            generator=generator,
        )
    if strategy == "geodesic_endcaps":
        if growth != "dijkstra":
            raise ValueError("geodesic_endcaps currently requires growth='dijkstra'")
        if adjacency is not None:
            return _sample_geodesic_endcaps_from_adjacency(
                adjacency,
                size=size,
                farthest_quantile=farthest_quantile,
                lobe_size_ratio=lobe_size_ratio,
                min_lobe_tokens=min_lobe_tokens,
                generator=generator,
                eligible=eligible,
            )
        return sample_geodesic_endcaps(
            graph,
            size=size,
            farthest_quantile=farthest_quantile,
            lobe_size_ratio=lobe_size_ratio,
            min_lobe_tokens=min_lobe_tokens,
            generator=generator,
        )
    if strategy == "topological_endcaps":
        if growth != "bfs":
            raise ValueError("topological_endcaps requires growth='bfs'")
        if adjacency is not None:
            return _sample_topological_endcaps_from_adjacency(
                adjacency,
                size=size,
                farthest_quantile=farthest_quantile,
                lobe_size_ratio=lobe_size_ratio,
                min_lobe_tokens=min_lobe_tokens,
                generator=generator,
                eligible=eligible,
            )
        return sample_topological_endcaps(
            graph,
            size=size,
            farthest_quantile=farthest_quantile,
            lobe_size_ratio=lobe_size_ratio,
            min_lobe_tokens=min_lobe_tokens,
            generator=generator,
        )
    raise ValueError(
        "Unsupported region strategy "
        f"{strategy!r}; expected 'connected_region', 'random', 'stratified_random', "
        "'stratified_connected_lobes', "
        "'graph_farthest_points', "
        "'geodesic_corridor', 'geodesic_endcaps', or 'topological_endcaps'"
    )


def _normalize_target_groups(
    *,
    num_target_masks: int,
    target_strategy: TargetRegionStrategy,
    target_growth: GrowthStrategy,
    target_groups: Optional[Sequence[Mapping]],
) -> list[dict]:
    if target_groups is None:
        return [
            {
                "count": num_target_masks,
                "strategy": target_strategy,
                "growth": target_growth,
                "farthest_quantile": 0.8,
                "lobe_size_ratio": 0.5,
                "min_lobe_tokens": 1,
                "min_per_group": 1,
                "remainder_allocation": (
                    "equal"
                    if target_strategy == "stratified_connected_lobes"
                    else "proportional"
                ),
                "max_pairwise_overlap_ratio": 1.0,
                "max_previous_overlap_ratio": 1.0,
                "scale": None,
            }
        ]
    if not isinstance(target_groups, Sequence) or isinstance(target_groups, (str, bytes)):
        raise ValueError("target.groups must be a sequence of mappings")

    normalized = []
    for group_id, group in enumerate(target_groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"target.groups[{group_id}] must be a mapping")
        count = int(group.get("count", 0))
        if count <= 0:
            raise ValueError(f"target.groups[{group_id}].count must be positive")
        strategy = str(group.get("strategy", target_strategy))
        growth = str(group.get("growth", target_growth))
        endpoint_sampling = str(group.get("endpoint_sampling", "farthest_quantile"))
        farthest_quantile = float(group.get("farthest_quantile", 0.8))
        lobe_size_ratio = float(group.get("lobe_size_ratio", 0.5))
        min_lobe_tokens = int(group.get("min_lobe_tokens", 1))
        min_per_group = int(group.get("min_per_group", 1))
        remainder_allocation = str(
            group.get(
                "remainder_allocation",
                "equal"
                if strategy == "stratified_connected_lobes"
                else "proportional",
            )
        )
        max_overlap = float(group.get("max_pairwise_overlap_ratio", 1.0))
        max_previous_overlap = float(group.get("max_previous_overlap_ratio", 1.0))
        raw_scale = group.get("scale")
        if raw_scale is None:
            scale = None
        else:
            if not isinstance(raw_scale, Sequence) or isinstance(raw_scale, (str, bytes)):
                raise ValueError(f"target.groups[{group_id}].scale must contain two values")
            scale = tuple(float(value) for value in raw_scale)
            if len(scale) != 2:
                raise ValueError(f"target.groups[{group_id}].scale must contain two values")
            if not 0.0 < scale[0] <= scale[1] <= 1.0:
                raise ValueError(
                    f"target.groups[{group_id}].scale must satisfy 0 < min <= max <= 1"
                )
        if strategy not in {
            "connected_region",
            "random",
            "stratified_random",
            "stratified_connected_lobes",
            "graph_farthest_points",
            "geodesic_corridor",
            "geodesic_endcaps",
            "topological_endcaps",
        }:
            raise ValueError(
                f"Unsupported target strategy {strategy!r} in target.groups[{group_id}]"
            )
        if growth not in {"dijkstra", "bfs"}:
            raise ValueError(f"Unsupported target growth {growth!r} in target.groups[{group_id}]")
        if strategy in {
            "geodesic_corridor",
            "geodesic_endcaps",
            "topological_endcaps",
        } and (
            endpoint_sampling != "farthest_quantile"
        ):
            raise ValueError(
                f"{strategy} supports only endpoint_sampling='farthest_quantile'"
            )
        if strategy in {"geodesic_corridor", "geodesic_endcaps"} and growth != "dijkstra":
            raise ValueError(f"{strategy} currently requires growth='dijkstra'")
        if strategy == "graph_farthest_points" and growth != "dijkstra":
            raise ValueError("graph_farthest_points requires growth='dijkstra'")
        if strategy == "topological_endcaps" and growth != "bfs":
            raise ValueError("topological_endcaps requires growth='bfs'")
        if not 0.0 <= farthest_quantile < 1.0:
            raise ValueError("farthest_quantile must be within [0, 1)")
        if not 0.0 < lobe_size_ratio < 1.0:
            raise ValueError("lobe_size_ratio must be within (0, 1)")
        if min_lobe_tokens <= 0:
            raise ValueError("min_lobe_tokens must be positive")
        if min_per_group <= 0:
            raise ValueError("min_per_group must be positive")
        if strategy == "stratified_random" and remainder_allocation != "proportional":
            raise ValueError(
                "stratified_random supports only remainder_allocation='proportional'"
            )
        if (
            strategy == "stratified_connected_lobes"
            and remainder_allocation != "equal"
        ):
            raise ValueError(
                "stratified_connected_lobes supports only remainder_allocation='equal'"
            )
        if not 0.0 <= max_overlap <= 1.0:
            raise ValueError("max_pairwise_overlap_ratio must be within [0, 1]")
        if not 0.0 <= max_previous_overlap <= 1.0:
            raise ValueError("max_previous_overlap_ratio must be within [0, 1]")
        normalized.append(
            {
                "count": count,
                "strategy": strategy,
                "growth": growth,
                "farthest_quantile": farthest_quantile,
                "lobe_size_ratio": lobe_size_ratio,
                "min_lobe_tokens": min_lobe_tokens,
                "min_per_group": min_per_group,
                "remainder_allocation": remainder_allocation,
                "max_pairwise_overlap_ratio": max_overlap,
                "max_previous_overlap_ratio": max_previous_overlap,
                "scale": scale,
            }
        )
    total = sum(group["count"] for group in normalized)
    if total != num_target_masks:
        raise ValueError(
            f"target.groups counts sum to {total}, but num_target_masks={num_target_masks}"
        )
    return normalized


def _batched_graphs(
    graph_info: Mapping,
    batch_size: int,
    num_nodes: int,
    *,
    require_edge_weight: bool,
) -> list[WeightedSensorGraph]:
    if "edge_index" not in graph_info:
        raise ValueError(
            "multiblock_graph masking requires graph_info['edge_index']; "
            "use a per-window graph topology"
        )
    if "edge_count" not in graph_info:
        raise ValueError("multiblock_graph masking requires graph_info['edge_count']")
    if require_edge_weight and "edge_attr" not in graph_info:
        raise ValueError("multiblock_graph masking requires graph_info['edge_attr'] for Dijkstra growth")

    edge_index = _as_cpu_tensor(graph_info["edge_index"], dtype=torch.long)
    edge_count = _as_cpu_tensor(graph_info["edge_count"], dtype=torch.long).reshape(-1)
    if edge_index.ndim == 2:
        edge_index = edge_index.unsqueeze(0)
    if edge_index.ndim != 3 or edge_index.shape[1] != 2:
        raise ValueError(
            "Batched edge_index must have shape (B, 2, E); "
            f"got {tuple(edge_index.shape)}"
        )
    if require_edge_weight:
        edge_attr = _as_cpu_tensor(graph_info["edge_attr"], dtype=torch.float32)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(0).unsqueeze(-1)
        elif edge_attr.ndim == 2:
            if edge_attr.shape[0] == batch_size:
                edge_attr = edge_attr.unsqueeze(-1)
            elif batch_size == 1:
                edge_attr = edge_attr.unsqueeze(0)
        if edge_attr.ndim != 3 or edge_attr.shape[-1] != 1:
            raise ValueError(
                "Batched edge_attr must have shape (B, E, 1); "
                f"got {tuple(edge_attr.shape)}"
            )
    else:
        # BFS and random sampling are deliberately independent of edge attributes.
        edge_attr = torch.ones(
            (edge_index.shape[0], edge_index.shape[-1], 1),
            dtype=torch.float32,
        )
    if (
        edge_index.shape[0] != batch_size
        or edge_attr.shape[0] != batch_size
        or edge_count.numel() != batch_size
    ):
        raise ValueError(
            "Graph batch size does not match sensor batch size: "
            f"edge_index={edge_index.shape[0]}, edge_attr={edge_attr.shape[0]}, "
            f"edge_count={edge_count.numel()}, batch={batch_size}"
        )

    graphs = []
    for batch_id in range(batch_size):
        count = int(edge_count[batch_id])
        if count < 0 or count > edge_index.shape[-1] or count > edge_attr.shape[1]:
            raise ValueError(f"Invalid edge_count={count} for graph {batch_id}")
        graphs.append(
            WeightedSensorGraph(
                edge_index=edge_index[batch_id, :, :count].numpy(),
                edge_weight=edge_attr[batch_id, :count].reshape(-1).numpy(),
                num_nodes=num_nodes,
            )
        )
    return graphs


def _batched_node_group_ids(
    graph_info: Mapping,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    if "node_group_id" not in graph_info:
        raise ValueError(
            "stratified_random requires graph_info['node_group_id'] "
            "with one sensor-panel ID per node"
        )
    group_ids = _as_cpu_tensor(graph_info["node_group_id"], dtype=torch.long)
    if group_ids.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                "Unbatched node_group_id is valid only when batch_size=1"
            )
        group_ids = group_ids.unsqueeze(0)
    if tuple(group_ids.shape) != (batch_size, num_nodes):
        raise ValueError(
            "Batched node_group_id must have shape "
            f"({batch_size}, {num_nodes}); got {tuple(group_ids.shape)}"
        )
    return group_ids


def sample_multiblock_graph_masks(
    *,
    graph_info: Mapping,
    batch_size: int,
    num_nodes: int,
    context_size: int,
    target_size: int,
    num_context_masks: int,
    num_target_masks: int,
    context_strategy: RegionStrategy,
    target_strategy: TargetRegionStrategy,
    context_growth: GrowthStrategy,
    target_growth: GrowthStrategy,
    min_context_keep_tokens: int,
    min_context_keep_ratio: float,
    max_resample_attempts: int,
    target_groups: Optional[Sequence[Mapping]] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]:
    """Sample I-JEPA-style graph regions for a padded batch of sensor graphs."""
    if num_context_masks <= 0 or num_target_masks <= 0:
        raise ValueError("num_context_masks and num_target_masks must be positive")
    if max_resample_attempts <= 0:
        raise ValueError("max_resample_attempts must be positive")
    if not 0.0 <= min_context_keep_ratio <= 1.0:
        raise ValueError("min_context_keep_ratio must be within [0, 1]")

    min_context_keep = max(
        int(min_context_keep_tokens),
        int(math.ceil(float(min_context_keep_ratio) * num_nodes)),
    )
    if min_context_keep <= 0:
        raise ValueError("The effective minimum context size must be positive")
    if context_size < min_context_keep:
        raise ValueError(
            f"Raw context size {context_size} is smaller than required minimum {min_context_keep}"
        )

    normalized_target_groups = _normalize_target_groups(
        num_target_masks=num_target_masks,
        target_strategy=target_strategy,
        target_growth=target_growth,
        target_groups=target_groups,
    )
    for group in normalized_target_groups:
        group_scale = group["scale"]
        group_target_size = (
            target_size
            if group_scale is None
            else sample_block_size_1d(num_nodes, group_scale, generator=generator)[0]
        )
        if not 0 < group_target_size <= num_nodes:
            raise ValueError(
                f"Target group scale produced invalid size {group_target_size} "
                f"for a {num_nodes}-node graph"
            )
        group["target_size"] = group_target_size
    require_edge_weight = (
        context_strategy == "connected_region" and context_growth == "dijkstra"
    ) or any(
        group["strategy"]
        in {
            "graph_farthest_points",
            "geodesic_corridor",
            "geodesic_endcaps",
        }
        or (
            group["strategy"] == "stratified_connected_lobes"
            and group["growth"] == "dijkstra"
        )
        or (group["strategy"] == "connected_region" and group["growth"] == "dijkstra")
        for group in normalized_target_groups
    )
    graphs = _batched_graphs(
        graph_info,
        batch_size=batch_size,
        num_nodes=num_nodes,
        require_edge_weight=require_edge_weight,
    )
    uses_node_groups = any(
        group["strategy"]
        in {"stratified_random", "stratified_connected_lobes"}
        for group in normalized_target_groups
    )
    batched_node_group_ids = (
        _batched_node_group_ids(graph_info, batch_size, num_nodes)
        if uses_node_groups
        else None
    )
    all_contexts: list[list[torch.Tensor]] = []
    all_targets: list[list[torch.Tensor]] = []

    for sample_id, graph in enumerate(graphs):
        uses_connected_regions = (
            context_strategy == "connected_region"
            or any(
                group["strategy"]
                in {
                    "connected_region",
                    "stratified_connected_lobes",
                    "graph_farthest_points",
                    "geodesic_corridor",
                    "geodesic_endcaps",
                    "topological_endcaps",
                }
                for group in normalized_target_groups
            )
        )
        adjacency = (
            _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
            if uses_connected_regions
            else None
        )
        eligible_by_size = {}
        stratified_plans = {}
        if adjacency is not None:
            components = _connected_components(adjacency)
            if context_strategy == "connected_region":
                eligible_by_size[context_size] = _eligible_seed_nodes(
                    adjacency,
                    context_size,
                    components,
                )
            connected_target_sizes = {
                group["target_size"]
                for group in normalized_target_groups
                if group["strategy"]
                in {
                    "connected_region",
                    "graph_farthest_points",
                    "geodesic_corridor",
                    "geodesic_endcaps",
                    "topological_endcaps",
                }
            }
            for connected_target_size in connected_target_sizes:
                eligible_by_size[connected_target_size] = _eligible_seed_nodes(
                    adjacency,
                    connected_target_size,
                    components,
                )
            if batched_node_group_ids is not None:
                for group_id, group in enumerate(normalized_target_groups):
                    if group["strategy"] != "stratified_connected_lobes":
                        continue
                    stratified_plans[group_id] = (
                        _prepare_stratified_connected_lobe_plan(
                            adjacency,
                            batched_node_group_ids[sample_id],
                            group["target_size"],
                            min_per_group=group["min_per_group"],
                            remainder_allocation=group["remainder_allocation"],
                        )
                    )

        accepted_contexts: Optional[list[torch.Tensor]] = None
        accepted_targets: Optional[list[torch.Tensor]] = None
        for _ in range(max_resample_attempts):
            targets: list[torch.Tensor] = []
            targets_valid = True
            for group_id, group in enumerate(normalized_target_groups):
                group_targets: list[torch.Tensor] = []
                group_target_size = group["target_size"]
                for _ in range(group["count"]):
                    accepted_target = None
                    for _ in range(max_resample_attempts):
                        candidate = _sample_region(
                            graph,
                            group_target_size,
                            group["strategy"],
                            group["growth"],
                            generator,
                            adjacency,
                            eligible_by_size.get(group_target_size),
                            group["farthest_quantile"],
                            group["lobe_size_ratio"],
                            group["min_lobe_tokens"],
                            (
                                batched_node_group_ids[sample_id]
                                if batched_node_group_ids is not None
                                else None
                            ),
                            group["min_per_group"],
                            group["remainder_allocation"],
                            stratified_plans.get(group_id),
                        )
                        max_overlap = group["max_pairwise_overlap_ratio"]
                        pairwise_overlap_valid = max_overlap >= 1.0 or all(
                            torch.isin(candidate, previous).sum().item()
                            / float(candidate.numel())
                            <= max_overlap
                            for previous in group_targets
                        )
                        previous_overlap_valid = (
                            group["max_previous_overlap_ratio"] >= 1.0
                            or all(
                                torch.isin(candidate, previous).sum().item()
                                / float(candidate.numel())
                                <= group["max_previous_overlap_ratio"]
                                for previous in targets
                            )
                        )
                        if pairwise_overlap_valid and previous_overlap_valid:
                            accepted_target = candidate
                            break
                    if accepted_target is None:
                        targets_valid = False
                        break
                    group_targets.append(accepted_target)
                if not targets_valid:
                    break
                targets.extend(group_targets)
            if not targets_valid:
                continue

            target_union = torch.zeros(num_nodes, dtype=torch.bool)
            for target in targets:
                target_union[target] = True

            contexts: list[torch.Tensor] = []
            layout_valid = True
            for _ in range(num_context_masks):
                final_context = None
                for _ in range(max_resample_attempts):
                    raw_context = _sample_region(
                        graph,
                        context_size,
                        context_strategy,
                        context_growth,
                        generator,
                        adjacency,
                        eligible_by_size.get(context_size),
                    )
                    candidate = raw_context[~target_union[raw_context]]
                    if candidate.numel() >= min_context_keep:
                        final_context = candidate
                        break
                if final_context is None:
                    layout_valid = False
                    break
                contexts.append(final_context)

            if layout_valid:
                accepted_contexts = contexts
                accepted_targets = targets
                break

        if accepted_contexts is None or accepted_targets is None:
            raise RuntimeError(
                f"Could not sample a valid graph-mask layout for batch item {sample_id} "
                f"after {max_resample_attempts} attempts"
            )
        all_contexts.append(accepted_contexts)
        all_targets.append(accepted_targets)

    common_context_size = min(context.numel() for contexts in all_contexts for context in contexts)
    if common_context_size < min_context_keep:
        raise RuntimeError(
            f"Common context size {common_context_size} fell below required minimum {min_context_keep}"
        )

    context_masks = torch.stack(
        [
            torch.stack(
                [
                    torch.sort(
                        all_contexts[batch_id][context_id][:common_context_size]
                    ).values
                    for batch_id in range(batch_size)
                ]
            )
            for context_id in range(num_context_masks)
        ]
    )
    target_masks_by_id = [
        torch.stack(
            [torch.sort(all_targets[batch_id][target_id]).values for batch_id in range(batch_size)]
        )
        for target_id in range(num_target_masks)
    ]
    target_sizes = {mask.shape[-1] for mask in target_masks_by_id}
    if len(target_sizes) == 1:
        target_masks: Union[torch.Tensor, list[torch.Tensor]] = torch.stack(
            target_masks_by_id
        ).to(device)
    else:
        target_masks = [mask.to(device) for mask in target_masks_by_id]
    return context_masks.to(device), target_masks


class JEPAGraphMaskCollator:
    """Build graph masks in DataLoader workers before Fabric moves a batch to GPU."""

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        num_context_masks: int,
        num_target_masks: int,
        masking: Mapping,
    ) -> None:
        self.context_mask_scale = tuple(float(value) for value in context_mask_scale)
        self.target_mask_scale = tuple(float(value) for value in target_mask_scale)
        if len(self.context_mask_scale) != 2 or len(self.target_mask_scale) != 2:
            raise ValueError("JEPA mask scales must each contain exactly two values")
        self.num_context_masks = int(num_context_masks)
        self.num_target_masks = int(num_target_masks)
        self.masking = masking
        if str(masking.get("mode", "legacy")) != "multiblock_graph":
            raise ValueError("JEPAGraphMaskCollator requires masking.mode=multiblock_graph")
        self._generator: Optional[torch.Generator] = None
        self._generator_seed: Optional[int] = None

    def _worker_generator(self) -> torch.Generator:
        worker_info = get_worker_info()
        seed = int(worker_info.seed if worker_info is not None else torch.initial_seed())
        if self._generator is None or self._generator_seed != seed:
            self._generator = torch.Generator().manual_seed(seed)
            self._generator_seed = seed
        return self._generator

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        batch = default_collate(samples)
        graph_info = batch.pop("graph", None)
        if graph_info is None:
            raise ValueError("JEPAGraphMaskCollator requires every sample to contain a graph")
        if "sensor" not in batch:
            raise ValueError("JEPAGraphMaskCollator requires every sample to contain sensor data")

        sensor = batch["sensor"]
        batch_size, _, num_nodes, _ = sensor.shape
        generator = self._worker_generator()
        context_size = sample_block_size_1d(
            num_nodes,
            self.context_mask_scale,
            generator=generator,
        )[0]
        target_size = sample_block_size_1d(
            num_nodes,
            self.target_mask_scale,
            generator=generator,
        )[0]

        context_cfg = self.masking.get("context", {})
        target_cfg = self.masking.get("target", {})
        overlap_cfg = self.masking.get("overlap", {})
        expected_overlap = {
            "target_target": "allow",
            "context_target": "subtract_from_context",
            "context_context": "allow",
        }
        actual_overlap = {
            key: str(overlap_cfg.get(key, value)) for key, value in expected_overlap.items()
        }
        if actual_overlap != expected_overlap:
            raise ValueError(
                "JEPAGraphMaskCollator supports only I-JEPA overlap semantics: "
                f"{expected_overlap}; got {actual_overlap}"
            )

        context_masks, target_masks = sample_multiblock_graph_masks(
            graph_info=graph_info,
            batch_size=batch_size,
            num_nodes=num_nodes,
            context_size=context_size,
            target_size=target_size,
            num_context_masks=self.num_context_masks,
            num_target_masks=self.num_target_masks,
            context_strategy=str(context_cfg.get("strategy", "connected_region")),
            target_strategy=str(target_cfg.get("strategy", "connected_region")),
            context_growth=str(context_cfg.get("growth", "dijkstra")),
            target_growth=str(target_cfg.get("growth", "dijkstra")),
            target_groups=target_cfg.get("groups"),
            min_context_keep_tokens=int(self.masking.get("min_context_keep_tokens", 32)),
            min_context_keep_ratio=float(self.masking.get("min_context_keep_ratio", 0.15)),
            max_resample_attempts=int(self.masking.get("max_resample_attempts", 32)),
            generator=generator,
        )
        batch["context_masks"] = context_masks
        batch["target_masks"] = target_masks
        return batch

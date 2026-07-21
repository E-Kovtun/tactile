import heapq
import math
from collections.abc import Mapping, Sequence
from typing import Literal, Optional

import torch

from tactile_ssl.graph.types import WeightedSensorGraph


GrowthStrategy = Literal["dijkstra", "bfs"]
RegionStrategy = Literal["connected_region", "random"]


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
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError(f"edge_index values must be within 0..{num_nodes - 1}")
    if edge_weight.numel() and not torch.all(torch.isfinite(edge_weight) & (edge_weight > 0)):
        raise ValueError("All graph edge weights must be finite and positive")

    neighbors: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    for edge_id in range(edge_index.shape[1]):
        left = int(edge_index[0, edge_id])
        right = int(edge_index[1, edge_id])
        if left == right:
            continue
        weight = float(edge_weight[edge_id])
        previous = neighbors[left].get(right)
        if previous is None or weight < previous:
            neighbors[left][right] = weight
            neighbors[right][left] = weight
    return [sorted(node_neighbors.items()) for node_neighbors in neighbors]


def _eligible_seed_nodes(adjacency: Sequence[Sequence[tuple[int, float]]], size: int) -> list[int]:
    seen = [False] * len(adjacency)
    eligible: list[int] = []
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
        if len(component) >= size:
            eligible.extend(component)
    return eligible


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


def _sample_region(
    graph: WeightedSensorGraph,
    size: int,
    strategy: RegionStrategy,
    growth: GrowthStrategy,
    generator: Optional[torch.Generator],
    adjacency: Optional[Sequence[Sequence[tuple[int, float]]]] = None,
    eligible: Optional[Sequence[int]] = None,
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
    raise ValueError(
        f"Unsupported region strategy {strategy!r}; expected 'connected_region' or 'random'"
    )


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
    target_strategy: RegionStrategy,
    context_growth: GrowthStrategy,
    target_growth: GrowthStrategy,
    min_context_keep_tokens: int,
    min_context_keep_ratio: float,
    max_resample_attempts: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    require_edge_weight = (
        context_strategy == "connected_region" and context_growth == "dijkstra"
    ) or (target_strategy == "connected_region" and target_growth == "dijkstra")
    graphs = _batched_graphs(
        graph_info,
        batch_size=batch_size,
        num_nodes=num_nodes,
        require_edge_weight=require_edge_weight,
    )
    all_contexts: list[list[torch.Tensor]] = []
    all_targets: list[list[torch.Tensor]] = []

    for sample_id, graph in enumerate(graphs):
        uses_connected_regions = (
            context_strategy == "connected_region" or target_strategy == "connected_region"
        )
        adjacency = (
            _build_undirected_adjacency(graph.num_nodes, graph.edge_index, graph.edge_weight)
            if uses_connected_regions
            else None
        )
        eligible_by_size = {}
        if adjacency is not None:
            if context_strategy == "connected_region":
                eligible_by_size[context_size] = _eligible_seed_nodes(adjacency, context_size)
            if target_strategy == "connected_region":
                eligible_by_size[target_size] = _eligible_seed_nodes(adjacency, target_size)

        accepted_contexts: Optional[list[torch.Tensor]] = None
        accepted_targets: Optional[list[torch.Tensor]] = None
        for _ in range(max_resample_attempts):
            targets = [
                _sample_region(
                    graph,
                    target_size,
                    target_strategy,
                    target_growth,
                    generator,
                    adjacency,
                    eligible_by_size.get(target_size),
                )
                for _ in range(num_target_masks)
            ]
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
    target_masks = torch.stack(
        [
            torch.stack(
                [torch.sort(all_targets[batch_id][target_id]).values for batch_id in range(batch_size)]
            )
            for target_id in range(num_target_masks)
        ]
    )
    return context_masks.to(device), target_masks.to(device)

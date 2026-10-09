import numpy as np
import pytest
import torch

import tactile_ssl.utils.jepa_masking as jepa_masking
from tactile_ssl.graph.types import WeightedSensorGraph
from tactile_ssl.utils.jepa_masking import (
    JEPAGraphMaskCollator,
    sample_graph_farthest_points,
    sample_geodesic_corridor,
    sample_geodesic_endcaps,
    sample_multiblock_graph_masks,
    sample_stratified_connected_lobes,
    sample_stratified_random_region,
    sample_topological_endcaps,
)


def _line_graph(num_nodes: int, *, edge_weight: float = 1.0) -> WeightedSensorGraph:
    left = np.arange(num_nodes - 1, dtype=np.int64)
    right = left + 1
    return WeightedSensorGraph(
        edge_index=np.stack([left, right]),
        edge_weight=np.full(num_nodes - 1, edge_weight, dtype=np.float32),
        num_nodes=num_nodes,
    )


def _batched_graph_info(graph: WeightedSensorGraph, batch_size: int = 1) -> dict:
    edge_index = torch.from_numpy(graph.edge_index).unsqueeze(0).repeat(batch_size, 1, 1)
    edge_attr = (
        torch.from_numpy(graph.edge_weight)
        .reshape(1, -1, 1)
        .repeat(batch_size, 1, 1)
    )
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_count": torch.full((batch_size,), graph.edge_index.shape[1], dtype=torch.long),
    }


def test_stratified_random_is_exact_covers_groups_and_is_deterministic():
    group_ids = torch.repeat_interleave(
        torch.arange(18),
        torch.tensor([30, 16, 16, 30, 16, 16, 16, 30, 16, 16, 16, 30, 16, 16, 16, 24, 24, 24]),
    )
    first = sample_stratified_random_region(
        group_ids,
        size=50,
        generator=torch.Generator().manual_seed(71),
    )
    second = sample_stratified_random_region(
        group_ids,
        size=50,
        generator=torch.Generator().manual_seed(71),
    )

    assert torch.equal(first, second)
    assert first.numel() == first.unique().numel() == 50
    assert torch.equal(torch.unique(group_ids[first]), torch.arange(18))


def test_stratified_random_rejects_insufficient_budget():
    with pytest.raises(ValueError, match="need at least 4"):
        sample_stratified_random_region(
            torch.tensor([0, 0, 1, 1]),
            size=3,
            min_per_group=2,
        )


def test_stratified_connected_lobes_are_exact_balanced_connected_and_deterministic():
    graph = _line_graph(50)
    group_ids = torch.repeat_interleave(torch.arange(5), 10)

    first = sample_stratified_connected_lobes(
        graph,
        group_ids,
        size=23,
        generator=torch.Generator().manual_seed(91),
    )
    second = sample_stratified_connected_lobes(
        graph,
        group_ids,
        size=23,
        generator=torch.Generator().manual_seed(91),
    )

    assert torch.equal(first, second)
    assert first.numel() == first.unique().numel() == 23
    group_counts = torch.bincount(group_ids[first], minlength=5)
    assert int(group_counts.min()) == 4
    assert int(group_counts.max()) == 5
    for group_id in range(5):
        lobe = first[group_ids[first] == group_id]
        assert _is_connected(lobe, graph)


def test_prepared_stratified_plan_preserves_exact_rng_sequence():
    graph = _line_graph(50)
    group_ids = torch.repeat_interleave(torch.arange(5), 10)
    adjacency = jepa_masking._build_undirected_adjacency(
        graph.num_nodes,
        graph.edge_index,
        graph.edge_weight,
    )
    plan = jepa_masking._prepare_stratified_connected_lobe_plan(
        adjacency,
        group_ids,
        size=23,
    )
    cached_generator = torch.Generator().manual_seed(912)
    uncached_generator = torch.Generator().manual_seed(912)

    cached = [
        jepa_masking._sample_stratified_connected_lobes_from_adjacency(
            adjacency,
            group_ids,
            size=23,
            generator=cached_generator,
            prepared_plan=plan,
        )
        for _ in range(4)
    ]
    uncached = [
        jepa_masking._sample_stratified_connected_lobes_from_adjacency(
            adjacency,
            group_ids,
            size=23,
            generator=uncached_generator,
        )
        for _ in range(4)
    ]

    assert all(
        torch.equal(cached_target, uncached_target)
        for cached_target, uncached_target in zip(cached, uncached)
    )


def test_multiblock_prepares_hypertaxel_subgraphs_once_per_sample(monkeypatch):
    graph = _line_graph(50)
    graph_info = _batched_graph_info(graph, batch_size=2)
    graph_info["node_group_id"] = torch.repeat_interleave(
        torch.arange(5),
        10,
    ).unsqueeze(0).repeat(2, 1)
    original_prepare = jepa_masking._prepare_stratified_connected_lobe_plan
    prepare_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        jepa_masking,
        "_prepare_stratified_connected_lobe_plan",
        counted_prepare,
    )
    sample_multiblock_graph_masks(
        graph_info=graph_info,
        batch_size=2,
        num_nodes=50,
        context_size=40,
        target_size=15,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.1,
        max_resample_attempts=32,
        target_groups=[
            {
                "count": 4,
                "strategy": "stratified_connected_lobes",
                "growth": "dijkstra",
                "min_per_group": 1,
                "remainder_allocation": "equal",
            }
        ],
        generator=torch.Generator().manual_seed(913),
    )

    assert prepare_calls == 2


def test_stratified_connected_lobes_reject_disconnected_group_quota():
    graph = WeightedSensorGraph(
        edge_index=np.asarray([[0, 1, 3, 4, 6, 7, 8], [1, 2, 4, 5, 7, 8, 9]]),
        edge_weight=np.ones(7, dtype=np.float32),
        num_nodes=10,
    )
    group_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    with pytest.raises(ValueError, match="Node group 0 has no connected component"):
        sample_stratified_connected_lobes(
            graph,
            group_ids,
            size=8,
            generator=torch.Generator().manual_seed(7),
        )


def test_multiblock_four_stratified_connected_lobe_targets_cover_all_groups():
    graph = _line_graph(50)
    graph_info = _batched_graph_info(graph)
    group_ids = torch.repeat_interleave(torch.arange(5), 10)
    graph_info["node_group_id"] = group_ids.unsqueeze(0)

    contexts, targets = sample_multiblock_graph_masks(
        graph_info=graph_info,
        batch_size=1,
        num_nodes=50,
        context_size=40,
        target_size=15,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {
                "count": 4,
                "strategy": "stratified_connected_lobes",
                "growth": "dijkstra",
                "remainder_allocation": "equal",
            }
        ],
        min_context_keep_tokens=1,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(101),
    )

    assert targets.shape == (4, 1, 15)
    for target in targets[:, 0]:
        assert torch.equal(torch.bincount(group_ids[target], minlength=5), torch.full((5,), 3))
        for group_id in range(5):
            lobe = target[group_ids[target] == group_id]
            assert _is_connected(lobe, graph)
        assert not torch.isin(contexts[0, 0], target).any()


def _is_connected(nodes: torch.Tensor, graph: WeightedSensorGraph) -> bool:
    selected = set(nodes.tolist())
    neighbors = {node: set() for node in selected}
    for left, right in graph.edge_index.T.tolist():
        if left in selected and right in selected:
            neighbors[left].add(right)
            neighbors[right].add(left)
    seen = {next(iter(selected))}
    frontier = list(seen)
    while frontier:
        node = frontier.pop()
        for neighbor in neighbors[node] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    return seen == selected


def test_geodesic_corridor_is_exact_connected_long_and_deterministic():
    graph = _line_graph(20)

    first = sample_geodesic_corridor(
        graph,
        size=7,
        farthest_quantile=0.8,
        generator=torch.Generator().manual_seed(17),
    )
    second = sample_geodesic_corridor(
        graph,
        size=7,
        farthest_quantile=0.8,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first, second)
    assert first.unique().numel() == 7
    assert _is_connected(first, graph)
    assert int(first.max() - first.min()) == 6


def test_geodesic_corridor_uses_a_large_enough_component_and_errors_otherwise():
    graph = WeightedSensorGraph(
        edge_index=np.asarray([[0, 1, 3, 4, 5, 6], [1, 2, 4, 5, 6, 7]]),
        edge_weight=np.ones(6, dtype=np.float32),
        num_nodes=8,
    )
    corridor = sample_geodesic_corridor(
        graph,
        size=4,
        generator=torch.Generator().manual_seed(3),
    )
    assert set(corridor.tolist()).issubset({3, 4, 5, 6, 7})
    assert _is_connected(corridor, graph)

    with pytest.raises(ValueError, match="No connected component"):
        sample_geodesic_corridor(graph, size=6)


def test_geodesic_endcaps_are_exact_disjoint_connected_lobes_and_deterministic():
    graph = _line_graph(30)

    first = sample_geodesic_endcaps(
        graph,
        size=12,
        farthest_quantile=0.8,
        lobe_size_ratio=0.5,
        min_lobe_tokens=6,
        generator=torch.Generator().manual_seed(23),
    )
    second = sample_geodesic_endcaps(
        graph,
        size=12,
        farthest_quantile=0.8,
        lobe_size_ratio=0.5,
        min_lobe_tokens=6,
        generator=torch.Generator().manual_seed(23),
    )

    assert torch.equal(first, second)
    assert first.unique().numel() == 12
    assert _is_connected(first[:6], graph)
    assert _is_connected(first[6:], graph)
    assert not torch.isin(first[:6], first[6:]).any()
    assert int(first.max() - first.min()) >= 20


def test_geodesic_endcaps_validate_minimum_lobe_budget():
    graph = _line_graph(20)
    with pytest.raises(ValueError, match="below min_lobe_tokens=6"):
        sample_geodesic_endcaps(
            graph,
            size=10,
            lobe_size_ratio=0.5,
            min_lobe_tokens=6,
        )


def test_topological_endcaps_are_exact_disjoint_deterministic_and_ignore_weights():
    first_graph = _line_graph(30, edge_weight=1.0)
    second_graph = _line_graph(30, edge_weight=17.0)

    first = sample_topological_endcaps(
        first_graph,
        size=12,
        farthest_quantile=0.8,
        lobe_size_ratio=0.5,
        min_lobe_tokens=6,
        generator=torch.Generator().manual_seed(41),
    )
    second = sample_topological_endcaps(
        second_graph,
        size=12,
        farthest_quantile=0.8,
        lobe_size_ratio=0.5,
        min_lobe_tokens=6,
        generator=torch.Generator().manual_seed(41),
    )

    assert torch.equal(first, second)
    assert first.unique().numel() == 12
    assert _is_connected(first[:6], first_graph)
    assert _is_connected(first[6:], first_graph)
    assert not torch.isin(first[:6], first[6:]).any()
    inter_lobe_distance = torch.cdist(
        first[:6, None].float(),
        first[6:, None].float(),
        p=1,
    ).min()
    assert int(inter_lobe_distance) >= 8


def test_graph_farthest_points_are_exact_dispersed_and_deterministic():
    graph = _line_graph(40)

    first = sample_graph_farthest_points(
        graph,
        size=10,
        generator=torch.Generator().manual_seed(47),
    )
    second = sample_graph_farthest_points(
        graph,
        size=10,
        generator=torch.Generator().manual_seed(47),
    )

    assert torch.equal(first, second)
    assert first.unique().numel() == 10
    assert int(first.min()) == 0
    assert int(first.max()) == 39
    sorted_points = torch.sort(first).values
    assert int(torch.diff(sorted_points).max()) <= 7


def test_graph_farthest_points_select_a_large_enough_component():
    graph = WeightedSensorGraph(
        edge_index=np.asarray([[0, 1, 3, 4, 5, 6], [1, 2, 4, 5, 6, 7]]),
        edge_weight=np.ones(6, dtype=np.float32),
        num_nodes=8,
    )

    points = sample_graph_farthest_points(
        graph,
        size=4,
        generator=torch.Generator().manual_seed(53),
    )
    assert set(points.tolist()).issubset({3, 4, 5, 6, 7})

    with pytest.raises(ValueError, match="No connected component"):
        sample_graph_farthest_points(graph, size=6)


def test_multiblock_sampling_supports_bfs_local_and_topological_endcaps():
    graph = _line_graph(48)
    contexts, targets = sample_multiblock_graph_masks(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=48,
        context_size=30,
        target_size=10,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {"count": 2, "strategy": "connected_region", "growth": "bfs"},
            {
                "count": 2,
                "strategy": "topological_endcaps",
                "growth": "bfs",
                "farthest_quantile": 0.8,
                "lobe_size_ratio": 0.5,
                "min_lobe_tokens": 5,
                "max_pairwise_overlap_ratio": 0.5,
            },
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(43),
    )

    assert targets.shape == (4, 1, 10)
    for target_id in range(4):
        assert not torch.isin(contexts[0, 0], targets[target_id, 0]).any()
    assert _is_connected(targets[0, 0], graph)
    assert _is_connected(targets[1, 0], graph)
    assert torch.isin(targets[2, 0], targets[3, 0]).sum().item() <= 5


def test_multiblock_sampling_caps_farthest_target_overlap_with_previous_endcap():
    graph = _line_graph(64)
    contexts, targets = sample_multiblock_graph_masks(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=64,
        context_size=44,
        target_size=12,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {
                "count": 1,
                "strategy": "geodesic_endcaps",
                "growth": "dijkstra",
                "farthest_quantile": 0.8,
                "lobe_size_ratio": 0.5,
                "min_lobe_tokens": 6,
            },
            {
                "count": 1,
                "strategy": "graph_farthest_points",
                "growth": "dijkstra",
                "max_previous_overlap_ratio": 0.5,
            },
            {"count": 2, "strategy": "connected_region", "growth": "dijkstra"},
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(59),
    )

    assert targets.shape == (4, 1, 12)
    assert torch.isin(targets[1, 0], targets[0, 0]).sum().item() <= 6
    for target_id in range(4):
        assert not torch.isin(contexts[0, 0], targets[target_id, 0]).any()


def _sample_hybrid_masks(generator_seed: int = 5):
    graph = _line_graph(32)
    return graph, sample_multiblock_graph_masks(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=32,
        context_size=20,
        target_size=6,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {
                "count": 2,
                "strategy": "connected_region",
                "growth": "dijkstra",
            },
            {
                "count": 2,
                "strategy": "geodesic_corridor",
                "growth": "dijkstra",
                "endpoint_sampling": "farthest_quantile",
                "farthest_quantile": 0.8,
                "max_pairwise_overlap_ratio": 0.5,
            },
        ],
        min_context_keep_tokens=4,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(generator_seed),
    )


def test_hybrid_target_groups_overlap_and_context_invariants():
    graph, (contexts, targets) = _sample_hybrid_masks()

    assert contexts.shape[0:2] == (1, 1)
    assert targets.shape == (4, 1, 6)
    for target_id in range(4):
        assert _is_connected(targets[target_id, 0], graph)
        assert not torch.isin(contexts[0, 0], targets[target_id, 0]).any()

    corridor_overlap = torch.isin(targets[2, 0], targets[3, 0]).sum().item()
    assert corridor_overlap <= 3


def test_multiblock_sampling_supports_local_and_endcap_target_groups():
    graph = _line_graph(48)
    contexts, targets = sample_multiblock_graph_masks(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=48,
        context_size=30,
        target_size=10,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {"count": 2, "strategy": "connected_region", "growth": "dijkstra"},
            {
                "count": 2,
                "strategy": "geodesic_endcaps",
                "growth": "dijkstra",
                "farthest_quantile": 0.8,
                "lobe_size_ratio": 0.5,
                "min_lobe_tokens": 5,
                "max_pairwise_overlap_ratio": 0.5,
            },
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(29),
    )

    assert targets.shape == (4, 1, 10)
    for target_id in range(4):
        assert not torch.isin(contexts[0, 0], targets[target_id, 0]).any()
    assert torch.isin(targets[2, 0], targets[3, 0]).sum().item() <= 5


def test_multiblock_sampling_supports_local_and_true_random_target_groups():
    graph = _line_graph(48)
    kwargs = dict(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=48,
        context_size=30,
        target_size=10,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {"count": 2, "strategy": "connected_region", "growth": "dijkstra"},
            {"count": 2, "strategy": "random"},
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
    )

    first_contexts, first_targets = sample_multiblock_graph_masks(
        **kwargs,
        generator=torch.Generator().manual_seed(31),
    )
    second_contexts, second_targets = sample_multiblock_graph_masks(
        **kwargs,
        generator=torch.Generator().manual_seed(31),
    )

    assert torch.equal(first_contexts, second_contexts)
    assert torch.equal(first_targets, second_targets)
    assert first_targets.shape == (4, 1, 10)
    assert _is_connected(first_targets[0, 0], graph)
    assert _is_connected(first_targets[1, 0], graph)
    assert not _is_connected(first_targets[2, 0], graph)
    assert not _is_connected(first_targets[3, 0], graph)
    for target_id in range(4):
        assert not torch.isin(first_contexts[0, 0], first_targets[target_id, 0]).any()


def test_multiblock_sampling_supports_local_and_stratified_random_targets():
    graph = _line_graph(48)
    graph_info = _batched_graph_info(graph)
    graph_info["node_group_id"] = torch.arange(6).repeat_interleave(8).unsqueeze(0)
    contexts, targets = sample_multiblock_graph_masks(
        graph_info=graph_info,
        batch_size=1,
        num_nodes=48,
        context_size=34,
        target_size=12,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {"count": 2, "strategy": "connected_region", "growth": "dijkstra"},
            {
                "count": 2,
                "strategy": "stratified_random",
                "min_per_group": 1,
                "remainder_allocation": "proportional",
            },
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(73),
    )

    assert targets.shape == (4, 1, 12)
    assert _is_connected(targets[0, 0], graph)
    assert _is_connected(targets[1, 0], graph)
    for target_id in (2, 3):
        assert torch.unique(graph_info["node_group_id"][0, targets[target_id, 0]]).numel() == 6
    for target_id in range(4):
        assert not torch.isin(contexts[0, 0], targets[target_id, 0]).any()


def test_multiblock_stratified_random_requires_group_metadata():
    graph = _line_graph(24)
    with pytest.raises(ValueError, match="node_group_id"):
        sample_multiblock_graph_masks(
            graph_info=_batched_graph_info(graph),
            batch_size=1,
            num_nodes=24,
            context_size=16,
            target_size=8,
            num_context_masks=1,
            num_target_masks=2,
            context_strategy="connected_region",
            target_strategy="connected_region",
            context_growth="dijkstra",
            target_growth="dijkstra",
            target_groups=[
                {"count": 1, "strategy": "connected_region"},
                {"count": 1, "strategy": "stratified_random"},
            ],
            min_context_keep_tokens=3,
            min_context_keep_ratio=0.0,
            max_resample_attempts=16,
        )


def test_multiblock_sampling_supports_group_specific_target_scales():
    graph = _line_graph(48)
    contexts, targets = sample_multiblock_graph_masks(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=48,
        context_size=40,
        target_size=10,
        num_context_masks=1,
        num_target_masks=4,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        target_groups=[
            {
                "count": 2,
                "strategy": "connected_region",
                "growth": "dijkstra",
                "scale": [0.125, 0.125],
            },
            {
                "count": 2,
                "strategy": "geodesic_endcaps",
                "growth": "dijkstra",
                "scale": [0.25, 0.25],
                "lobe_size_ratio": 0.5,
                "min_lobe_tokens": 5,
            },
        ],
        min_context_keep_tokens=5,
        min_context_keep_ratio=0.0,
        max_resample_attempts=128,
        generator=torch.Generator().manual_seed(37),
    )

    assert isinstance(targets, list)
    assert [target.shape for target in targets] == [
        torch.Size([1, 6]),
        torch.Size([1, 6]),
        torch.Size([1, 12]),
        torch.Size([1, 12]),
    ]
    for target in targets:
        assert not torch.isin(contexts[0, 0], target[0]).any()
    assert _is_connected(targets[0][0], graph)
    assert _is_connected(targets[1][0], graph)


def test_target_group_counts_must_match_num_target_masks():
    graph = _line_graph(12)
    with pytest.raises(ValueError, match="counts sum to 2"):
        sample_multiblock_graph_masks(
            graph_info=_batched_graph_info(graph),
            batch_size=1,
            num_nodes=12,
            context_size=8,
            target_size=3,
            num_context_masks=1,
            num_target_masks=4,
            context_strategy="connected_region",
            target_strategy="connected_region",
            context_growth="dijkstra",
            target_growth="dijkstra",
            target_groups=[{"count": 2, "strategy": "connected_region"}],
            min_context_keep_tokens=2,
            min_context_keep_ratio=0.0,
            max_resample_attempts=8,
        )


def test_single_target_config_keeps_previous_sampling_sequence():
    graph = _line_graph(24)
    kwargs = dict(
        graph_info=_batched_graph_info(graph),
        batch_size=1,
        num_nodes=24,
        context_size=16,
        target_size=4,
        num_context_masks=1,
        num_target_masks=2,
        context_strategy="connected_region",
        target_strategy="connected_region",
        context_growth="dijkstra",
        target_growth="dijkstra",
        min_context_keep_tokens=3,
        min_context_keep_ratio=0.0,
        max_resample_attempts=32,
    )
    implicit = sample_multiblock_graph_masks(
        **kwargs,
        generator=torch.Generator().manual_seed(11),
    )
    explicit = sample_multiblock_graph_masks(
        **kwargs,
        target_groups=[
            {
                "count": 2,
                "strategy": "connected_region",
                "growth": "dijkstra",
            }
        ],
        generator=torch.Generator().manual_seed(11),
    )
    assert all(torch.equal(left, right) for left, right in zip(implicit, explicit))


def test_graph_mask_collator_smoke_with_hybrid_targets():
    graph = _line_graph(16)
    masking = {
        "mode": "multiblock_graph",
        "context": {"strategy": "connected_region", "growth": "dijkstra"},
        "target": {
            "groups": [
                {"count": 2, "strategy": "connected_region", "growth": "dijkstra"},
                {
                    "count": 2,
                    "strategy": "geodesic_corridor",
                    "growth": "dijkstra",
                    "farthest_quantile": 0.8,
                    "max_pairwise_overlap_ratio": 0.5,
                },
            ]
        },
        "overlap": {
            "target_target": "allow",
            "context_target": "subtract_from_context",
            "context_context": "allow",
        },
        "min_context_keep_tokens": 2,
        "min_context_keep_ratio": 0.0,
        "max_resample_attempts": 128,
    }
    collator = JEPAGraphMaskCollator(
        context_mask_scale=(0.75, 0.75),
        target_mask_scale=(0.25, 0.25),
        num_context_masks=1,
        num_target_masks=4,
        masking=masking,
    )
    edge_index = torch.from_numpy(graph.edge_index)
    edge_attr = torch.from_numpy(graph.edge_weight).unsqueeze(-1)
    samples = [
        {
            "sensor": torch.zeros(2, 16, 3),
            "graph": {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_count": torch.tensor(graph.edge_index.shape[1]),
            },
        }
        for _ in range(2)
    ]

    batch = collator(samples)
    assert batch["sensor"].shape == (2, 2, 16, 3)
    assert batch["context_masks"].shape[0:2] == (1, 2)
    assert batch["target_masks"].shape == (4, 2, 4)


def test_jepa_forward_supports_balanced_variable_target_sizes(monkeypatch):
    # CPU regression uses the equivalent dense attention implementation.
    from tactile_ssl.model.layers.attention import Attention, MemEffAttention
    monkeypatch.setattr(MemEffAttention, "forward", Attention.forward)
    from functools import partial

    from tactile_ssl.algorithm.xela_jepa import XelaJEPAModule
    from tactile_ssl.model.xela_transformer import xela_tinier

    encoder = xela_tinier(
        in_dim=48,
        in_chans=3,
        sequence_length=2,
        time_chunk_size=2,
        depth=1,
        use_taxel_type_embedding=False,
    )
    module = XelaJEPAModule(
        encoder=encoder,
        optim_cfg=partial(torch.optim.AdamW, lr=1.0e-4),
        lr_scheduler_cfg=None,
        wd_scheduler_cfg=None,
        num_context_masks=1,
        num_target_masks=4,
    )
    sensor = torch.randn(1, 2, 48, 3)
    context_masks = torch.arange(24).reshape(1, 1, 24)
    target_masks = [
        torch.arange(24, 30).reshape(1, 6),
        torch.arange(30, 36).reshape(1, 6),
        torch.arange(24, 36).reshape(1, 12),
        torch.arange(36, 48).reshape(1, 12),
    ]

    loss = module.forward(sensor, context_masks, target_masks)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in module.context_encoder.parameters()
        if parameter.requires_grad
    )

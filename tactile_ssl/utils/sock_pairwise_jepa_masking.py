from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch.utils.data import default_collate, get_worker_info

from tactile_ssl.utils.jepa_masking import (
    _build_undirected_adjacency,
    _connected_components_for_nodes,
    _eligible_seed_nodes,
    _sample_connected_from_adjacency,
)
from tactile_ssl.utils.masking import sample_block_size_1d


class SockPairwiseJEPAGraphMaskCollator:
    """Sample one foot-local context per foot and four foot-local targets."""

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        min_context_keep_tokens: int = 32,
        min_context_keep_ratio: float = 0.15,
        context_feet: Sequence[int] = (0, 1),
        target_specs: Sequence[Mapping] = (
            {"foot": 0, "strategy": "connected_region"},
            {"foot": 0, "strategy": "random"},
            {"foot": 1, "strategy": "connected_region"},
            {"foot": 1, "strategy": "random"},
        ),
        growth: str = "dijkstra",
        max_resample_attempts: int = 32,
    ) -> None:
        self.context_mask_scale = tuple(float(v) for v in context_mask_scale)
        self.target_mask_scale = tuple(float(v) for v in target_mask_scale)
        self.min_context_keep_tokens = int(min_context_keep_tokens)
        self.min_context_keep_ratio = float(min_context_keep_ratio)
        self.context_feet = tuple(int(foot) for foot in context_feet)
        self.target_specs = [dict(spec) for spec in target_specs]
        self.growth = str(growth)
        self.max_resample_attempts = int(max_resample_attempts)
        if self.context_feet != (0, 1):
            raise ValueError("Sock pairwise JEPA expects contexts in left/right order (0, 1)")
        if len(self.target_specs) != 4:
            raise ValueError("Sock pairwise JEPA expects exactly four target masks")
        if any(spec.get("foot") not in (0, 1) for spec in self.target_specs):
            raise ValueError("Every Sock pairwise target must specify foot 0 or 1")
        if self.growth not in {"dijkstra", "bfs"}:
            raise ValueError("growth must be 'dijkstra' or 'bfs'")
        self._generator: Optional[torch.Generator] = None
        self._seed: Optional[int] = None

    def _worker_generator(self) -> torch.Generator:
        worker = get_worker_info()
        seed = int(worker.seed if worker is not None else torch.initial_seed())
        if self._generator is None or self._seed != seed:
            self._generator = torch.Generator().manual_seed(seed)
            self._seed = seed
        return self._generator

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        batch = default_collate(samples)
        graph = batch.pop("graph", None)
        if graph is None:
            raise ValueError("SockPairwiseJEPAGraphMaskCollator requires graph metadata")
        sensor = batch["sensor"]
        batch_size, _, num_nodes, _ = sensor.shape
        generator = self._worker_generator()

        node_groups = graph["node_group_id"]
        if node_groups.ndim == 1:
            node_groups = node_groups.unsqueeze(0).expand(batch_size, -1)
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
            edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1)

        group_template = node_groups[0].cpu().long()
        group_sizes = {
            foot: int((group_template == foot).sum().item()) for foot in self.context_feet
        }
        if any(size == 0 for size in group_sizes.values()):
            raise ValueError(f"Both Sock feet must contain nodes; got {group_sizes}")
        context_sizes = {
            foot: sample_block_size_1d(
                group_sizes[foot], self.context_mask_scale, generator=generator
            )[0]
            for foot in self.context_feet
        }
        target_sizes = {
            foot: sample_block_size_1d(
                group_sizes[foot], self.target_mask_scale, generator=generator
            )[0]
            for foot in self.context_feet
        }

        contexts_by_sample: list[list[torch.Tensor]] = []
        targets_by_sample: list[list[torch.Tensor]] = []
        for sample_idx in range(batch_size):
            groups = node_groups[sample_idx].cpu().long()
            adjacency = _build_undirected_adjacency(
                num_nodes, edge_index[sample_idx], edge_attr[sample_idx]
            )
            targets: list[torch.Tensor] = []
            for spec in self.target_specs:
                foot = int(spec["foot"])
                strategy = str(spec["strategy"])
                foot_nodes = torch.nonzero(groups == foot, as_tuple=False).flatten()
                size = target_sizes[foot]
                if strategy == "random":
                    target = foot_nodes[
                        torch.randperm(foot_nodes.numel(), generator=generator)[:size]
                    ]
                elif strategy == "connected_region":
                    components = _connected_components_for_nodes(
                        adjacency, foot_nodes.tolist()
                    )
                    eligible = _eligible_seed_nodes(adjacency, size, components)
                    target = _sample_connected_from_adjacency(
                        adjacency, size, self.growth, generator, eligible
                    )
                else:
                    raise ValueError(f"Unsupported Sock target strategy {strategy!r}")
                targets.append(torch.sort(target).values)

            contexts: list[torch.Tensor] = []
            for foot in self.context_feet:
                foot_nodes = torch.nonzero(groups == foot, as_tuple=False).flatten()
                foot_targets = [
                    target
                    for target, spec in zip(targets, self.target_specs)
                    if int(spec["foot"]) == foot
                ]
                target_union = torch.unique(torch.cat(foot_targets))
                min_context = max(
                    min(self.min_context_keep_tokens, foot_nodes.numel()),
                    int(self.min_context_keep_ratio * foot_nodes.numel() + 0.999999),
                )
                keep = None
                for _ in range(self.max_resample_attempts):
                    raw = foot_nodes[
                        torch.randperm(foot_nodes.numel(), generator=generator)[
                            : context_sizes[foot]
                        ]
                    ]
                    candidate = raw[~torch.isin(raw, target_union)]
                    if candidate.numel() >= min_context:
                        keep = torch.sort(candidate).values
                        break
                if keep is None:
                    raise RuntimeError(
                        f"Could not sample a sufficiently large context for foot {foot}"
                    )
                contexts.append(keep)

            contexts_by_sample.append(contexts)
            targets_by_sample.append(targets)

        common_context = min(
            mask.numel() for sample_contexts in contexts_by_sample for mask in sample_contexts
        )
        batch["context_masks"] = torch.stack(
            [
                torch.stack(
                    [contexts_by_sample[b][context_id][:common_context] for b in range(batch_size)]
                )
                for context_id in range(len(self.context_feet))
            ]
        )
        masks = [
            torch.stack([targets_by_sample[b][target_id] for b in range(batch_size)])
            for target_id in range(len(self.target_specs))
        ]
        sizes = {mask.shape[-1] for mask in masks}
        batch["target_masks"] = torch.stack(masks) if len(sizes) == 1 else masks
        return batch

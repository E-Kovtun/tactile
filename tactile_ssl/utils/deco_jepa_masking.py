from __future__ import annotations

from collections.abc import Mapping, Sequence

from typing import Optional
import math
from functools import lru_cache
import torch
from torch.utils.data import default_collate
from tactile_ssl.utils.jepa_masking import (
    _build_undirected_adjacency, _connected_components_for_nodes,
    _eligible_seed_nodes, _sample_connected_from_adjacency,
)
from tactile_ssl.utils.masking import sample_block_size_1d


from tactile_ssl.utils.sock_jepa_masking import SockJEPAGraphMaskCollator


def uniform_context_subset(mask, length, generator):
    if not 0 < length <= mask.numel():
        raise ValueError("Invalid context length")
    if length == mask.numel():
        return mask
    selected = torch.randperm(mask.numel(), generator=generator)[:length]
    return torch.sort(mask[selected]).values


@lru_cache(maxsize=256)
def _overlap_distribution(n_inside, n_outside, size, cap):
    lower = max(0, size - n_outside)
    upper = min(cap, size, n_inside)
    if lower > upper:
        raise ValueError("Target size and overlap limit are infeasible")
    values = tuple(range(lower, upper + 1))
    # Hypergeometric mass, truncated at cap. Weighting k correctly avoids
    # forcing overlap to zero or always saturating the upper bound.
    masses = [math.comb(n_inside, k) * math.comb(n_outside, size-k) for k in values]
    total = sum(masses)
    return values, torch.tensor([m/total for m in masses], dtype=torch.float64)


def sample_capped_random_target(nodes, size, previous, max_overlap, generator):
    inside_mask = torch.isin(nodes, previous)
    inside, outside = nodes[inside_mask], nodes[~inside_mask]
    cap = math.floor(max_overlap * min(size, previous.numel()) + 1e-12)
    values, weights = _overlap_distribution(inside.numel(), outside.numel(), size, cap)
    k = values[int(torch.multinomial(weights, 1, generator=generator))]
    a = inside[torch.randperm(inside.numel(), generator=generator)[:k]]
    b = outside[torch.randperm(outside.numel(), generator=generator)[:size-k]]
    return torch.cat([a, b])


class DecoJEPAGraphMaskCollator(SockJEPAGraphMaskCollator):
    """One global context plus local/random targets on each DECO hand."""

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        min_context_keep_tokens: int = 32,
        min_context_keep_ratio: float = 0.15,
        target_specs: Sequence[Mapping] = (
            {"hand": 0, "strategy": "connected_region"},
            {"hand": 0, "strategy": "random"},
            {"hand": 1, "strategy": "connected_region"},
            {"hand": 1, "strategy": "random"},
        ),
        growth: str = "dijkstra",
        max_resample_attempts: int = 32,
        random_context_crop: bool = False,
        max_target_overlap: float | None = None,
    ) -> None:
        translated = []
        for spec in target_specs:
            item = dict(spec)
            if "hand" not in item:
                raise ValueError("Every DECO target spec must contain hand=0 or hand=1")
            item["foot"] = item.pop("hand")
            translated.append(item)
        super().__init__(
            context_mask_scale=context_mask_scale,
            target_mask_scale=target_mask_scale,
            min_context_keep_tokens=min_context_keep_tokens,
            min_context_keep_ratio=min_context_keep_ratio,
            target_specs=translated,
            growth=growth,
            max_resample_attempts=max_resample_attempts,
        )

        self.random_context_crop = bool(random_context_crop)
        self.max_target_overlap = max_target_overlap
        if max_target_overlap is not None:
            self.max_target_overlap = float(max_target_overlap)
            if not 0 <= self.max_target_overlap <= 1:
                raise ValueError("max_target_overlap must lie in [0, 1]")
            expected = [(0, "connected_region"), (0, "random"),
                        (1, "connected_region"), (1, "random")]
            actual = [(s.get("foot"), s["strategy"]) for s in self.target_specs]
            if actual != expected:
                raise ValueError("Overlap cap requires per-hand 2L2R, local then random")

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        if not self.random_context_crop and self.max_target_overlap is None:
            return super().__call__(samples)
        batch = default_collate(samples)
        graph = batch.pop("graph", None)
        if graph is None:
            raise ValueError("DECOJEPAGraphMaskCollator requires graph metadata")
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

        contexts: list[torch.Tensor] = []
        targets_by_sample: list[list[torch.Tensor]] = []
        raw_context_size = sample_block_size_1d(
            num_nodes, self.context_mask_scale, generator=generator
        )[0]
        min_context = max(
            self.min_context_keep_tokens,
            int(self.min_context_keep_ratio * num_nodes + 0.999999),
        )
        group_template = node_groups[0].cpu().long()
        target_sizes: dict[Optional[int], int] = {
            foot: sample_block_size_1d(
                int((group_template == foot).sum().item()),
                self.target_mask_scale,
                generator=generator,
            )[0]
            for foot in (0, 1)
        }
        target_sizes[None] = sample_block_size_1d(
            num_nodes, self.target_mask_scale, generator=generator
        )[0]

        for sample_idx in range(batch_size):
            groups = node_groups[sample_idx].cpu().long()
            adjacency = _build_undirected_adjacency(
                num_nodes, edge_index[sample_idx], edge_attr[sample_idx]
            )
            targets: list[torch.Tensor] = []
            for spec in self.target_specs:
                foot_value = spec.get("foot")
                foot = None if foot_value is None else int(foot_value)
                strategy = str(spec["strategy"])
                foot_nodes = (
                    torch.arange(num_nodes)
                    if foot is None
                    else torch.nonzero(groups == foot, as_tuple=False).flatten()
                )
                size = target_sizes[foot]
                if strategy == "random":
                    # The local target stays connected and unchanged. Draw the
                    # random target uniformly conditional on the overlap cap.
                    if self.max_target_overlap is None:
                        target = foot_nodes[torch.randperm(foot_nodes.numel(), generator=generator)[:size]]
                    else:
                        target = sample_capped_random_target(
                            foot_nodes, size, targets[-1], self.max_target_overlap, generator
                        )
                elif strategy == "connected_region":
                    node_list = foot_nodes.tolist()
                    components = _connected_components_for_nodes(adjacency, node_list)
                    eligible = _eligible_seed_nodes(adjacency, size, components)
                    target = _sample_connected_from_adjacency(
                        adjacency, size, self.growth, generator, eligible
                    )
                else:
                    raise ValueError(f"Unsupported DECO target strategy {strategy!r}")
                targets.append(torch.sort(target).values)

            target_union = torch.unique(torch.cat(targets))
            keep = None
            for _ in range(self.max_resample_attempts):
                raw = torch.randperm(num_nodes, generator=generator)[:raw_context_size]
                candidate = raw[~torch.isin(raw, target_union)]
                if candidate.numel() >= min_context:
                    keep = torch.sort(candidate).values
                    break
            if keep is None:
                raise RuntimeError("Could not sample a sufficiently large DECO context")
            contexts.append(keep)
            targets_by_sample.append(targets)

        common_context = min(mask.numel() for mask in contexts)
        batch["context_masks"] = torch.stack(
            [torch.stack([(uniform_context_subset(mask, common_context, generator)
                           if self.random_context_crop else mask[:common_context])
                          for mask in contexts])]
        )
        masks = [
            torch.stack([targets_by_sample[b][target_id] for b in range(batch_size)])
            for target_id in range(4)
        ]
        sizes = {mask.shape[-1] for mask in masks}
        batch["target_masks"] = torch.stack(masks) if len(sizes) == 1 else masks
        return batch

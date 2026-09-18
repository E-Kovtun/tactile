"""DECO masking with one large Dijkstra-grown context region per hand."""

from collections.abc import Sequence

import torch

from tactile_ssl.utils.deco_jepa_masking import DecoJEPAGraphMaskCollator
from tactile_ssl.utils.jepa_masking import (
    _build_undirected_adjacency,
    _connected_components_for_nodes,
    _eligible_seed_nodes,
    _sample_connected_from_adjacency,
)
from tactile_ssl.utils.masking import sample_block_size_1d


class DecoTwoHandConnectedContextCollator(DecoJEPAGraphMaskCollator):
    """Concatenate independently grown per-hand contexts after target exclusion.

    The requested global context fraction is split proportionally between the two
    hands. Each raw part is a Dijkstra region restricted to that hand. Targets are
    then removed. Batch equalisation is performed independently per hand with a
    random subset, avoiding the low-sensor-id/first-hand bias of sorted truncation.
    """

    def __call__(self, samples: Sequence[dict]) -> dict:
        batch = super().__call__(samples)
        generator = self._worker_generator()
        num_nodes = int(batch["sensor"].shape[2])
        raw_total = sample_block_size_1d(
            num_nodes, self.context_mask_scale, generator=generator
        )[0]

        retained_by_sample: list[list[torch.Tensor]] = []
        for sample_index, sample in enumerate(samples):
            graph = sample["graph"]
            groups = graph["node_group_id"].cpu().long()
            adjacency = _build_undirected_adjacency(
                num_nodes, graph["edge_index"], graph["edge_attr"]
            )
            hand_nodes = [torch.nonzero(groups == hand, as_tuple=False).flatten() for hand in (0, 1)]
            hand_sizes = [int(nodes.numel()) for nodes in hand_nodes]
            left_raw = round(raw_total * hand_sizes[0] / num_nodes)
            requested = [left_raw, raw_total - left_raw]
            target_union = torch.unique(
                torch.cat([
                    mask[sample_index]
                    for mask in batch["target_masks"]
                ])
            )

            retained: list[torch.Tensor] = []
            for hand in (0, 1):
                minimum = max(
                    round(self.min_context_keep_tokens * hand_sizes[hand] / num_nodes),
                    int(self.min_context_keep_ratio * hand_sizes[hand] + 0.999999),
                )
                components = _connected_components_for_nodes(adjacency, hand_nodes[hand].tolist())
                eligible = _eligible_seed_nodes(adjacency, requested[hand], components)
                keep = None
                for _ in range(self.max_resample_attempts):
                    raw = _sample_connected_from_adjacency(
                        adjacency, requested[hand], "dijkstra", generator, eligible
                    )
                    candidate = raw[~torch.isin(raw, target_union)]
                    if candidate.numel() >= minimum:
                        keep = candidate
                        break
                if keep is None:
                    raise RuntimeError(
                        f"Could not retain a sufficiently large connected context for DECO hand {hand}"
                    )
                retained.append(keep)
            retained_by_sample.append(retained)

        common = [
            min(parts[hand].numel() for parts in retained_by_sample)
            for hand in (0, 1)
        ]
        contexts = []
        for parts in retained_by_sample:
            balanced = []
            for hand in (0, 1):
                permutation = torch.randperm(parts[hand].numel(), generator=generator)
                balanced.append(parts[hand][permutation[: common[hand]]])
            contexts.append(torch.sort(torch.cat(balanced)).values)
        batch["context_masks"] = torch.stack(contexts).unsqueeze(0)
        return batch

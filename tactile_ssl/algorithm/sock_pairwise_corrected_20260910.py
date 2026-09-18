"""Pair-aware Socks masking and loss, with separate context lengths per foot."""
import math
from collections import defaultdict

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from tactile_ssl.algorithm.sock_pairwise_jepa import SockPairwiseJEPAModule
from tactile_ssl.utils.sock_pairwise_jepa_masking import SockPairwiseJEPAGraphMaskCollator
from tactile_ssl.utils.jepa_masking import (
    _build_undirected_adjacency, _connected_components_for_nodes,
    _eligible_seed_nodes, _sample_connected_from_adjacency,
)
from tactile_ssl.utils.masking import sample_block_size_1d


PAIRS = {
    'samefoot': ((0, 0), (0, 1), (1, 2), (1, 3)),
    'crossfoot': ((0, 2), (0, 3), (1, 0), (1, 1)),
    'bothfeet': tuple((c, t) for c in range(2) for t in range(4)),
}


class PairAwareSockCollator(SockPairwiseJEPAGraphMaskCollator):
    def __init__(self, context_target_pairs, **kwargs):
        super().__init__(**kwargs)
        self.pairs = tuple(tuple(map(int, p)) for p in context_target_pairs)
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError('Pairs must be nonempty and unique')
        if any(c not in (0, 1) or t not in range(4) for c, t in self.pairs):
            raise ValueError('Invalid context/target pair')

    def __call__(self, samples):
        batch = default_collate(samples)
        graph = batch.pop('graph')
        groups = graph['node_group_id'][0].cpu().long()
        if not torch.equal(graph['node_group_id'], groups.expand_as(graph['node_group_id'])):
            raise ValueError('A batch must use a single physical sensor ordering')
        nodes = [torch.where(groups == foot)[0] for foot in (0, 1)]
        if tuple(map(len, nodes)) != (237, 216):
            raise ValueError('Expected 237 left and 216 right sensors')
        generator = self._worker_generator()
        context_sizes = [sample_block_size_1d(len(n), self.context_mask_scale, generator=generator)[0] for n in nodes]
        # Each target draws its own size, measured against its own foot.
        target_sizes = [sample_block_size_1d(len(nodes[int(s['foot'])]), self.target_mask_scale, generator=generator)[0] for s in self.target_specs]
        all_targets, allowed_by_sample = [], []
        for b in range(len(samples)):
            adjacency = _build_undirected_adjacency(len(groups), graph['edge_index'][b], graph['edge_attr'][b])
            targets = []
            for spec, size in zip(self.target_specs, target_sizes):
                foot_nodes = nodes[int(spec['foot'])]
                if spec['strategy'] == 'random':
                    target = foot_nodes[torch.randperm(len(foot_nodes), generator=generator)[:size]]
                elif spec['strategy'] == 'connected_region':
                    components = _connected_components_for_nodes(adjacency, foot_nodes.tolist())
                    eligible = _eligible_seed_nodes(adjacency, size, components)
                    target = _sample_connected_from_adjacency(adjacency, size, self.growth, generator, eligible)
                else:
                    raise ValueError(spec['strategy'])
                if not torch.isin(target, foot_nodes).all():
                    raise RuntimeError('Target escaped its foot')
                targets.append(target.sort().values)
            allowed = []
            for c in (0, 1):
                # Exclude only the targets predicted from THIS context.
                forbidden = [targets[t] for pc, t in self.pairs if pc == c]
                union = torch.unique(torch.cat(forbidden)) if forbidden else torch.empty(0, dtype=torch.long)
                allowed.append(nodes[c][~torch.isin(nodes[c], union)])
            all_targets.append(targets)
            allowed_by_sample.append(allowed)

        contexts = []
        for c in (0, 1):
            # Sample directly from admissible nodes. No raw-mask subtraction,
            # no sorted-prefix clipping, and no coupling of the two foot sizes.
            size = min(context_sizes[c], min(len(a[c]) for a in allowed_by_sample))
            minimum = max(self.min_context_keep_tokens, math.ceil(self.min_context_keep_ratio * len(nodes[c])))
            if size < minimum:
                raise RuntimeError(f'Foot {c} has only {size} admissible context tokens, needs {minimum}')
            contexts.append(torch.stack([
                a[c][torch.randperm(len(a[c]), generator=generator)[:size]].sort().values
                for a in allowed_by_sample
            ]))
        batch['context_masks'] = contexts
        batch['target_masks'] = [torch.stack([ts[t] for ts in all_targets]) for t in range(4)]
        return batch


class PairAwareSockJEPA(SockPairwiseJEPAModule):
    """Compute only requested pairs, preserving each foot's context length."""
    def __init__(self, teacher_scope='foot', **kwargs):
        if teacher_scope != 'foot':
            raise ValueError('Corrected pairwise comparison requires a foot-isolated teacher')
        super().__init__(**kwargs)
        self.teacher_scope = teacher_scope

    @torch.no_grad()
    def _teacher_patch_tokens(self, xs):
        # Do not allow an opposite-foot target to import source-foot signals
        # through teacher attention. Keep global IDs for predictor positions.
        combined = None
        for foot in (0, 1):
            indices = torch.where(self.target_encoder.foot_ids == foot)[0]
            masks = indices[None, None].expand(1, xs.shape[0], -1)
            tokens = self.target_encoder.forward_features(xs, masks=masks, mask_type='tubelet')['x_norm_patchtokens']
            tokens = einops.rearrange(tokens, 'b (t n) d -> b t n d', n=len(indices))
            if combined is None:
                combined = tokens.new_empty(xs.shape[0], tokens.shape[1], xs.shape[-2], tokens.shape[-1])
            combined[:, :, indices] = tokens
        return combined

    def _batch_loss(self, batch):
        xs = batch['sensor']
        loss = self.forward(xs, [m.to(xs.device) for m in batch['context_masks']],
                            [m.to(xs.device) for m in batch['target_masks']])
        return {'loss': loss, 'ssl_loss': loss.item()}

    def training_step(self, batch, batch_idx):
        self.step += 1
        return self._batch_loss(batch)

    def validation_step(self, batch, batch_idx):
        return self._batch_loss(batch)

    def forward(self, xs, context_masks, target_masks):
        if len(context_masks) != 2 or len(target_masks) != 4:
            raise ValueError('Expected two foot contexts and four targets')
        if getattr(self.context_encoder, 'input_fusion', 'joint') == 'fresh_random':
            raise ValueError('Corrected Socks comparison expects no input fusion')
        with torch.no_grad():
            teacher = self._teacher_patch_tokens(xs)
            positions = self.target_encoder.get_position_embedding(xs.device)
        losses = []
        for c, mask in enumerate(context_masks):
            context = self.context_encoder.forward_features(xs, masks=mask[None], mask_type='tubelet')['x_norm_patchtokens']
            context = einops.rearrange(context, 'b (t n) d -> b t n d', n=mask.shape[-1])
            by_size = defaultdict(list)
            for pc, t in self.context_target_pairs:
                if pc == c:
                    by_size[target_masks[t].shape[-1]].append(t)
            for ids in by_size.values():
                masks = torch.stack([target_masks[t] for t in ids])
                predictions = self.predictor(context, context_masks=mask[None], masks=masks, context_pos_embed=positions)
                for prediction, t in zip(predictions, ids):
                    target = self.target_encoder.apply_tubelet_masks(teacher, target_masks[t][None])
                    losses.append(F.mse_loss(prediction, target))
        if len(losses) != len(self.context_target_pairs):
            raise RuntimeError('Incorrect number of pair losses')
        return torch.stack(losses).mean()

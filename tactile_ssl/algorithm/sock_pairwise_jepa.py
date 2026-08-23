from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, List, Union

import einops
import torch
import torch.nn.functional as F

from tactile_ssl.algorithm.xela_jepa import XelaJEPAModule


class SockPairwiseJEPAModule(XelaJEPAModule):
    """JEPA loss restricted to configured (context, target) pairs."""

    def __init__(
        self,
        context_target_pairs: Sequence[Sequence[int]],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.context_target_pairs = tuple(
            (int(pair[0]), int(pair[1])) for pair in context_target_pairs
        )
        if not self.context_target_pairs:
            raise ValueError("context_target_pairs must not be empty")
        if any(len(pair) != 2 for pair in context_target_pairs):
            raise ValueError("Each context_target_pairs entry must contain two indices")
        if any(
            context_id < 0
            or context_id >= self.num_context_masks
            or target_id < 0
            or target_id >= self.num_target_masks
            for context_id, target_id in self.context_target_pairs
        ):
            raise ValueError(
                "context_target_pairs contains an index outside configured mask counts"
            )
        if len(set(self.context_target_pairs)) != len(self.context_target_pairs):
            raise ValueError("context_target_pairs must not contain duplicates")

    def forward(
        self,
        xs: torch.Tensor,
        context_masks: torch.Tensor,
        target_masks: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        if context_masks is None or target_masks is None:
            raise AssertionError("Masks are required for JEPA training")
        k, b = context_masks.shape[:2]
        if k != self.num_context_masks:
            raise ValueError(f"Expected {self.num_context_masks} contexts, got {k}")

        auxiliary_embedding = None
        if getattr(self.context_encoder, "input_fusion", "joint") == "fresh_random":
            auxiliary_embedding = self.context_encoder.sample_fresh_random_embedding(xs)
        context_out = self.context_encoder.forward_features(
            xs,
            masks=context_masks,
            mask_type="tubelet",
            auxiliary_embedding=auxiliary_embedding,
        )
        context_patch_tokens = einops.rearrange(
            context_out["x_norm_patchtokens"],
            "b (t n) c -> b t n c",
            n=context_masks.shape[-1],
        )

        with torch.no_grad():
            target_out = self.target_encoder.forward_features(
                xs, auxiliary_embedding=auxiliary_embedding
            )
        target_patch_tokens = einops.rearrange(
            target_out["x_norm_patchtokens"],
            "b (t n) c -> b t n c",
            n=xs.shape[-2],
        )

        def predict_group(mask_group: torch.Tensor):
            p = mask_group.shape[0]
            prediction = torch.cat(
                self.predictor(
                    context_patch_tokens,
                    context_masks=context_masks,
                    masks=mask_group,
                    context_pos_embed=self.target_encoder.get_position_embedding(xs.device),
                ),
                dim=0,
            )
            prediction = einops.rearrange(
                prediction,
                "(p k b) t n c -> k p b t n c",
                k=k,
                p=p,
                b=b,
                n=mask_group.shape[-1],
            )
            target = self.target_encoder.apply_tubelet_masks(
                target_patch_tokens, masks=mask_group
            )
            target = einops.rearrange(
                target, "(p b) t n c -> p b t n c", b=b, p=p
            )
            target = einops.repeat(target, "p b t n c -> k p b t n c", k=k)
            return prediction, target.detach()

        indexed_masks = (
            list(enumerate(target_masks.unbind(0)))
            if isinstance(target_masks, torch.Tensor)
            else list(enumerate(target_masks))
        )
        masks_by_size: Dict[int, List[tuple[int, torch.Tensor]]] = {}
        for target_id, target_mask in indexed_masks:
            masks_by_size.setdefault(target_mask.shape[-1], []).append(
                (target_id, target_mask)
            )

        pair_losses = []
        requested_pairs = set(self.context_target_pairs)
        for same_size_masks in masks_by_size.values():
            target_ids = [target_id for target_id, _ in same_size_masks]
            mask_group = torch.stack([mask for _, mask in same_size_masks], dim=0)
            prediction, target = predict_group(mask_group)
            for local_target_id, target_id in enumerate(target_ids):
                for context_id in range(k):
                    if (context_id, target_id) in requested_pairs:
                        pair_losses.append(
                            F.mse_loss(
                                prediction[context_id, local_target_id],
                                target[context_id, local_target_id],
                            )
                        )
        if len(pair_losses) != len(self.context_target_pairs):
            raise RuntimeError(
                f"Computed {len(pair_losses)} pair losses for "
                f"{len(self.context_target_pairs)} configured pairs"
            )
        return torch.stack(pair_losses).mean()

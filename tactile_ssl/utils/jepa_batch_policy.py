"""Opt-in, whole-batch mixtures of existing graph-mask policies."""

from copy import deepcopy
import math
from typing import Mapping, Sequence

import torch

from tactile_ssl.utils.jepa_masking import JEPAGraphMaskCollator


class JEPABatchPolicyMaskCollator(JEPAGraphMaskCollator):
    """Draw one target policy per batch using the DataLoader worker's RNG.

    Context sampling, overlap constraints, and cropping remain delegated to the
    ordinary collator. The extra batch field records which policy was sampled.
    """

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        num_context_masks: int,
        num_target_masks: int,
        masking: Mapping,
        policies: Sequence[Mapping],
    ) -> None:
        super().__init__(context_mask_scale, target_mask_scale,
                         num_context_masks, num_target_masks, masking)
        if not policies:
            raise ValueError("At least one batch policy is required")
        self.policy_names = []
        self.policy_maskings = []
        weights = []
        for policy in policies:
            name = str(policy["name"])
            weight = float(policy["weight"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Batch policy weights must be finite and nonnegative")
            groups = deepcopy(policy["groups"])
            if sum(int(group["count"]) for group in groups) != self.num_target_masks:
                raise ValueError("Each policy must have num_target_masks targets")
            policy_masking = deepcopy(dict(masking))
            policy_masking["target"] = deepcopy(dict(masking.get("target", {})))
            policy_masking["target"]["groups"] = groups
            self.policy_names.append(name)
            self.policy_maskings.append(policy_masking)
            weights.append(weight)
        if len(set(self.policy_names)) != len(self.policy_names):
            raise ValueError("Batch policy names must be unique")
        if sum(weights) <= 0:
            raise ValueError("At least one batch policy must have positive weight")
        self.policy_weights = torch.tensor(weights, dtype=torch.float64)

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        index = int(torch.multinomial(
            self.policy_weights, 1, generator=self._worker_generator()
        ).item())
        original = self.masking
        self.masking = self.policy_maskings[index]
        try:
            batch = super().__call__(samples)
        finally:
            self.masking = original
        batch["mask_policy_id"] = torch.tensor(index, dtype=torch.long)
        return batch

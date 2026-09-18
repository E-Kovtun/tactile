"""Image-style rectangular I-JEPA masks on the sparse Xela hand atlas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch.utils.data import default_collate, get_worker_info

from tactile_ssl.data.xela.atlas import (
    XELA_ATLAS_HEIGHT,
    XELA_ATLAS_WIDTH,
    xela_atlas_coordinates,
)


class XelaIJEPA2DMaskCollator:
    """Sample one rectangular context and rectangular targets on the 2D atlas.

    Empty atlas cells are layout gaps, not model tokens. Each sampled rectangle
    is intersected with the 368 real taxels; target rectangles may overlap and
    their union is subtracted from the context, matching I-JEPA semantics.
    """

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        num_context_masks: int = 1,
        num_target_masks: int = 4,
        aspect_ratio: Sequence[float] = (0.75, 1.5),
        min_context_keep_tokens: int = 32,
        min_context_keep_ratio: float = 0.15,
        max_resample_attempts: int = 64,
    ) -> None:
        self.context_mask_scale = self._pair("context_mask_scale", context_mask_scale)
        self.target_mask_scale = self._pair("target_mask_scale", target_mask_scale)
        self.aspect_ratio = self._pair("aspect_ratio", aspect_ratio)
        self.num_context_masks = int(num_context_masks)
        self.num_target_masks = int(num_target_masks)
        self.min_context_keep_tokens = int(min_context_keep_tokens)
        self.min_context_keep_ratio = float(min_context_keep_ratio)
        self.max_resample_attempts = int(max_resample_attempts)
        if self.num_context_masks != 1:
            raise ValueError("Original-style Xela I-JEPA requires exactly one context mask")
        if self.num_target_masks <= 0:
            raise ValueError("num_target_masks must be positive")
        self.coordinates = xela_atlas_coordinates()
        self.num_nodes = int(self.coordinates.shape[0])
        self._generator: Optional[torch.Generator] = None
        self._generator_seed: Optional[int] = None

    @staticmethod
    def _pair(name: str, values: Sequence[float]) -> tuple[float, float]:
        result = tuple(float(value) for value in values)
        if len(result) != 2 or not 0 < result[0] <= result[1]:
            raise ValueError(f"{name} must be a positive [min, max] pair")
        return result

    def _worker_generator(self) -> torch.Generator:
        worker = get_worker_info()
        seed = int(worker.seed if worker is not None else torch.initial_seed())
        if self._generator is None or self._generator_seed != seed:
            self._generator = torch.Generator().manual_seed(seed)
            self._generator_seed = seed
        return self._generator

    @staticmethod
    def _uniform(bounds: tuple[float, float], generator: torch.Generator) -> float:
        return bounds[0] + torch.rand((), generator=generator).item() * (bounds[1] - bounds[0])

    def _block_shape(
        self, scale: tuple[float, float], generator: torch.Generator
    ) -> tuple[int, int, int]:
        desired = max(1, round(self.num_nodes * self._uniform(scale, generator)))
        ratio = self._uniform(self.aspect_ratio, generator)
        occupancy = self.num_nodes / float(XELA_ATLAS_HEIGHT * XELA_ATLAS_WIDTH)
        area = desired / occupancy
        height = max(1, min(XELA_ATLAS_HEIGHT, round(math.sqrt(area * ratio))))
        width = max(1, min(XELA_ATLAS_WIDTH, round(math.sqrt(area / ratio))))
        return height, width, desired

    def _sample_rectangle(
        self,
        shape: tuple[int, int, int],
        generator: torch.Generator,
        exclude: Optional[torch.Tensor] = None,
        min_keep: int = 1,
    ) -> torch.Tensor:
        height, width, desired = shape
        best: Optional[torch.Tensor] = None
        best_error = self.num_nodes + 1
        for _ in range(self.max_resample_attempts):
            top = int(torch.randint(XELA_ATLAS_HEIGHT - height + 1, (), generator=generator))
            left = int(torch.randint(XELA_ATLAS_WIDTH - width + 1, (), generator=generator))
            inside = (
                (self.coordinates[:, 0] >= top)
                & (self.coordinates[:, 0] < top + height)
                & (self.coordinates[:, 1] >= left)
                & (self.coordinates[:, 1] < left + width)
            )
            indices = torch.nonzero(inside, as_tuple=False).flatten()
            if exclude is not None:
                indices = indices[~torch.isin(indices, exclude)]
            error = abs(int(indices.numel()) - desired)
            if indices.numel() and error < best_error:
                best, best_error = indices, error
            if error == 0:
                break
        if best is None or best.numel() < min_keep:
            kept = 0 if best is None else int(best.numel())
            raise RuntimeError(
                f"Could not sample a Xela atlas rectangle with {min_keep} taxels; best kept {kept}"
            )
        return best

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        batch = default_collate(samples)
        if "sensor" not in batch:
            raise ValueError("XelaIJEPA2DMaskCollator requires sensor data")
        batch_size, _, num_nodes, _ = batch["sensor"].shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} Xela taxels, got {num_nodes}")

        generator = self._worker_generator()
        context_shape = self._block_shape(self.context_mask_scale, generator)
        target_shape = self._block_shape(self.target_mask_scale, generator)
        contexts: list[torch.Tensor] = []
        targets: list[list[torch.Tensor]] = []
        for _ in range(batch_size):
            sample_targets = [
                self._sample_rectangle(target_shape, generator)
                for _ in range(self.num_target_masks)
            ]
            target_union = torch.unique(torch.cat(sample_targets))
            required_context = max(
                self.min_context_keep_tokens,
                math.ceil(self.num_nodes * self.min_context_keep_ratio),
            )
            context = self._sample_rectangle(
                context_shape,
                generator,
                exclude=target_union,
                min_keep=required_context,
            )
            contexts.append(torch.sort(context).values)
            targets.append([torch.sort(mask).values for mask in sample_targets])

        common_context = min(mask.numel() for mask in contexts)
        common_target = min(
            targets[b][target_id].numel()
            for b in range(batch_size)
            for target_id in range(self.num_target_masks)
        )
        context_masks = torch.stack(
            [torch.stack([mask[:common_context] for mask in contexts])]
        )
        target_masks = torch.stack(
            [
                torch.stack([targets[b][target_id][:common_target] for b in range(batch_size)])
                for target_id in range(self.num_target_masks)
            ]
        )
        batch.pop("graph", None)
        batch["context_masks"] = context_masks
        batch["target_masks"] = target_masks
        return batch

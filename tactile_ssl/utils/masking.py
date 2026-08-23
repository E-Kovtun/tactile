# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Union, List, Optional
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import math

from multiprocessing import Value

from logging import getLogger

import torch
import torch.utils

logger = getLogger()


def flattened_mask_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return flat token indices selected by an arbitrary boolean mask."""
    if mask.dtype != torch.bool:
        raise TypeError(f"Expected a boolean mask, got {mask.dtype}")
    return torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()


def split_crop_major_batch(
    tensor: torch.Tensor,
    num_crops: int,
) -> tuple[torch.Tensor, ...]:
    """Split a crop-major ``[num_crops * batch, ...]`` tensor by crop."""
    if num_crops <= 0:
        raise ValueError("num_crops must be positive")
    if tensor.shape[0] % num_crops:
        raise ValueError(
            f"Leading dimension {tensor.shape[0]} is not divisible by {num_crops} crops"
        )
    return tensor.chunk(num_crops, dim=0)


class MultiMaskWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None, *args, **kwargs):
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=[m], *args, **kwargs)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, mc, mt) in enumerate(zip(ctxt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, [mc], [mt], mask_index=i)]
        return outs


class MaskCollator(object):
    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get("spatial_scale"),
                temporal_pred_mask_scale=m.get("temporal_scale"),
                aspect_ratio=m.get("aspect_ratio"),
                npred=m.get("num_blocks"),
                max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                max_keep=m.get("max_keep", None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        batch_size = len(batch)
        # collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):
    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]] = (224, 224),
        num_frames: int = 16,
        spatial_patch_size: Tuple[int, int] = (16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert len(crop_size) == 2, "Only supports 2D images"
        self.crop_size = crop_size
        self.height, self.width = (
            crop_size[0] // spatial_patch_size[0],
            crop_size[1] // spatial_patch_size[1],
        )
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred


def extend_masking(
    id_keep: torch.Tensor,
    id_restore: torch.Tensor,
    shape: List[int],
    mask_dim: int,
):
    # This function populates "id_keep" and "id_restore" in mask_dim to their global indices with respect to "shape"
    num_dims = len(shape)
    mask_dim = mask_dim + num_dims if mask_dim < 0 else mask_dim

    assert mask_dim >= 0 and mask_dim < num_dims
    assert shape[mask_dim] == id_restore.shape[1]

    b = id_keep.shape[0]
    n = shape[mask_dim]
    k = id_keep.shape[1]
    r = n - k

    device = id_keep.device

    inner_size = np.prod(shape[mask_dim + 1 :]).astype(int)
    id_inner = torch.arange(inner_size, device=device, dtype=int)
    outer_size = np.prod(shape[:mask_dim]).astype(int)
    id_outer = torch.arange(outer_size, device=device, dtype=int)[None, :, None, None]

    id_keep = (id_outer * shape[mask_dim] * inner_size + id_keep[:, None, :, None] * inner_size + id_inner).reshape(
        b, -1
    )

    id_inner = id_restore[:, None, :, None] * inner_size + id_inner
    id_restore = torch.where(
        id_restore[:, None, :, None] < k,
        id_outer * k * inner_size + id_inner,
        id_outer * r * inner_size + id_inner + (outer_size - 1) * k * inner_size,
    ).reshape(b, -1)

    return id_keep, id_restore


def random_masking(
    batch: int,
    shape: List[int],
    mask_dim: Optional[int],
    mask_ratio: float,
    device: torch.device,
):
    num_dims = len(shape)

    if mask_dim is not None:
        mask_dim = mask_dim + num_dims if mask_dim < 0 else mask_dim
        assert mask_dim >= 0 and mask_dim < num_dims

    b = batch
    n = np.prod(shape).astype(int) if mask_dim is None else shape[mask_dim]
    k = int(n * (1 - mask_ratio)) if n > 1 else int(np.random.rand(1)[0] > mask_ratio)  # length kept

    noise = torch.rand(b, n, device=device)  # noise in [0, 1]
    # sort noise for each sample
    id_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    id_restore = torch.argsort(id_shuffle, dim=-1)
    # keep the first subset
    id_keep = id_shuffle[:, :k]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([b, n], device=device)
    mask[:, :k] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=-1, index=id_restore)

    if mask_dim is not None:
        mask_shapes = [1] * len(shape)
        mask_shapes[mask_dim] = -1
        mask = mask.reshape([b] + mask_shapes).expand([b] + shape)
        mask = mask.contiguous().reshape([b, -1])
        id_keep, id_restore = extend_masking(id_keep, id_restore, shape, mask_dim)

    return (id_keep, mask, id_restore)


def block_masking(
    batch: int,
    shape: List[int],
    mask_dim: Optional[int],
    mask_ratio: float,
    device: torch.device,
):
    num_dims = len(shape)

    if mask_dim is not None:
        mask_dim = mask_dim + num_dims if mask_dim < 0 else mask_dim
        assert mask_dim >= 0 and mask_dim < num_dims

    b = batch
    n = np.prod(shape).astype(int) if mask_dim is None else shape[mask_dim]
    k = int(n * (1 - mask_ratio)) if n > 1 else int(np.random.rand(1)[0] > mask_ratio)  # length kept
    r = n - k  # length removed

    id_mask = torch.randint(0, k + 1, (b, 1), device=device)
    id_keep = torch.arange(0, k, device=device).repeat(b, 1)
    id_keep = torch.where(id_keep < id_mask, id_keep, id_keep + r)
    id_shuffle = torch.cat(
        [
            id_keep,
            torch.arange(0, r, device=device).repeat(b, 1) + id_mask,
        ],
        dim=-1,
    )
    id_restore = torch.argsort(id_shuffle, dim=-1)
    mask = torch.ones([b, n], device=device)
    mask[:, :k] = 0
    mask = torch.gather(mask, dim=-1, index=id_restore)

    if mask_dim is not None:
        mask_shapes = [1] * len(shape)
        mask_shapes[mask_dim] = -1
        mask = mask.reshape([b] + mask_shapes).expand([b] + shape)
        mask = mask.contiguous().reshape([b, -1])
        id_keep, id_restore = extend_masking(id_keep, id_restore, shape, mask_dim)

    return (id_keep, mask, id_restore)


def sample_block_size_1d(size: int, scale: Tuple[float, float], generator: Optional[torch.Generator] = None):
    _rand = torch.rand(1, generator=generator).item()
    # -- Sample block scale
    min_s, max_s = scale
    mask_scale = min_s + _rand * (max_s - min_s)
    mask_keep = int(size * mask_scale)
    mask_keep = min(mask_keep, size)
    return (mask_keep,)


def sample_block_size_2d(
    height: int,
    width: int,
    scale: Tuple[float, float],
    aspect_ratio_scale: Tuple[float, float],
    generator: Optional[torch.Generator] = None,
):
    _rand = torch.rand(1, generator=generator).item()
    # -- Sample block scale
    min_s, max_s = scale
    mask_scale = min_s + _rand * (max_s - min_s)
    max_keep = int(height * width * mask_scale)
    # -- Sample block aspect-ratio
    min_ar, max_ar = aspect_ratio_scale
    aspect_ratio = min_ar + _rand * (max_ar - min_ar)
    # -- Compute block height and width (given scale and aspect-ratio)
    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    w = int(round(math.sqrt(max_keep / aspect_ratio)))
    h = min(h, height)
    w = min(w, width)
    return (h, w)


def sample_block_mask(
    orig_shape: List[int],
    mask_shape: List[int],
    min_mask_size: int,
    acceptable_regions: Optional[List[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
):
    def constrain_mask(mask, tries=0):
        """Helper to restrict given mask to a set of acceptable regions"""
        N = max(int(len(acceptable_regions) - tries), 0)
        for k in range(N):
            mask *= acceptable_regions[k]

    tries = 0
    timeout = og_timeout = 20
    valid_mask = False
    while not valid_mask:
        mask_start = [
            torch.randint(0, orig_size - mask_size + 1, (1,), generator=generator)
            for orig_size, mask_size in zip(orig_shape, mask_shape)
        ]
        mask = torch.zeros(orig_shape, dtype=torch.int32)
        mask_complement = torch.ones_like(mask)
        slices = tuple([slice(start, start + size) for start, size in zip(mask_start, mask_shape)])
        mask[slices] = 1
        if acceptable_regions is not None:
            constrain_mask(mask, tries)
        mask = torch.nonzero(mask.flatten())
        valid_mask = len(mask) >= min_mask_size
        if not valid_mask:
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = og_timeout
                logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
    mask = mask.squeeze(-1)
    mask_complement = torch.ones(orig_shape, dtype=torch.int32)
    mask_complement[slices] = 0
    return mask, mask_complement


def sample_masks(
    batch_size: int,
    orig_shape: List[int],
    context_mask_shape: List[int],
    num_context_masks: int,
    target_mask_shape: List[int],
    num_target_masks: int,
    min_mask_size: int,
    allow_mask_overlap: bool,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    assert len(orig_shape) == len(context_mask_shape) and len(orig_shape) == len(target_mask_shape)
    min_keep_context_patches, min_keep_target_patches = (np.prod(orig_shape),) * 2
    collated_context_masks, collated_target_masks = [], []

    for _ in range(batch_size):
        target_masks_cached = []
        target_masks_complement = []
        for _ in range(num_target_masks):
            mask, mask_complement = sample_block_mask(
                orig_shape=orig_shape, mask_shape=target_mask_shape, min_mask_size=min_mask_size, generator=generator
            )
            target_masks_cached.append(mask)
            target_masks_complement.append(mask_complement)
            min_keep_target_patches = min(min_keep_target_patches, len(mask))
        collated_target_masks.append(target_masks_cached)

        acceptable_regions = None if allow_mask_overlap else target_masks_complement

        context_masks_cached = []
        for _ in range(num_context_masks):
            mask, _ = sample_block_mask(
                orig_shape=orig_shape,
                mask_shape=context_mask_shape,
                min_mask_size=min_mask_size,
                acceptable_regions=acceptable_regions,
                generator=generator,
            )
            context_masks_cached.append(mask)
            min_keep_context_patches = min(min_keep_context_patches, len(mask))
        collated_context_masks.append(context_masks_cached)

    collated_context_masks = [[cm[:min_keep_context_patches] for cm in masks] for masks in collated_context_masks]
    collated_target_masks = [[cm[:min_keep_target_patches] for cm in masks] for masks in collated_target_masks]

    context_masks = data.default_collate(collated_context_masks)
    target_masks = data.default_collate(collated_target_masks)

    context_masks = [context_mask.to(device) for context_mask in context_masks]
    target_masks = [target_mask.to(device) for target_mask in target_masks]

    return context_masks, target_masks

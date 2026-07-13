# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from functools import partial
from typing import Callable, Optional, List, Literal
from omegaconf import DictConfig

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.model import SignalTransformer

from .layers import PatchEmbed1d

log = get_pylogger(__name__)


class XelaTransformer(SignalTransformer):
    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        head: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
        normalization: Optional[DictConfig] = None,
    ):
        self.in_dim: int = in_dim
        self.in_chans: int = in_chans
        self.sequence_length: int = sequence_length
        self.time_chunk_size: int = time_chunk_size
        self.num_chunks: int = int(sequence_length // time_chunk_size)
        assert sequence_length % time_chunk_size == 0, "sequence length must be divisible by patch size"

        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            time_chunk_size=self.time_chunk_size,
            sequence_length=self.sequence_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pos_embed_fn=pos_embed_fn,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            with_masktoken=with_masktoken,
            causal=causal,
        )

        if normalization is not None:
            self.register_buffer("xela_mean", torch.as_tensor(normalization.mean, dtype=torch.float32))
            self.register_buffer("xela_std", torch.as_tensor(normalization.std, dtype=torch.float32))
        else:
            self.register_buffer("xela_mean", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
            self.register_buffer("xela_std", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        print(f"Xela mean: {self.xela_mean}, Xela std: {self.xela_std}")
        self.patch_embed = PatchEmbed1d(
            modal_chans=in_chans,
            modal_lens=sequence_length,
            chunk_size=self.time_chunk_size,
            embed_dim=self.embed_dim,
        )
        # self.patch_embed = nn.Linear(in_chans, self.embed_dim)
        self.taxeltypes = ["4x4", "4x6", "curved"]
        self.taxeltype_embed = nn.Parameter(torch.zeros(3, self.embed_dim))

        self.head = nn.Identity() if head is None else head

        nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)
        self.init_weights()

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == 3
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize(self, x: torch.Tensor):
        if hasattr(self, "xela_mean") and hasattr(self, "xela_std"):
            if self.in_chans == 3:
                x = (x - self.xela_mean) / self.xela_std
            elif self.in_chans == 6:
                # First three channels are the Xela values, rest are sensor positions
                xela_mean = torch.cat([self.xela_mean, torch.zeros_like(self.xela_mean)], dim=-1)
                xela_std = torch.cat([self.xela_std, torch.ones_like(self.xela_std)], dim=-1)
                x = (x - xela_mean) / xela_std
            else:
                raise ValueError("Bad number of channels, must be 3 or 6")
        return x

    def pre_embed(self, x: torch.Tensor):
        b = x.shape[0]
        x = self.normalize(x)

        x = einops.rearrange(x, "b t n c -> (b n) c t")

        sensor_embed = self.patch_embed(x)
        sensor_embed = einops.rearrange(sensor_embed, "(b n) c t -> b t n c", b=b)

        # We add a learnable embedding to identify different types of xela taxels
        prev_idx = 0
        for i, (k, v) in enumerate(XELA_FLATTEN_ORDER.items()):
            x = None
            if "4x4" in k:
                x = self.taxeltype_embed[0]
            elif "4x6" in k:
                x = self.taxeltype_embed[1]
            elif "aftc" in k:
                x = self.taxeltype_embed[2]
            else:
                raise ValueError("Bad taxel type")
            sensor_embed[..., prev_idx : prev_idx + v, :] += x[None, None, :]
            prev_idx += v
        return sensor_embed

    def create_causal_mask(self, x):
        """
        Create lower triangular block mask for Xela signals
        """
        _, chunked_t, n, _ = x.shape
        bias_size = chunked_t * n + self.num_register_tokens
        bias_size_multiple = int((bias_size // 8 + 1) * 8)  # cutlassF needs size to be multiple of 8
        attn_bias = torch.ones(
            (1, self.num_heads, bias_size, bias_size_multiple),
            dtype=torch.float32,
            device=x.device,
        )[..., :bias_size]

        # Mask out the future tokens
        attn_bias[..., self.num_register_tokens :, self.num_register_tokens :] = attn_bias[
            ..., self.num_register_tokens :, self.num_register_tokens :
        ].tril()

        # Prevent patch tokens from piggybacking on register tokens
        attn_bias[..., self.num_register_tokens :, : self.num_register_tokens] = 0

        # Create block causal mask
        for i in range(chunked_t):
            start = i * n + self.num_register_tokens
            end = (i + 1) * n + self.num_register_tokens
            attn_bias[..., start:end, start:end] = 1

        # Convert to additive bias
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)

        return attn_bias


def xela_tinier(
    in_dim: int,
    in_chans: List[int],
    sequence_length,
    depth=8,
    num_register_tokens=0,
    time_chunk_size=5,
    **kwargs,
):
    model = XelaTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=96,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def xela_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=8,
    num_register_tokens=0,
    time_chunk_size=5,
    **kwargs,
):
    model = XelaTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def xela_small(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=12,
    num_register_tokens=0,
    time_chunk_size=5,
    **kwargs,
):
    model = XelaTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def xela_base(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=12,
    num_register_tokens=0,
    time_chunk_size=5,
    **kwargs,
):
    model = XelaTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

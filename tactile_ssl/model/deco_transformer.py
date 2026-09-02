from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

from tactile_ssl.data.deco_geometry import NUM_HYPERTAXELS, build_deco_geometry

from .signal_transformer import SignalTransformer


class DecoHypertaxelTokenizer(nn.Module):
    """Map a three-frame, 4/5-member hypertaxel to one embedding."""

    MODES = {"mean_mlp", "spatial_conv_mlp", "direct"}

    def __init__(
        self,
        sequence_length: int = 3,
        embed_dim: int = 192,
        mode: Literal["mean_mlp", "spatial_conv_mlp", "direct"] = "direct",
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported DECO tokenizer {mode!r}; choose from {sorted(self.MODES)}")
        self.sequence_length = int(sequence_length)
        self.embed_dim = int(embed_dim)
        self.mode = str(mode)
        hidden_dim = int(hidden_dim or embed_dim)

        geometry = build_deco_geometry()
        self.register_buffer(
            "member_mask",
            torch.from_numpy(geometry.member_mask.copy()),
            persistent=True,
        )
        self.register_buffer(
            "group_size",
            torch.from_numpy(geometry.group_size.copy()).long(),
            persistent=True,
        )
        if self.mode in {"mean_mlp", "spatial_conv_mlp"}:
            self.temporal_mlp = nn.Sequential(
                nn.Linear(self.sequence_length, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim, eps=1e-6),
            )
        if self.mode == "spatial_conv_mlp":
            # One shared learned reduction for every 2x2 group and another for
            # the six 3x3 remainders on each hand. It is a Conv1d over ordered
            # members, applied independently at every time step.
            self.member_conv4 = nn.Conv1d(1, 1, kernel_size=4)
            self.member_conv5 = nn.Conv1d(1, 1, kernel_size=5)
        elif self.mode == "direct":
            self.direct4 = nn.Sequential(
                nn.Linear(4 * self.sequence_length, self.embed_dim),
                nn.LayerNorm(self.embed_dim, eps=1e-6),
            )
            self.direct5 = nn.Sequential(
                nn.Linear(5 * self.sequence_length, self.embed_dim),
                nn.LayerNorm(self.embed_dim, eps=1e-6),
            )

    def _validate(self, x: torch.Tensor) -> None:
        expected = (self.sequence_length, NUM_HYPERTAXELS, 5)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"DECO tokenizer expects [B, {expected[0]}, {expected[1]}, 5], got {tuple(x.shape)}")

    def _member_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean_mlp":
            mask = self.member_mask.to(dtype=x.dtype)[None, None]
            return (x * mask).sum(dim=-1) / self.group_size.to(dtype=x.dtype)[None, None]

        reduced = x.new_empty(x.shape[:3])
        for size, layer in ((4, self.member_conv4), (5, self.member_conv5)):
            group_indices = torch.nonzero(self.group_size == size, as_tuple=False).flatten()
            values = x[:, :, group_indices, :size]
            flat = values.reshape(-1, 1, size)
            encoded = layer(flat).reshape(x.shape[0], x.shape[1], -1)
            reduced[:, :, group_indices] = encoded.to(dtype=reduced.dtype)
        return reduced

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        if self.mode in {"mean_mlp", "spatial_conv_mlp"}:
            # [B,T,N] -> each node gets its complete temporal signal [T].
            return self.temporal_mlp(self._member_reduce(x).transpose(1, 2))

        output = x.new_empty((x.shape[0], NUM_HYPERTAXELS, self.embed_dim))
        for size, layer in ((4, self.direct4), (5, self.direct5)):
            group_indices = torch.nonzero(self.group_size == size, as_tuple=False).flatten()
            values = x[:, :, group_indices, :size].permute(0, 2, 1, 3).reshape(
                x.shape[0], group_indices.numel(), self.sequence_length * size
            )
            # Under mixed precision the input/output buffer can be BF16 while
            # an un-autocast tokenizer layer returns FP32. Index assignment does
            # not apply PyTorch's usual promotion rules, so cast explicitly.
            encoded = layer(values)
            output[:, group_indices] = encoded.to(dtype=output.dtype)
        return output


class DecoTransformer(SignalTransformer):
    """SignalTransformer whose 528 tokens are DECO hypertaxels."""

    def __init__(
        self,
        in_dim: int = NUM_HYPERTAXELS,
        in_chans: int = 5,
        time_chunk_size: int = 3,
        sequence_length: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        tokenizer_mode: Literal["mean_mlp", "spatial_conv_mlp", "direct"] = "direct",
        tokenizer_hidden_dim: Optional[int] = None,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        head: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: str = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
    ) -> None:
        if in_dim != NUM_HYPERTAXELS or in_chans != 5:
            raise ValueError("DECO model contract is in_dim=528 and in_chans=5")
        if sequence_length != 3 or time_chunk_size != 3:
            raise ValueError("DECO integration currently uses the agreed three-frame window")
        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            time_chunk_size=time_chunk_size,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            head=head,
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
        self.tokenizer = DecoHypertaxelTokenizer(
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            mode=tokenizer_mode,
            hidden_dim=tokenizer_hidden_dim,
        )

    def pre_embed(
        self,
        x: torch.Tensor,
        auxiliary_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if auxiliary_embedding is not None:
            raise ValueError("DECO does not support an auxiliary input embedding")
        return self.tokenizer(x).unsqueeze(1)


def _deco_transformer(embed_dim: int, depth: int, num_heads: int, **kwargs) -> DecoTransformer:
    return DecoTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)


def deco_tiny(**kwargs) -> DecoTransformer:
    return _deco_transformer(192, 12, 3, **kwargs)


def deco_small(**kwargs) -> DecoTransformer:
    return _deco_transformer(384, 12, 6, **kwargs)


def deco_base(**kwargs) -> DecoTransformer:
    return _deco_transformer(768, 12, 12, **kwargs)

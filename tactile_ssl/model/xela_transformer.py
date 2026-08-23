# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from functools import partial
from typing import Callable, Optional, List, Literal, Union
from omegaconf import DictConfig

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.model import SignalTransformer

from .layers import PatchEmbed1d

log = get_pylogger(__name__)


class AxisSeparatedPatchEmbed(nn.Module):
    """Embed a three-axis temporal window without mixing axes initially."""

    def __init__(
        self,
        sequence_length: int,
        embed_dim: int = 192,
        axis_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if embed_dim != 3 * axis_embed_dim:
            raise ValueError(
                "Axis-separated embedding requires "
                "embed_dim == 3 * axis_embed_dim"
            )
        self.sequence_length = int(sequence_length)
        self.axis_projections = nn.ModuleList(
            nn.Linear(self.sequence_length, axis_embed_dim) for _ in range(3)
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (3, self.sequence_length):
            raise ValueError(
                "AxisSeparatedPatchEmbed expects [batch, 3, sequence_length], "
                f"got {tuple(x.shape)}"
            )
        axes = [
            self.activation(projection(x[:, axis_index, :]))
            for axis_index, projection in enumerate(self.axis_projections)
        ]
        embedding = self.output_projection(torch.cat(axes, dim=-1))
        embedding = self.output_norm(embedding)
        return embedding.unsqueeze(-1)


class LSTMPatchEmbed(nn.Module):
    """Represent a three-axis temporal window by the final LSTM output."""

    def __init__(self, sequence_length: int, embed_dim: int = 192) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (3, self.sequence_length):
            raise ValueError(
                "LSTMPatchEmbed expects [batch, 3, sequence_length], "
                f"got {tuple(x.shape)}"
            )
        sequence = x.transpose(1, 2).contiguous()
        # deepcopy (target encoder) and DDP can invalidate cuDNN's packed RNN
        # weights. The check is cheap when they are already packed and avoids
        # repacking them on every sensor-window forward otherwise.
        self.lstm.flatten_parameters()
        output, _ = self.lstm(sequence)
        return output[:, -1, :].unsqueeze(-1)


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
        use_taxel_type_embedding: bool = True,
        input_fusion: Literal[
            "joint",
            "axis_separated",
            "lstm",
            "separate_coordinates",
            "fresh_random",
            "coordinates_only_patch",
            "coordinates_only_mean_patch",
        ] = "joint",
        signal_chans: int = 3,
        coordinate_chans: int = 3,
        random_embedding_std: float = 1.0,
        temporal_patch_padding: Union[int, Literal["auto"]] = "auto",
    ):
        self.in_dim: int = in_dim
        self.in_chans: int = in_chans
        self.sequence_length: int = sequence_length
        self.time_chunk_size: int = time_chunk_size
        self.num_chunks: int = int(sequence_length // time_chunk_size)
        assert sequence_length % time_chunk_size == 0, "sequence length must be divisible by patch size"
        self.input_fusion = str(input_fusion)
        self.signal_chans = int(signal_chans)
        self.coordinate_chans = int(coordinate_chans)
        self.random_embedding_std = float(random_embedding_std)
        self.temporal_patch_padding_mode = temporal_patch_padding
        self.temporal_patch_padding = (
            0
            if temporal_patch_padding == "auto"
            else int(temporal_patch_padding)
        )
        supported_input_fusions = {
            "joint",
            "axis_separated",
            "lstm",
            "separate_coordinates",
            "fresh_random",
            "coordinates_only_patch",
            "coordinates_only_mean_patch",
        }
        if self.input_fusion not in supported_input_fusions:
            raise ValueError(
                f"Unsupported input_fusion={self.input_fusion!r}; "
                f"expected one of {sorted(supported_input_fusions)}"
            )
        if self.input_fusion in {"separate_coordinates", "fresh_random"} and (
            self.in_chans != self.signal_chans + self.coordinate_chans
        ):
            raise ValueError(
                "Separate coordinate/random fusion requires "
                "in_chans == signal_chans + coordinate_chans"
            )
        if self.random_embedding_std <= 0:
            raise ValueError("random_embedding_std must be positive")
        if self.temporal_patch_padding not in {0, 2}:
            raise ValueError(
                "temporal_patch_padding must be 'auto', 0, or 2"
            )
        if self.input_fusion in {"axis_separated", "lstm"}:
            if self.in_chans != 3:
                raise ValueError(
                    f"{self.input_fusion} requires exactly three input channels"
                )
            if self.sequence_length != self.time_chunk_size:
                raise ValueError(
                    f"{self.input_fusion} requires one full-window temporal chunk"
                )
            if embed_dim != 192:
                raise ValueError(f"{self.input_fusion} requires embed_dim=192")

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
        # Version 0 is the historical padding=2 behavior. New models start at
        # version 1 (padding=0); old checkpoints have no marker and are treated
        # as version 0 while loading.
        self.register_buffer(
            "_temporal_patch_version",
            torch.tensor(
                1 if self.temporal_patch_padding == 0 else 0,
                dtype=torch.int8,
            ),
            persistent=True,
        )

        if normalization is not None:
            self.register_buffer("xela_mean", torch.as_tensor(normalization.mean, dtype=torch.float32))
            self.register_buffer("xela_std", torch.as_tensor(normalization.std, dtype=torch.float32))
        else:
            self.register_buffer("xela_mean", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
            self.register_buffer("xela_std", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        print(f"Xela mean: {self.xela_mean}, Xela std: {self.xela_std}")
        coordinate_only = self.input_fusion in {
            "coordinates_only_patch",
            "coordinates_only_mean_patch",
        }
        if coordinate_only and self.in_chans != self.signal_chans + self.coordinate_chans:
            raise ValueError(
                "Coordinate-only input requires "
                "in_chans == signal_chans + coordinate_chans"
            )
        patch_embed_chans = (
            self.coordinate_chans
            if coordinate_only
            else self.in_chans
            if self.input_fusion == "joint"
            else self.signal_chans
        )
        patch_modal_lens = (
            1 if self.input_fusion == "coordinates_only_mean_patch" else sequence_length
        )
        patch_chunk_size = (
            1 if self.input_fusion == "coordinates_only_mean_patch" else self.time_chunk_size
        )
        if self.input_fusion == "axis_separated":
            self.patch_embed = AxisSeparatedPatchEmbed(
                sequence_length=sequence_length,
                embed_dim=self.embed_dim,
            )
        elif self.input_fusion == "lstm":
            self.patch_embed = LSTMPatchEmbed(
                sequence_length=sequence_length,
                embed_dim=self.embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed1d(
                modal_chans=patch_embed_chans,
                modal_lens=patch_modal_lens,
                chunk_size=patch_chunk_size,
                embed_dim=self.embed_dim,
                # Temporal patches are non-overlapping. Padding would shift the
                # receptive field and leave trailing samples unused whenever a
                # sequence is exactly divisible by the chunk size.
                padding=(
                    0
                    if self.input_fusion == "coordinates_only_mean_patch"
                    else self.temporal_patch_padding
                ),
            )
        if self.input_fusion == "separate_coordinates":
            self.coordinate_patch_embed = PatchEmbed1d(
                modal_chans=self.coordinate_chans,
                modal_lens=sequence_length,
                chunk_size=self.time_chunk_size,
                embed_dim=self.embed_dim,
                padding=self.temporal_patch_padding,
            )
        else:
            self.coordinate_patch_embed = None
        self.input_fusion_projection = (
            nn.Linear(2 * self.embed_dim, self.embed_dim)
            if self.input_fusion in {"separate_coordinates", "fresh_random"}
            else None
        )
        # self.patch_embed = nn.Linear(in_chans, self.embed_dim)
        self.taxeltypes = ["4x4", "4x6", "curved"]
        self.use_taxel_type_embedding = bool(use_taxel_type_embedding)
        self.taxeltype_embed = nn.Parameter(
            torch.zeros(3, self.embed_dim),
            requires_grad=self.use_taxel_type_embedding,
        )

        self.head = nn.Identity() if head is None else head

        if self.use_taxel_type_embedding:
            nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)
        self.init_weights()

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == 3
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def _set_temporal_patch_padding(self, padding: int) -> None:
        """Apply a resolved padding mode after checkpoint-version detection."""
        self.temporal_patch_padding = int(padding)
        main_padding = (
            0
            if self.input_fusion == "coordinates_only_mean_patch"
            else self.temporal_patch_padding
        )
        if isinstance(self.patch_embed, PatchEmbed1d):
            self.patch_embed.proj.padding = (main_padding,)
        if isinstance(self.coordinate_patch_embed, PatchEmbed1d):
            self.coordinate_patch_embed.proj.padding = (
                self.temporal_patch_padding,
            )
        self._temporal_patch_version.fill_(
            1 if self.temporal_patch_padding == 0 else 0
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        marker_key = prefix + "_temporal_patch_version"
        marker = state_dict.get(marker_key)
        if self.temporal_patch_padding_mode == "auto":
            # Checkpoints created before the padding fix have no marker.
            checkpoint_version = 0 if marker is None else int(marker.item())
            self._set_temporal_patch_padding(
                0 if checkpoint_version >= 1 else 2
            )
        else:
            # An explicit config value always wins over checkpoint metadata.
            self._set_temporal_patch_padding(
                int(self.temporal_patch_padding_mode)
            )

        # Keep strict=True compatible with old checkpoints and make an
        # explicitly configured override persist if the model is saved again.
        state_dict[marker_key] = self._temporal_patch_version.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def sample_fresh_random_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one auxiliary token per temporal chunk and sensor."""
        b, sequence_length, num_sensors, _ = x.shape
        projection = self.patch_embed.proj
        chunked_length = (
            sequence_length
            + 2 * int(projection.padding[0])
            - int(projection.dilation[0]) * (int(projection.kernel_size[0]) - 1)
            - 1
        ) // int(projection.stride[0]) + 1
        return torch.randn(
            (b, chunked_length, num_sensors, self.embed_dim),
            device=x.device,
            dtype=x.dtype,
        ) * self.random_embedding_std

    def pre_embed(
        self,
        x: torch.Tensor,
        auxiliary_embedding: Optional[torch.Tensor] = None,
    ):
        b = x.shape[0]
        x = self.normalize(x)

        if self.input_fusion in {
            "coordinates_only_patch",
            "coordinates_only_mean_patch",
        }:
            signal = x[
                ...,
                self.signal_chans : self.signal_chans + self.coordinate_chans,
            ]
            if self.input_fusion == "coordinates_only_mean_patch":
                signal = signal.mean(dim=1, keepdim=True)
        elif self.input_fusion == "joint":
            signal = x
        else:
            signal = x[..., : self.signal_chans]
        signal = einops.rearrange(signal, "b t n c -> (b n) c t")

        sensor_embed = self.patch_embed(signal)
        sensor_embed = einops.rearrange(sensor_embed, "(b n) c t -> b t n c", b=b)

        if self.input_fusion == "separate_coordinates":
            coordinates = x[
                ...,
                self.signal_chans : self.signal_chans + self.coordinate_chans,
            ]
            coordinates = einops.rearrange(
                coordinates,
                "b t n c -> (b n) c t",
            )
            auxiliary_embedding = self.coordinate_patch_embed(coordinates)
            auxiliary_embedding = einops.rearrange(
                auxiliary_embedding,
                "(b n) c t -> b t n c",
                b=b,
            )
        elif self.input_fusion == "fresh_random":
            if auxiliary_embedding is None:
                auxiliary_embedding = self.sample_fresh_random_embedding(x)

        if self.input_fusion in {"separate_coordinates", "fresh_random"}:
            if auxiliary_embedding.shape != sensor_embed.shape:
                raise ValueError(
                    "Auxiliary embedding must match signal embedding shape; "
                    f"got {tuple(auxiliary_embedding.shape)} and "
                    f"{tuple(sensor_embed.shape)}"
                )
            auxiliary_embedding = auxiliary_embedding.to(
                device=sensor_embed.device,
                dtype=sensor_embed.dtype,
            )
            sensor_embed = self.input_fusion_projection(
                torch.cat([sensor_embed, auxiliary_embedding], dim=-1)
            )

        if self.use_taxel_type_embedding:
            # We add a learnable embedding to identify different types of xela taxels
            taxeltype_chunks = []
            prev_idx = 0
            for k, v in XELA_FLATTEN_ORDER.items():
                if "4x4" in k:
                    taxeltype_embed = self.taxeltype_embed[0]
                elif "4x6" in k:
                    taxeltype_embed = self.taxeltype_embed[1]
                elif "aftc" in k:
                    taxeltype_embed = self.taxeltype_embed[2]
                else:
                    raise ValueError("Bad taxel type")
                taxeltype_chunks.append(
                    taxeltype_embed.reshape(1, 1, 1, -1).expand(1, 1, v, -1)
                )
                prev_idx += v
            taxeltype_map = torch.cat(taxeltype_chunks, dim=2)
            sensor_embed = sensor_embed + taxeltype_map
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


class XelaLateFusionEncoder(nn.Module):
    """Frozen signal/coordinate backbones with a trainable per-taxel fusion patch."""

    supports_graph_info = False
    trainable_fusion = True

    def __init__(
        self,
        signal_encoder: nn.Module,
        coordinate_encoder: Optional[nn.Module] = None,
        fusion_mode: Literal["coordinate_encoder", "fresh_random"] = "coordinate_encoder",
        random_embedding_std: float = 1.0,
        **legacy_encoder_config: object,
    ) -> None:
        super().__init__()
        # Hydra recursively merges this node with the legacy single-encoder
        # task config. These keys are superseded by signal_encoder and retained
        # only so old task defaults can be reused without changing them.
        allowed_legacy_keys = {
            "in_dim",
            "in_chans",
            "sequence_length",
            "time_chunk_size",
            "num_register_tokens",
            "pos_embed_fn",
            "with_masktoken",
            "drop_path_rate",
            "drop_path_uniform",
            "normalization",
            "causal",
        }
        unknown_legacy_keys = set(legacy_encoder_config) - allowed_legacy_keys
        if unknown_legacy_keys:
            raise TypeError(
                "Unexpected late-fusion encoder options: "
                f"{sorted(unknown_legacy_keys)}"
            )
        if fusion_mode not in {"coordinate_encoder", "fresh_random"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; "
                "expected 'coordinate_encoder' or 'fresh_random'"
            )
        if fusion_mode == "coordinate_encoder" and coordinate_encoder is None:
            raise ValueError("coordinate_encoder is required for coordinate_encoder fusion")
        if random_embedding_std <= 0:
            raise ValueError("random_embedding_std must be positive")
        if signal_encoder.embed_dim != 192:
            raise ValueError(
                "Late fusion currently expects a 192D signal encoder, "
                f"got {signal_encoder.embed_dim}"
            )
        if coordinate_encoder is not None:
            for attribute in ("embed_dim", "sequence_length", "time_chunk_size", "in_dim"):
                if getattr(coordinate_encoder, attribute) != getattr(signal_encoder, attribute):
                    raise ValueError(f"Signal and coordinate encoders disagree on {attribute}")

        self.signal_encoder = signal_encoder
        self.coordinate_encoder = coordinate_encoder
        self.fusion_mode = fusion_mode
        self.random_embedding_std = float(random_embedding_std)
        self.in_chans = 6
        self.in_dim = signal_encoder.in_dim
        self.embed_dim = signal_encoder.embed_dim
        self.sequence_length = signal_encoder.sequence_length
        self.time_chunk_size = signal_encoder.time_chunk_size
        self.num_register_tokens = 0
        self.fusion_patch_embed = PatchEmbed1d(
            modal_chans=2 * self.embed_dim,
            modal_lens=1,
            chunk_size=1,
            embed_dim=self.embed_dim,
            padding=0,
        )
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.signal_encoder.requires_grad_(False)
        self.signal_encoder.eval()
        if self.coordinate_encoder is not None:
            self.coordinate_encoder.requires_grad_(False)
            self.coordinate_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen JEPA backbones must not re-enter stochastic training mode when
        # the containing downstream module is switched to train().
        self.signal_encoder.eval()
        if self.coordinate_encoder is not None:
            self.coordinate_encoder.eval()
        return self

    def _auxiliary_tokens(
        self,
        x: torch.Tensor,
        signal_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_mode == "fresh_random":
            return torch.randn_like(signal_tokens) * self.random_embedding_std
        assert self.coordinate_encoder is not None
        with torch.no_grad():
            return self.coordinate_encoder.forward_features(x)["x_norm_patchtokens"]

    def forward_features(self, x: torch.Tensor, **_: object):
        if x.ndim != 4 or x.shape[-1] < 6:
            raise ValueError(
                "Late-fusion encoder expects [batch, time, sensors, >=6] input"
            )
        with torch.no_grad():
            signal_tokens = self.signal_encoder.forward_features(
                x[..., :3]
            )["x_norm_patchtokens"]
        auxiliary_tokens = self._auxiliary_tokens(x[..., :6], signal_tokens)
        if auxiliary_tokens.shape != signal_tokens.shape:
            raise ValueError(
                "Signal and auxiliary token shapes must match; "
                f"got {tuple(signal_tokens.shape)} and {tuple(auxiliary_tokens.shape)}"
            )

        batch_size, token_count, embed_dim = signal_tokens.shape
        fused = torch.cat([signal_tokens, auxiliary_tokens], dim=-1)
        fused = einops.rearrange(fused, "b n c -> (b n) c 1")
        fused = self.fusion_patch_embed(fused)
        fused = einops.rearrange(
            fused,
            "(b n) c 1 -> b n c",
            b=batch_size,
            n=token_count,
        )
        empty_registers = fused.new_empty(batch_size, 0, embed_dim)
        return {
            "x_norm_regtokens": empty_registers,
            "x_norm_patchtokens": fused,
            "x_prenorm": fused,
        }

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

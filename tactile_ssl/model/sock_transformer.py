from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from tactile_ssl.model.xela_transformer import XelaTransformer


class SockTransformer(XelaTransformer):
    """Sensor-token transformer for the two physical SensTextile sock maps."""

    LEFT_SENSORS = 237
    RIGHT_SENSORS = 216

    def __init__(
        self,
        *args,
        normalization: Optional[DictConfig] = None,
        use_sensor_id_embedding: bool = True,
        use_foot_embedding: bool = True,
        **kwargs,
    ) -> None:
        kwargs["use_taxel_type_embedding"] = False
        super().__init__(*args, normalization=None, signal_chans=1, **kwargs)
        if self.in_dim != self.LEFT_SENSORS + self.RIGHT_SENSORS:
            raise ValueError(f"SockTransformer expects 453 physical sensors, got {self.in_dim}")
        if self.in_chans != 1:
            raise ValueError(f"SockTransformer expects one pressure channel, got {self.in_chans}")

        self.use_sensor_id_embedding = bool(use_sensor_id_embedding)
        self.use_foot_embedding = bool(use_foot_embedding)
        self.sensor_id_embed = nn.Parameter(
            torch.zeros(1, self.in_dim, self.embed_dim),
            requires_grad=self.use_sensor_id_embedding,
        )
        self.foot_embed = nn.Parameter(
            torch.zeros(2, self.embed_dim),
            requires_grad=self.use_foot_embedding,
        )
        foot_ids = torch.cat(
            [
                torch.zeros(self.LEFT_SENSORS, dtype=torch.long),
                torch.ones(self.RIGHT_SENSORS, dtype=torch.long),
            ]
        )
        self.register_buffer("foot_ids", foot_ids, persistent=True)

        if normalization is None or normalization.get("mean") is None:
            mean = torch.tensor([0.0], dtype=torch.float32)
            std = torch.tensor([1.0], dtype=torch.float32)
        else:
            mean = torch.as_tensor(normalization.get("mean"), dtype=torch.float32).reshape(1)
            std = torch.as_tensor(normalization.get("std"), dtype=torch.float32).reshape(1)
        self.xela_mean = mean
        self.xela_std = std.clamp_min(1e-6)

        if self.use_sensor_id_embedding:
            nn.init.trunc_normal_(self.sensor_id_embed, std=0.02)
        if self.use_foot_embedding:
            nn.init.trunc_normal_(self.foot_embed, std=0.02)

    def update_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != 1 or std.numel() != 1:
            raise ValueError("Sock normalization must contain one mean and one std")
        self.xela_mean = mean.reshape(1)
        self.xela_std = std.reshape(1).clamp_min(1e-6)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.xela_mean) / self.xela_std

    def get_position_embedding(self, device: torch.device) -> torch.Tensor:
        positional = super().get_position_embedding(device)
        chunks = self.sequence_length // self.time_chunk_size
        structural = positional.new_zeros(1, self.in_dim, self.embed_dim)
        if self.use_sensor_id_embedding:
            structural = structural + self.sensor_id_embed.to(dtype=positional.dtype)
        if self.use_foot_embedding:
            structural = structural + self.foot_embed[self.foot_ids].unsqueeze(0).to(
                dtype=positional.dtype
            )
        structural = structural.repeat(1, chunks, 1)
        return positional + structural


def sock_tiny(
    in_dim: int = 453,
    in_chans: int = 1,
    sequence_length: int = 5,
    depth: int = 8,
    num_register_tokens: int = 0,
    time_chunk_size: int = 5,
    **kwargs,
) -> SockTransformer:
    return SockTransformer(
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


def sock_small(**kwargs) -> SockTransformer:
    return SockTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)


def sock_base(**kwargs) -> SockTransformer:
    return SockTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)

"""Supervised spatial self-attention driven by relative sensor geometry."""

from typing import Optional

import torch
import torch.nn as nn

from tactile_ssl.model.layers.attention import Attention
from tactile_ssl.model.layers.block import Block


class XelaDistanceBiasedAttentionBlock(nn.Module):
    """Transformer block with a learned per-head bias from relative geometry."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        distance_hidden_dim: int = 16,
        coordinate_dim: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_std: float = 0.02,
        eps: float = 1e-8,
        use_distance_bias: bool = True,
        use_directional_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if distance_hidden_dim <= 0:
            raise ValueError("distance_hidden_dim must be positive")
        if coordinate_dim != 3:
            raise ValueError(f"Xela coordinates must have exactly three channels; got {coordinate_dim}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if use_directional_bias and not use_distance_bias:
            raise ValueError("use_directional_bias requires use_distance_bias")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.coordinate_dim = coordinate_dim
        self.use_distance_bias = use_distance_bias
        self.use_directional_bias = use_directional_bias
        self.eps = eps
        self.init_std = init_std

        self.distance_mlp = (
            nn.Sequential(
                nn.Linear(4 if use_directional_bias else 1, distance_hidden_dim),
                nn.GELU(),
                nn.Linear(distance_hidden_dim, num_heads),
            )
            if use_distance_bias
            else None
        )
        self.block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_class=Attention,
        )

        self.apply(self._init_weights)
        # Start from ordinary self-attention and let geometry enter gradually.
        if self.distance_mlp is not None:
            nn.init.zeros_(self.distance_mlp[-1].weight)
            nn.init.zeros_(self.distance_mlp[-1].bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_coordinates(self, spatial_coords: torch.Tensor) -> None:
        if spatial_coords.ndim != 3:
            raise ValueError(
                "spatial_coords must have shape [graphs, sensors, coordinates]; "
                f"got {tuple(spatial_coords.shape)}"
            )
        if spatial_coords.shape[1] < 2:
            raise ValueError("spatial attention requires at least two sensor tokens")
        if spatial_coords.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"expected {self.coordinate_dim} coordinate channels, got {spatial_coords.shape[-1]}"
            )
        if not torch.isfinite(spatial_coords).all():
            raise ValueError("spatial_coords must contain only finite values")

    def distance_bias(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        """Return additive attention bias with shape [G, H, N, N].

        Directional mode uses normalized ``(dx, dy, dz, distance)``. It is
        translation and scale invariant, but deliberately retains direction in
        the sensor coordinate frame.
        """
        if self.distance_mlp is None:
            raise RuntimeError("distance bias is disabled for this attention block")
        self._validate_coordinates(spatial_coords)
        distances = torch.cdist(spatial_coords.float(), spatial_coords.float(), p=2)
        max_distance = distances.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        normalized_distances = distances / max_distance
        geometry = normalized_distances.unsqueeze(-1)
        if self.use_directional_bias:
            relative_vectors = spatial_coords.float().unsqueeze(2) - spatial_coords.float().unsqueeze(1)
            normalized_vectors = relative_vectors / max_distance.unsqueeze(-1)
            geometry = torch.cat([normalized_vectors, geometry], dim=-1)
        mlp_dtype = self.distance_mlp[0].weight.dtype
        bias = self.distance_mlp(geometry.to(dtype=mlp_dtype))
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [graphs, sensors, channels]; got {tuple(tokens.shape)}")
        if tokens.shape[1] < 2:
            raise ValueError("spatial attention requires at least two sensor tokens")
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"expected token dimension {self.embed_dim}, got {tokens.shape[-1]}")

        attention_bias = None
        if self.use_distance_bias:
            if spatial_coords is None:
                raise ValueError("spatial_coords are required when use_distance_bias is true")
            if spatial_coords.shape[:2] != tokens.shape[:2]:
                raise ValueError(
                    "spatial_coords and tokens must have matching graph and sensor dimensions; "
                    f"got {tuple(spatial_coords.shape)} and {tuple(tokens.shape)}"
                )
            attention_bias = self.distance_bias(spatial_coords)

        return self.block(tokens, attn_bias=attention_bias)

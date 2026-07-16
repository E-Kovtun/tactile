import torch
from torch import nn


class LearnedConcatEmbeddingFusion(nn.Module):
    """Control fusion with a learned embedding that contains no sample or spatial information."""

    def __init__(
        self,
        signal_embed_dim: int,
        baseline_embedding_dim: int = 64,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if signal_embed_dim <= 0:
            raise ValueError("signal_embed_dim must be positive")
        if baseline_embedding_dim <= 0:
            raise ValueError("baseline_embedding_dim must be positive")

        self.signal_embed_dim = signal_embed_dim
        self.baseline_embedding_dim = baseline_embedding_dim
        self.embedding = nn.Parameter(torch.empty(baseline_embedding_dim))
        self.fusion = nn.Linear(signal_embed_dim + baseline_embedding_dim, signal_embed_dim)
        self.norm = nn.LayerNorm(signal_embed_dim)

        nn.init.trunc_normal_(self.embedding, std=init_std)
        nn.init.trunc_normal_(self.fusion.weight, std=init_std)
        nn.init.zeros_(self.fusion.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, signal_embedding: torch.Tensor) -> torch.Tensor:
        if signal_embedding.ndim < 2:
            raise ValueError(
                "signal_embedding must have at least batch and channel dimensions; "
                f"got {tuple(signal_embedding.shape)}"
            )
        if signal_embedding.shape[-1] != self.signal_embed_dim:
            raise ValueError(
                f"expected signal embedding dimension {self.signal_embed_dim}, "
                f"got {signal_embedding.shape[-1]}"
            )

        view_shape = (1,) * (signal_embedding.ndim - 1) + (self.baseline_embedding_dim,)
        baseline_embedding = self.embedding.to(dtype=signal_embedding.dtype).view(view_shape)
        baseline_embedding = baseline_embedding.expand(*signal_embedding.shape[:-1], -1)
        fused = self.fusion(torch.cat([signal_embedding, baseline_embedding], dim=-1))
        return self.norm(fused)

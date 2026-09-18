"""Canonical sparse 2D atlas geometry for the 368 Xela taxels."""

from __future__ import annotations

import torch

from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.data.xela_tdex.utils import XELA_IMG_ORDER


XELA_ATLAS_HEIGHT = 26
XELA_ATLAS_WIDTH = 24


def xela_atlas_coordinates() -> torch.Tensor:
    """Return canonical ``[row, column]`` coordinates in flattened sensor order."""
    coordinates: list[tuple[int, int]] = []
    for name, count in XELA_FLATTEN_ORDER.items():
        row, column = (int(value) for value in XELA_IMG_ORDER[name])
        if "aftc" in name:
            local = (
                [(0, 2), (0, 3)]
                + [(1, offset) for offset in range(1, 5)]
                + [(r, c) for r in range(2, 6) for c in range(6)]
            )
        elif "4x4" in name:
            local = [(r, c + 2) for r in range(4) for c in range(4)]
        elif "4x6" in name:
            local = [(r, c) for r in range(4) for c in range(6)]
        else:  # pragma: no cover - guarded by the canonical Xela layout
            raise ValueError(f"Unknown Xela pad type {name!r}")
        if len(local) != count:
            raise RuntimeError(f"Atlas layout for {name!r} has {len(local)} cells, expected {count}")
        coordinates.extend((row + r, column + c) for r, c in local)

    result = torch.tensor(coordinates, dtype=torch.long)
    if result.shape != (368, 2):
        raise RuntimeError(f"Expected 368 Xela atlas coordinates, got {tuple(result.shape)}")
    if torch.unique(result, dim=0).shape[0] != result.shape[0]:
        raise RuntimeError("Canonical Xela atlas maps multiple taxels to one cell")
    if result[:, 0].min() < 0 or result[:, 0].max() >= XELA_ATLAS_HEIGHT:
        raise RuntimeError("Xela atlas row is outside the declared canvas")
    if result[:, 1].min() < 0 or result[:, 1].max() >= XELA_ATLAS_WIDTH:
        raise RuntimeError("Xela atlas column is outside the declared canvas")
    return result


def xela_2d_sincos_position_embedding(embed_dim: int) -> torch.Tensor:
    """Return fixed 2D sine-cosine embeddings in canonical taxel order.

    This follows the factorization used by image ViTs/I-JEPA: half of the
    channels encode the atlas column and half encode the atlas row.
    """
    if embed_dim % 4 != 0:
        raise ValueError("2D sine-cosine embedding dimension must be divisible by 4")

    coordinates = xela_atlas_coordinates().to(torch.float32)

    def encode_1d(position: torch.Tensor, dim: int) -> torch.Tensor:
        frequencies = torch.arange(dim // 2, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (frequencies / (dim / 2.0)))
        angles = torch.einsum("n,d->nd", position, frequencies)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    half = embed_dim // 2
    columns = encode_1d(coordinates[:, 1], half)
    rows = encode_1d(coordinates[:, 0], half)
    return torch.cat((columns, rows), dim=1).unsqueeze(0)

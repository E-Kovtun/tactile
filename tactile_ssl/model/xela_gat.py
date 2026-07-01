from functools import partial
from typing import Callable, Optional, List, Literal
from omegaconf import DictConfig

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.model import SignalTransformer
from tactile_ssl.utils import apply_masks

from .layers import SinusoidalEmbed, SwiGLUFFNFused
from .layers import PatchEmbed1d

log = get_pylogger(__name__)

# The following are assumed to already exist in the same module/package as XelaLinear and are
# reused as-is here:
#   - PatchEmbed1d
#   - SinusoidalEmbed
#   - apply_masks
#   - XELA_FLATTEN_ORDER
# from .xela_linear import PatchEmbed1d, SinusoidalEmbed, apply_masks, XELA_FLATTEN_ORDER
# from omegaconf import DictConfig


def knn_graph_from_positions(pos: torch.Tensor, k: int, include_self: bool = True) -> torch.Tensor:
    """Build a boolean k-NN adjacency mask from taxel 3D positions.

    Connects each taxel to its k nearest neighbors by Euclidean distance. k-NN selection is
    rank-based, so it's invariant to the unit/scale of `pos` -- only the magnetic signal channels
    need normalization, not the positions.

    Args:
        pos: (..., n, 3) taxel positions.
        k: number of nearest neighbors per taxel (excluding itself).
        include_self: add a self-loop for every taxel. Recommended -- lets each node keep its own
            signal in the attention mix and avoids a degenerate softmax for isolated taxels.

    Returns:
        adj: (..., n, n) bool tensor. adj[..., i, j] is True if j is a neighbor of i (or i == j).
    """
    n = pos.shape[-2]
    k = min(k, n - 1)
    eye = torch.eye(n, device=pos.device, dtype=torch.bool)

    dist = torch.cdist(pos, pos)
    dist = dist.masked_fill(eye, float("inf"))

    _, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)
    adj = torch.zeros_like(dist, dtype=torch.bool)
    adj.scatter_(-1, knn_idx, True)

    # k-NN is directed in general (i can be j's nearest neighbor without the reverse holding);
    # symmetrize so attention can flow both ways along every spatial edge.
    adj = adj | adj.transpose(-1, -2)

    if include_self:
        adj = adj | eye

    return adj


class DenseGATLayer(nn.Module):
    """Multi-head graph attention layer (Velickovic et al., 2018) on dense batches.

    Tactile taxel graphs are small (tens to a few hundred nodes), so a dense (B, N, N) attention
    mask is simpler and just as fast as a sparse/scatter implementation here. Swap in
    torch_geometric's GATConv if you later need this to scale to much larger taxel counts.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.empty(num_heads, self.head_dim))  # query-side attention vector
        self.attn_r = nn.Parameter(torch.empty(num_heads, self.head_dim))  # neighbor-side attention vector
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and (in_dim == out_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) node features.
            adj: (B, N, N) bool adjacency, adj[b, i, j] = True if i attends to j.

        Returns:
            (B, N, out_dim) updated node features.
        """
        b, n, _ = x.shape
        h = self.proj(x).view(b, n, self.num_heads, self.head_dim)  # (B, N, H, D)

        score_i = torch.einsum("bnhd,hd->bnh", h, self.attn_l)  # (B, N, H)
        score_j = torch.einsum("bnhd,hd->bnh", h, self.attn_r)  # (B, N, H)
        e = score_i.unsqueeze(2) + score_j.unsqueeze(1)  # (B, N_i, N_j, H)
        e = self.leaky_relu(e)

        mask = adj.unsqueeze(-1)  # (B, N_i, N_j, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)  # normalize attention over neighbors j
        alpha = self.dropout(alpha)

        out = torch.einsum("bijh,bjhd->bihd", alpha, h).reshape(b, n, self.num_heads * self.head_dim)

        if self.residual:
            out = out + x
        return out


class XelaGAT(nn.Module):
    """Drop-in replacement for XelaLinear that encodes the magnetic signal with a GAT instead of a
    plain per-sensor temporal patch embed.

    Input `x` has shape (b, t, n, in_chans) where the last `pos_chans` channels are the taxel's
    fixed 3D position and the first `signal_chans` channels are the magnetic field reading
    (default in_chans=6 -> 3 signal + 3 position). Taxel positions are used only to build a k-NN
    spatial graph; they are never patch-embedded or treated as signal content.

    Pipeline: split signal/position -> patch-embed signal per taxel over time (same PatchEmbed1d
    as XelaLinear, now over signal_chans only) -> add taxel-type embedding -> run a small GAT
    stack across taxels, independently per time chunk with shared weights, using the k-NN graph
    built from taxel positions -> add positional embedding over (time chunk, taxel) -> optional
    masking -> flatten -> trunk blocks -> norm. Output dict matches XelaLinear's interface.
    """

    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int,
        num_register_tokens: int,
        signal_chans: int = 3,
        pos_chans: int = 3,
        k_neighbors: int = 6,
        gat_num_layers: int = 2,
        gat_num_heads: int = 4,
        gat_dropout: float = 0.0,
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        with_masktoken: bool = False,
        normalization: Optional["DictConfig"] = None,
    ):
        assert in_chans == signal_chans + pos_chans, "in_chans must equal signal_chans + pos_chans"
        assert embed_dim % gat_num_heads == 0, "embed_dim must be divisible by gat_num_heads"

        self.in_dim: int = in_dim  # number of taxels
        self.in_chans: int = in_chans
        self.signal_chans: int = signal_chans
        self.pos_chans: int = pos_chans
        self.sequence_length: int = sequence_length
        self.embed_dim = embed_dim
        self.time_chunk_size: int = time_chunk_size
        self.k_neighbors = k_neighbors

        super().__init__()

        if normalization is not None:
            self.register_buffer("xela_mean", torch.tensor(normalization.mean))
            self.register_buffer("xela_std", torch.tensor(normalization.std))
        else:
            self.register_buffer("xela_mean", torch.tensor([0, 0, 0]))
            self.register_buffer("xela_std", torch.tensor([1, 1, 1]))
        print(f"Xela mean: {self.xela_mean}, Xela std: {self.xela_std}")

        self.patch_embed = PatchEmbed1d(
            modal_chans=self.signal_chans,
            modal_lens=sequence_length,
            chunk_size=self.time_chunk_size,
            embed_dim=self.embed_dim,
        )

        self.taxeltypes = ["4x4", "4x6", "curved"]
        self.taxeltype_embed = nn.Parameter(torch.zeros(3, self.embed_dim))

        self.gat_layers = nn.ModuleList(
            [
                DenseGATLayer(
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    num_heads=gat_num_heads,
                    dropout=gat_dropout,
                    residual=True,
                )
                for _ in range(gat_num_layers)
            ]
        )
        self.gat_activation = nn.ELU()

        self.head = nn.Identity()
        self.register_tokens = None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if with_masktoken else None

        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)

        # self.init_pos_embed(pos_embed_fn)

        blocks_list = [nn.Linear(embed_dim, embed_dim, bias=False)]
        self.blocks = nn.ModuleList(blocks_list)

    def init_pos_embed(self, pos_embed_fn):
        self.pos_embed_fn = pos_embed_fn
        if pos_embed_fn == "sinusoidal":
            self.pos_embed = SinusoidalEmbed(
                [self.sequence_length, self.in_dim],
                [self.time_chunk_size, 1],
                embed_dim=self.embed_dim,
            )
        elif pos_embed_fn == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    (self.sequence_length // self.time_chunk_size) * self.in_dim,
                    self.embed_dim,
                )
            )

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == self.signal_chans
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize(self, x: torch.Tensor):
        if hasattr(self, "xela_mean") and hasattr(self, "xela_std"):
            # First three channels are the Xela values, rest are sensor positions
            xela_mean = torch.cat([self.xela_mean, torch.zeros_like(self.xela_mean, device=x.device)], dim=-1)
            xela_std = torch.cat([self.xela_std, torch.ones_like(self.xela_std, device=x.device)], dim=-1)
            x = (x - xela_mean) / xela_std
            # x = einops.rearrange(x, "b t n k c -> b t n (k c)")
        return x

    def build_graph(self, pos: torch.Tensor) -> torch.Tensor:
        """pos: (b, t, n, pos_chans) -> adj: (b, n, n) bool."""
        # Taxel positions are physically fixed; average over the (redundant) time axis to get one
        # reference layout per sample and smooth out any per-timestep numerical jitter. Swap for
        # pos[:, 0] if you'd rather just trust the first frame.
        # pos_ref = pos.mean(dim=1)
        return knn_graph_from_positions(pos, k=self.k_neighbors)

    def pre_embed(self, signal: torch.Tensor):
        b = signal.shape[0]

        signal = self.normalize(signal)
        signal = signal[..., : self.signal_chans]

        signal = einops.rearrange(signal, "b t n c -> (b n) c t")
        sensor_embed = self.patch_embed(signal)
        sensor_embed = einops.rearrange(sensor_embed, "(b n) c t -> b t n c", b=b)

        # Taxel-type embedding, same convention as XelaLinear.
        prev_idx = 0
        for key, v in XELA_FLATTEN_ORDER.items():
            if "4x4" in key:
                te = self.taxeltype_embed[0]
            elif "4x6" in key:
                te = self.taxeltype_embed[1]
            elif "aftc" in key:
                te = self.taxeltype_embed[2]
            else:
                raise ValueError("Bad taxel type")
            sensor_embed[..., prev_idx : prev_idx + v, :] += te[None, None, :]
            prev_idx += v

        return sensor_embed


    def graph_embed(self, sensor_embed, pos):

        adj = self.build_graph(pos)  # (b, n, n)
        # b = sensor_embed.shape[0]
        # n = sensor_embed.shape[1]
        # Graph attention across taxels, applied independently per time chunk with shared weights.
        # t, n = sensor_embed.shape[1], sensor_embed.shape[2]
        # h = einops.rearrange(sensor_embed, "b t n c -> (b t) n c")
        # adj_bt = adj.unsqueeze(1).expand(b, t, n, n).reshape(b * t, n, n)
        h = sensor_embed[...]
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, adj)
            if i < len(self.gat_layers) - 1:
                h = self.gat_activation(h)
        # sensor_embed = einops.rearrange(h, "(b t) n c -> b t n c", b=b)

        return h

    def apply_tubelet_masks(self, x, masks, concat=True):
        all_x = []
        _, t, _, c = x.shape
        for mask in masks:
            mask_keep = einops.repeat(mask, "b n -> b t n c", c=c, t=t)
            masked_x = torch.gather(x, dim=-2, index=mask_keep)
            all_x.append(masked_x)
        if not concat:
            return all_x
        return torch.cat(all_x, dim=0)

    def apply_masktokens(self, x, masktoken_masks):
        assert self.mask_token is not None, "Model does not have mask token"
        _, t, _, c = x.shape
        x = einops.rearrange(x, "b t n c -> b n t c")
        masks_flat = masktoken_masks.flatten(0, 1)
        masks_flat = einops.repeat(masks_flat, "b n -> b n t c", c=c, t=t)
        x = torch.where(masks_flat, self.mask_token, x)
        x = einops.rearrange(x, "b n t c -> b t n c")
        return x

    def prepare_tokens_with_mask(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
    ):
        t, n = x.shape[-3], x.shape[-2]

        assert t <= self.sequence_length, (
            f"Input sequence length {t} is greater than model sequence length {self.sequence_length}"
        )

        # if self.pos_embed_fn == "sinusoidal":
        #     pos_embed = self.pos_embed(x.device).float().unsqueeze(0)
        # elif self.pos_embed_fn == "learned":
        #     pos_embed = self.pos_embed.float()
        # else:
        #     raise NotImplementedError("Unknown position embeding function")

        # pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", n=n)
        # x = x + pos_embed[:, :t]

        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")

        attn_bias = None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

        x = einops.rearrange(x, "b t n c -> b (t n) c")
        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias

    def prepare_tokens_with_mask_pos(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
    ):
        t, n = x.shape[-3], x.shape[-2]

        assert t <= self.sequence_length, (
            f"Input sequence length {t} is greater than model sequence length {self.sequence_length}"
        )

        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")

        attn_bias = None

        # if masktoken_masks is not None:
        #     x = self.apply_masktokens(x, masktoken_masks)

        x = einops.rearrange(x, "b t n c -> b (t n) c")
        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias

    def transform(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return x, x_norm

    def forward_features(
        self,
        x,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):

        # signal = x[..., : self.signal_chans]
        pos = torch.mean(x[..., self.signal_chans :], dim=1, keepdim=True)

        x = self.pre_embed(x)
        # print('SHAPE AFTER PRE_EMBED', x.shape)
        # print('POS SHAPE', pos.shape)
        x_mask, _ = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        pos_mask, _ = self.prepare_tokens_with_mask_pos(pos, masks, mask_type, masktoken_masks)
        # print('SHAPE AFTER MASKING', x_mask.shape, pos_mask.shape)
        out = self.graph_embed(x_mask, pos_mask)
        x_prenorm, x_postnorm = self.transform(out)

        reg_tokens = torch.mean(x_postnorm, dim=-2, keepdim=True)
        patch_tokens = x_postnorm[:, :, :]
        patch_tokens_prenorm = x_prenorm[:, :, :]

        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }
        return out

    def forward(self, x, masks=None, mask_type=None, masktoken_masks=None):
        out = self.forward_features(x, masks, mask_type, masktoken_masks)
        return self.head(out["x_norm_patchtokens"])
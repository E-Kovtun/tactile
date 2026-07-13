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


class XelaLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int,
        num_register_tokens: int,
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        with_masktoken: bool = False,
        normalization: Optional[DictConfig] = None,
    ):
        self.in_dim: int = in_dim # number of sensors
        self.in_chans: int = in_chans
        self.sequence_length: int = sequence_length
        self.embed_dim = embed_dim
        self.time_chunk_size: int = time_chunk_size

        super().__init__()

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

        self.head = nn.Identity() 
        self.register_tokens = None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if with_masktoken else None

        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)

        self.init_pos_embed(pos_embed_fn)

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
        elif (
            pos_embed_fn == "learned"
        ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    (self.sequence_length // self.time_chunk_size) * self.in_dim,
                    self.embed_dim,
                )
            )

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == 3
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize(self, x: torch.Tensor):
        if hasattr(self, "xela_mean") and hasattr(self, "xela_std"):
            x = einops.rearrange(x, "b t n (k c) -> b t n k c", c=self.in_chans)
            if self.in_chans == 3:
                x = (x - self.xela_mean) / self.xela_std
            elif self.in_chans == 6:
                # First three channels are the Xela values, rest are sensor positions
                xela_mean = torch.cat([self.xela_mean, torch.zeros_like(self.xela_mean)], dim=-1)
                xela_std = torch.cat([self.xela_std, torch.ones_like(self.xela_std)], dim=-1)
                x = (x - xela_mean) / xela_std
            else:
                raise ValueError("Bad number of channels, must be 3 or 6")
            x = einops.rearrange(x, "b t n k c -> b t n (k c)")
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

    # def create_causal_mask(self, x):
    #     # Create lower triangular mask for causal attention
    #     _, chunked_t, n, _ = x.shape
    #     bias_size = chunked_t * n + self.num_register_tokens
    #     bias_size_multiple = int((bias_size // 8 + 1) * 8)  # cutlassF needs size to be multiple of 8

    #     attn_bias = torch.ones(
    #         (1, self.num_heads, bias_size, bias_size_multiple),
    #         dtype=torch.float32,
    #         device=x.device,
    #     )[..., :bias_size]

    #     # Mask out the future
    #     attn_bias[..., self.num_register_tokens :, self.num_register_tokens :] = attn_bias[
    #         ..., self.num_register_tokens :, self.num_register_tokens :
    #     ].tril()

    #     # Prevent patch tokens from piggybacking on register tokens to cheat
    #     attn_bias[..., self.num_register_tokens :, : self.num_register_tokens] = 0

    #     attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
    #     attn_bias.masked_fill_(attn_bias == 1, 0)
    #     return attn_bias

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

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(x.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embeding function")

        pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", n=n)
        x = x + pos_embed[:, :t]
        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")
        # if self.causal:
        #     attn_bias = self.create_causal_mask(x)
        # else:
        #     attn_bias = None
        attn_bias = None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

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
        x = self.pre_embed(x)
        x, _ = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        x_prenorm, x_postnorm = self.transform(x)

        # print('x_prenorm', x_prenorm.shape)
        # print('x_postnorm', x_postnorm.shape)

        # reg_tokens = x_postnorm[:, : self.num_register_tokens]
        reg_tokens = torch.mean(x_postnorm, dim=-2, keepdim=True)
        # patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens = x_postnorm[:, :, :]
        # patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
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


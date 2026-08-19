# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import partial
from typing import Callable, Optional, Literal, List
from abc import abstractmethod

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils import apply_masks
from tactile_ssl.utils.logging import get_pylogger

from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block
from .layers import SinusoidalEmbed, SwiGLUFFNFused
from .layers import init_weights_vit_timm


log = get_pylogger(__name__)


class SignalTransformer(nn.Module):
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
        qkv_bias: bool = False,
        proj_bias: bool = False,
        ffn_bias: bool = False,
        head: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned", "given"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.in_chans = in_chans
        self.sequence_length = sequence_length
        self.time_chunk_size = time_chunk_size
        assert sequence_length % time_chunk_size == 0, "sequence length must be divisible by patch size"
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.causal = causal

        assert num_register_tokens >= 0, "Number of register tokens must be non-negative"
        self.num_register_tokens = num_register_tokens
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None
        )

        self.init_pos_embed(pos_embed_fn)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        ffn_layer_ = None
        if ffn_layer == "mlp":
            log.info("using MLP layer as FFN")
            ffn_layer_ = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            log.info("using SwiGLU layer as FFN")
            ffn_layer_ = SwiGLUFFNFused
        elif ffn_layer == "identity":
            log.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer_ = f
        else:
            raise NotImplementedError

        blocks_list = [
            Block(
                attn_class=MemEffAttention,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer_,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        self.head = nn.Identity() if head is None else head
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if with_masktoken else None

        self.init_weights()

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

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=1e-6)
        if self.mask_token is not None:
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(init_weights_vit_timm)

    def get_position_embedding(self, device: torch.device) -> torch.Tensor:
        """Return positional tokens as ``[1, time_chunks * sensors, dim]``.

        Sensor-specific encoders can extend this with structural embeddings
        without changing the generic token preparation or JEPA predictor.
        """
        if self.pos_embed_fn == "sinusoidal":
            return self.pos_embed(device).float().unsqueeze(0)
        if self.pos_embed_fn == "learned":
            return self.pos_embed.float()
        raise NotImplementedError("Unknown position embedding function")

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

    def create_causal_mask(self, x):
        # Create lower triangular mask for causal attention
        _, chunked_t, n, _ = x.shape
        bias_size = chunked_t * n + self.num_register_tokens
        bias_size_multiple = int((bias_size // 8 + 1) * 8)  # cutlassF needs size to be multiple of 8

        attn_bias = torch.ones(
            (1, self.num_heads, bias_size, bias_size_multiple),
            dtype=torch.float32,
            device=x.device,
        )[..., :bias_size]

        # Mask out the future
        attn_bias[..., self.num_register_tokens :, self.num_register_tokens :] = attn_bias[
            ..., self.num_register_tokens :, self.num_register_tokens :
        ].tril()

        # Prevent patch tokens from piggybacking on register tokens to cheat
        attn_bias[..., self.num_register_tokens :, : self.num_register_tokens] = 0

        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        return attn_bias

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

        pos_embed = self.get_position_embedding(x.device)

        pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", n=n)
        x = x + pos_embed[:, :t]
        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")
        if self.causal:
            attn_bias = self.create_causal_mask(x)
        else:
            attn_bias = None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

        x = einops.rearrange(x, "b t n c -> b (t n) c")
        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias

    def transform(self, x, bias):
        for blk in self.blocks:
            x = blk(x, bias)

        x_norm = self.norm(x)
        return x, x_norm

    @abstractmethod
    def pre_embed(self, x: torch.Tensor):
        raise NotImplementedError

    def get_self_attention(self, x, layer_id=None, masks=None, mask_type=None, masktoken_masks=None):
        x = self.pre_embed(x)
        x, bias = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)

        if layer_id is None:
            layer_id = len(self.blocks) - 1

        assert 0 <= layer_id < len(self.blocks), f"Layer ID {layer_id} out of range"
        for blk in self.blocks[:layer_id]:
            x = blk(x, bias)

        return self.blocks[layer_id](x, bias, return_attn=True)

    def forward_features(
        self,
        x,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
        **pre_embed_kwargs,
    ):
        x = self.pre_embed(x, **pre_embed_kwargs)
        x, bias = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        x_prenorm, x_postnorm = self.transform(x, bias)

        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }
        return out

    def forward(self, x, masks=None, mask_type=None, masktoken_masks=None, **pre_embed_kwargs):
        out = self.forward_features(
            x,
            masks,
            mask_type,
            masktoken_masks,
            **pre_embed_kwargs,
        )
        return self.head(out["x_norm_patchtokens"])


class SignalDecoder(SignalTransformer):
    def __init__(self, input_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim

        self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.input_dim)

        assert self.num_register_tokens == 0, "Decoder cannot have register tokens"
        assert self.mask_token is not None, "Decoder must have mask token"

        self.patch_embed = nn.Identity()
        super().init_weights()

    def pre_embed(self, x: torch.Tensor):
        n = x.shape[-2]
        x = einops.rearrange(x, "b t n c -> b (t n) c")
        x = self.input_projection(x)
        return einops.rearrange(x, "b (t n) c -> b t n c", n=n)

    @abstractmethod
    def prepare_tokens_with_mask(self, x, *args, **kwargs):
        raise NotImplementedError

    def transform(self, x, bias=None):
        return super().transform(x, bias)

    @abstractmethod
    def post_transform(self, x_prenorm, x_postnorm, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, *args, **kwargs):
        x = self.pre_embed(x)
        x = self.prepare_tokens_with_mask(x, *args, **kwargs)
        x_prenorm, x_postnorm = self.transform(x)
        out = self.post_transform(x_prenorm, x_postnorm, *args, **kwargs)
        return out


class SignalJEPAPredictor(SignalDecoder):
    def __init__(self, zero_init_mask_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if zero_init_mask_tokens:
            nn.init.zeros_(self.mask_token)

    def prepare_tokens_with_mask(self, x, context_masks, masks):
        b, chunked_t, _, _ = x.shape

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(x.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embedding function")
        pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", t=chunked_t)
        pos_embed = einops.repeat(pos_embed, "1 t n c -> b t n c", b=b)

        context_masked_pos_embed = self.apply_tubelet_masks(pos_embed, context_masks)
        x = x + context_masked_pos_embed

        x = einops.repeat(x, "(k b) t n c -> (p k b) t n c", p=len(masks), k=len(context_masks))

        # (p b) t n c <- b t n c * p masks
        prediction_token_pos_embed = self.apply_tubelet_masks(pos_embed, masks)
        predition_token_pos_embed = einops.repeat(
            prediction_token_pos_embed,
            "(p b) t n c -> (p k b) t n c",
            p=len(masks),
            k=len(context_masks),
        )
        prediction_tokens = einops.repeat(
            self.mask_token,
            "1 1 c -> b t n c",
            b=predition_token_pos_embed.shape[0],
            t=predition_token_pos_embed.shape[1],
            n=predition_token_pos_embed.shape[2],
        )
        prediction_tokens = prediction_tokens + predition_token_pos_embed
        prediction_tokens = einops.rearrange(prediction_tokens, "b t n c -> b (t n) c")
        x = einops.rearrange(x, "b t n c -> b (t n) c")
        x = torch.cat([x, prediction_tokens], dim=1)
        return x

    def post_transform(self, x_prenorm, x_postnorm, num_context_tokens, context_masks, masks):
        x = x_postnorm[:, num_context_tokens:]
        x = self.output_projection(x)
        x = einops.rearrange(
            x,
            "(p k b) (t n) c -> p (k b) t n c",
            p=len(masks),
            k=len(context_masks),
            n=masks[0].shape[-1],
        )
        return list(x)

    def forward(self, x, context_masks, masks, *args, **kwargs):
        assert context_masks is not None, "JEPA Predictor requires context masks"
        assert masks is not None, "JEPA Predictor requires masks"

        x = self.pre_embed(x, *args, **kwargs)
        _, t, n, _ = x.shape
        num_context_tokens = t * n
        x = self.prepare_tokens_with_mask(x, context_masks, masks, *args, **kwargs)
        x_prenorm, x_postnorm = self.transform(x, *args, **kwargs)
        out = self.post_transform(
            x_prenorm,
            x_postnorm,
            *args,
            num_context_tokens=num_context_tokens,
            context_masks=context_masks,
            masks=masks,
            **kwargs,
        )
        return out


class SignalReconstructionDecoder(SignalTransformer):
    def __init__(self, input_dim, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.decoder_pred = nn.Linear(self.embed_dim, self.time_chunk_size * out_chans)

        self.input_projection = nn.Linear(input_dim, self.embed_dim)
        self.patch_embed = nn.Identity()
        self.init_weights()

    def forward(self, x):
        n = x.shape[-2]
        x = self.input_projection(x)
        x = einops.rearrange(x, "b t n c -> b (t n) c")

        time_embed = self.time_embed(x.device)
        time_embed = einops.repeat(time_embed, "t c -> 1 (t n) c", n=n)
        x = x + time_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = einops.rearrange(x, "b (t n) (k c) -> b (t k) n c", n=n, k=self.time_chunk_size)
        return x


class SignalMAEDecoder(SignalTransformer):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.decoder_pred = nn.Linear(self.embed_dim, self.time_chunk_size * self.in_chans)

        self.input_projection = nn.Linear(input_dim, self.embed_dim)
        self.patch_embed = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.init_weights()

    def forward(self, x, pose, ids_restore):
        x = self.input_projection(x)

        mask_tokens = einops.repeat(
            self.mask_token,
            "1 1 c -> b t n c",
            b=x.shape[0],
            t=x.shape[1],
            n=ids_restore.shape[-1] - x.shape[-2],
        )
        x_ = torch.cat([x, mask_tokens], dim=-2)
        x_ = einops.rearrange(x_, "b t n c -> b c t n")
        ids_restore = einops.repeat(ids_restore, "b n -> b c t n", c=x_.shape[-3], t=x_.shape[-2])
        x = torch.gather(x_, dim=-1, index=ids_restore)  # unshuffle
        n = x.shape[-1]

        if self.pose_embed:
            time_embed = self.time_embed(x.device)
            time_embed = einops.repeat(time_embed, "t c -> 1 c t n ", n=n)
            pose_encoding = self.compute_pose_encoding(pose)
            pose_encoding = einops.rearrange(pose_encoding, "b (t n) c -> b (c n) t", n=n)
            pose_encoding = nn.functional.interpolate(
                pose_encoding, scale_factor=1 / self.time_chunk_size, mode="linear"
            )
            pose_encoding = einops.rearrange(pose_encoding, "b (c n) t -> b c t n", n=n)
            x = x + pose_encoding + time_embed
        else:
            pos_embed = self.pos_embed(x_.device)
            pos_embed = einops.rearrange(pos_embed, "(t n) c -> 1 c t n", t=x_.shape[-2], n=n)
            x = x + pos_embed

        x = einops.rearrange(x, "b c t n -> b (t n) c")

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = einops.rearrange(x, "b (t n) c -> b n t c", n=n)
        return x


def signal_jepa_predictor(
    input_dim,
    in_chans,
    sequence_length,
    time_chunk_size=5,
    num_heads=6,
    embed_dim=384,
    **kwargs,
):
    model = SignalJEPAPredictor(
        input_dim=input_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        num_heads=num_heads,
        embed_dim=embed_dim,
        **kwargs,
    )
    return model


def signal_mae_decoder(
    input_dim,
    in_chans,
    num_chunks,
    depth,
    time_chunk_size=5,
    num_heads=12,
    embed_dim=192,
):
    model = SignalMAEDecoder(
        input_dim=input_dim,
        in_chans=in_chans,
        sequence_length=int(num_chunks * time_chunk_size),
        time_chunk_size=time_chunk_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        ffn_layer="mlp",
        num_register_tokens=0,
    )
    return model


def ret_reconstruction_decoder(
    input_dim,
    in_chans,
    out_chans,
    sequence_length,
    time_chunk_size=5,
    num_heads=12,
    embed_dim=192,
    depth=2,
    **kwargs,
):
    model = SignalReconstructionDecoder(
        input_dim=input_dim,
        in_chans=in_chans,
        out_chans=out_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        ffn_layer="mlp",
        num_register_tokens=0,
        **kwargs,
    )
    return model

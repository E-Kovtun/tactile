# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Reference:
# https://github.com/facebookresearch/sparsh/blob/main/tactile_ssl/downstream_task/force_sl.py

"""
3-axis Force Regression Module for Tactile Sensing

This module implements force regression module for different tactile sensor types:
- Vision-based tactile sensors (e.g., standard DIGIT)
- Digit 360
- Magnetic-based tactile sensors (e.g., Xela)

The module provides specialized implementations for estimating 3D force vectors (Fx, Fy, Fz)
or only normal forces (Fz) from tactile sensor readings using Sparsh embeddings.
"""

from typing import Any, Dict, Optional, List, Literal
from functools import partial
import einops

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule
from tactile_ssl.downstream_task.d360_sl import D360SLModule
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.downstream_task.concat_embedding_baseline import LearnedConcatEmbeddingFusion
from tactile_ssl.downstream_task.spatial_distance_attention import XelaDistanceBiasedAttentionBlock
from tactile_ssl.downstream_task.spatial_gatv2 import XelaSpatialGATv2Encoder
from tactile_ssl.downstream_task.spatial_wl_mlp import XelaSpatialWLMLPEncoder
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed
from tactile_ssl.model.xela_transformer import XelaTransformer
from tactile_ssl.data.xela.utils import get_pad_xela_indexes

from tactile_ssl.utils.plotting_forces import plot_correlation, plot_forces_error
from tactile_ssl.model import VIT_EMBED_DIMS

from tactile_ssl.model.d360_transformer import D360Transformer

log = get_pylogger(__name__)


class ForceLinearProbe(nn.Module):
    def __init__(
        self,
        embed_dim="base",
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        with_last_activations=False,
        only_normal_force=False,
    ):
        super().__init__()
        self.only_normal_force = only_normal_force
        self.n_outputs = 1 if only_normal_force else 3

        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )

        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.n_outputs),
        )
        self.with_last_activations = with_last_activations

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.probe(x)
        if self.only_normal_force:
            x = F.sigmoid(x) if self.with_last_activations else x
        else:
            if self.with_last_activations:
                x[:, -1] = F.sigmoid(x[:, -1])
                x[:, 0:2] = F.tanh(x[:, 0:2])
        return x


class ForceSLModule(SLModule):
    def __init__(
        self,
        model_encoder: nn.Module,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        checkpoint_encoder: Optional[str] = None,
        checkpoint_coordinate_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "jepa",
        signal_encoder_type: Optional[str] = None,
        coordinate_encoder_type: Optional[str] = None,
        encoder_normalization_float32: bool = False,
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_coordinate_encoder=checkpoint_coordinate_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
            signal_encoder_type=signal_encoder_type,
            coordinate_encoder_type=coordinate_encoder_type,
            encoder_normalization_float32=encoder_normalization_float32,
        )
        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []
        self.only_normal_force = self.model_task.only_normal_force

    def forward(self, x: torch.Tensor):
        z = self.model_encoder(x)
        if self.train_encoder:
            y_pred = self.model_task(z)
        else:
            y_pred = self.model_task(z.detach())
        return y_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        y_gt = batch["force"]

        if self.only_normal_force:
            y_gt = y_gt[:, 2].unsqueeze(1)

        y_pred = self.forward(x)
        loss = F.smooth_l1_loss(y_pred, y_gt)

        y_pred = y_pred.detach()
        y_gt = y_gt.detach()
        mse_xyz = F.mse_loss(y_pred, y_gt, reduction="none").mean(dim=0)

        if self.only_normal_force:
            y_out = torch.zeros_like(batch["force"]).to(y_pred.device)
            y_out[:, 2] = y_pred.squeeze(1)
            y_pred = y_out

        return {
            "loss": loss,
            "rmse_xyz": torch.sqrt(mse_xyz),
            "y_pred": y_pred,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if (
            trainer_instance is not None
            and trainer_instance.fabric.is_global_zero
            and trainer_instance.should_log
        ):
            trainer_instance.writer.add_scalar(f"{label}/loss", outputs["loss"], step)

            metric = "batch_rmse"

            if  self.only_normal_force:
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fz", outputs[f"{metric}"].item(), step)
            else:
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fx", outputs[f"{metric}"][0].item(), step)
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fy", outputs[f"{metric}"][1].item(), step)
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fz",outputs[f"{metric}"][2].item(), step)
            

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["force"])
        self.val_force_scale.append(batch["force_scale"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_validation_epoch_end(self, trainer_instance=None):
        forces_gt = torch.cat(self.val_gt, dim=0).cpu().numpy()
        forces_pred = torch.cat(self.val_pred, dim=0).cpu().numpy()
        force_scale = torch.cat(self.val_force_scale, dim=0).cpu().numpy()

        forces_gt = forces_gt * force_scale
        forces_pred = forces_pred * force_scale

        im_corr = plot_correlation(forces_gt, forces_pred)
        img_err, img_cone = plot_forces_error(forces_gt, forces_pred)

        # if trainer_instance is not None:
        #     trainer_instance.wandb.log(
        #         {
        #             "val/correlation": trainer_instance.wandb.Image(im_corr),
        #             "val/error": trainer_instance.wandb.Image(img_err),
        #             "val/error_cone": trainer_instance.wandb.Image(img_cone),
        #         }
        #     )

        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []


class D360ForceLinearProbe(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        complete_block: bool = True,
        with_last_activations: bool = False,
        attn_pooling: bool = False,
        only_normal_force: bool = False,
    ):
        super().__init__()
        self.attn_pooling = attn_pooling
        self.only_normal_force = only_normal_force
        self.n_outputs = 1 if only_normal_force else 3

        if attn_pooling:
            self.pooler = AttentivePooler(
                num_queries=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
            )

        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.n_outputs),
        )
        self.with_last_activations = with_last_activations

    def forward(self, x):
        if self.attn_pooling:
            z = self.pooler(x).squeeze(1)
        else:
            z = x.mean(dim=1)

        x = self.probe(z)
        if self.only_normal_force:
            x = F.sigmoid(x) if self.with_last_activations else x
        else:
            if self.with_last_activations:
                x[:, -1] = F.sigmoid(x[:, -1])
                x[:, 0:2] = F.tanh(x[:, 0:2])
        return x


class D360ForceSLModule(D360SLModule):
    def __init__(
        self,
        model_encoder: D360Transformer,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        sensors: Optional[List[str]] = None,
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "jepa",
        supervise_delta_force: bool = False,
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            sensors=sensors,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
        )
        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []
        self.only_normal_force = self.model_task.only_normal_force
        self.supervise_delta_force = supervise_delta_force

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        xs, _ = self.prepare_data(batch)
        force_gt = batch["force"]
        delta_force_gt = batch["delta_force"]
        force_scale = batch["force_scale"]
        y_gt = delta_force_gt if self.supervise_delta_force else force_gt

        y_pred = self.forward(xs)
        loss = F.smooth_l1_loss(y_pred, y_gt)

        y_pred = y_pred.detach() * force_scale
        y_gt = y_gt.detach() * force_scale
        force_mse_xyz = F.mse_loss(y_pred, y_gt, reduction="none").mean(dim=0)

        return {
            "loss": loss,
            "batch_rmse": torch.sqrt(force_mse_xyz),
            "y_pred": y_pred,
            "y_gt": y_gt,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.writer.add_scalar(f"{label}/loss", outputs["loss"], step)
            metric = "batch_rmse"

            if not self.only_normal_force:
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fx", outputs[f"{metric}"][0].item(), step)
                trainer_instance.writer.add_scalar(f"{label}/{metric}_Fy", outputs[f"{metric}"][1].item(), step)

            trainer_instance.writer.add_scalar(f"{label}/{metric}_Fz", outputs[f"{metric}"][-1].item(), step)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)  # type: ignore

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(outputs["y_gt"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")  # type: ignore

    def on_validation_epoch_end(self, trainer_instance=None):
        forces_gt = torch.cat(self.val_gt, dim=0).cpu().numpy()
        forces_pred = torch.cat(self.val_pred, dim=0).cpu().numpy()

        forces_gt = forces_gt
        forces_pred = forces_pred

        im_corr = plot_correlation(forces_gt, forces_pred)
        img_err, img_cone = plot_forces_error(forces_gt, forces_pred)

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "val/correlation": trainer_instance.wandb.Image(im_corr),
                    "val/error": trainer_instance.wandb.Image(img_err),
                    "val/error_cone": trainer_instance.wandb.Image(img_cone),
                }
            )

        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []


class XelaForceLinearProbe(nn.Module):
    def __init__(
        self,
        time_chunk_size: int,
        embed_dim="base",
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        n_outputs=3,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        with_last_activations=False,
        only_normal_force=False,
        pad_id=None,
        pooling_type: Literal["attention", "mean"] = "attention",
    ):
        super().__init__()
        self.only_normal_force = only_normal_force
        self.n_outputs = n_outputs if not only_normal_force else 1
        self.init_std = init_std
        self.pad_id = pad_id
        self.pooling_type = pooling_type

        if self.pad_id is not None:
            self.pad_range = get_pad_xela_indexes(pad_id)

        if self.pooling_type not in {"attention", "mean"}:
            raise ValueError(
                f"Unsupported pooling_type={self.pooling_type!r}; expected 'attention' or 'mean'"
            )

        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.pooler = (
            AttentivePooler(
                num_queries=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=1,
                norm_layer=norm_layer,
                qkv_bias=qkv_bias,
                init_std=init_std,
                complete_block=complete_block,
            )
            if self.pooling_type == "attention"
            else None
        )
        # self.blocks = nn.ModuleList(
        #     [
        #         Block(
        #             dim=embed_dim,
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             norm_layer=norm_layer,
        #             drop_path=0.1,
        #         )
        #         for _ in range(depth)
        #     ]
        # )
        self.layer_norm = norm_layer(embed_dim)
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.n_outputs),
        )
        self.with_last_activations = with_last_activations
        # self.pos_embed_fn = SinusoidalEmbed(10000, 1, embed_dim)

        # attn_bias = torch.ones(1, 1, 1000, 1000)
        # attn_bias = attn_bias.tril()
        # attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        # attn_bias.masked_fill_(attn_bias == 1, 0)
        # self.register_buffer("attn_bias", attn_bias)

        self.register_buffer("target_std", torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer("target_mean", torch.tensor([0.0, 0.0, 0.0]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def update_target_stats(self, target_mean, target_std, target_max):
        print(f"Updating target stats in ForceDecoder: {target_mean}, {target_std}, {target_max}")
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_max = target_max

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        return z

    def forward(self, z, spatial_coords=None, graph_info=None):
        z = self._prepare_tokens(z, spatial_coords, graph_info)

        if self.pad_id is not None:
            z = z[:, :, self.pad_range[0]:self.pad_range[1], :]

        b, t, _, c = z.shape
        if self.pooling_type == "attention":
            assert self.pooler is not None
            z = self.pooler(z.flatten(0, 1)).view(b, t, c)
        else:
            z = z.mean(dim=2)
        z = z.squeeze(1)

        # pos_embed = self.pos_embed_fn(z.device).float().unsqueeze(0)
        # z += pos_embed[:, : z.shape[1]]

        # for block in self.blocks:
        #     z = block(z, self.attn_bias[..., : z.shape[1], : z.shape[1]])
        z = self.layer_norm(z)

        y = self.probe(z)

        if self.only_normal_force:
            y = F.sigmoid(y) if self.with_last_activations else y
        else:
            if self.with_last_activations:
                y[..., -1] = F.sigmoid(y[..., -1])
                y[..., 0:2] = F.tanh(y[..., 0:2])
        
        return y


class ProjectedXelaForceLinearProbe(XelaForceLinearProbe):
    """Trainable low-dimensional adapter for wide frozen image features."""

    def __init__(self, input_embed_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_norm = nn.LayerNorm(int(input_embed_dim))
        output_embed_dim = self.layer_norm.normalized_shape[0]
        self.input_projection = nn.Linear(int(input_embed_dim), output_embed_dim)
        self.input_norm.apply(self._init_weights)
        self.input_projection.apply(self._init_weights)

    def forward(self, x, spatial_coords=None, graph_info=None):
        return super().forward(self.input_projection(self.input_norm(x)))


class XelaForceSpatialMLPProbe(XelaForceLinearProbe):
    """Force probe that learns a supervised spatial embedding from XYZ coordinates."""

    supports_spatial_coords = True

    def __init__(
        self,
        *args,
        spatial_hidden_dims: List[int],
        coordinate_dim: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if coordinate_dim <= 0:
            raise ValueError("coordinate_dim must be positive")
        if not spatial_hidden_dims:
            raise ValueError("spatial_hidden_dims must contain at least the output embedding dimension")
        if any(dim <= 0 for dim in spatial_hidden_dims):
            raise ValueError("all spatial_hidden_dims values must be positive")

        signal_embed_dim = self.layer_norm.normalized_shape[0]
        spatial_dims = [coordinate_dim, *spatial_hidden_dims]
        spatial_layers = []
        for layer_idx, (in_dim, out_dim) in enumerate(zip(spatial_dims[:-1], spatial_dims[1:])):
            spatial_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < len(spatial_dims) - 2:
                spatial_layers.append(nn.GELU())

        self.coordinate_dim = coordinate_dim
        self.spatial_hidden_dims = tuple(spatial_hidden_dims)
        self.spatial_encoder = nn.Sequential(*spatial_layers)
        self.spatial_norm = nn.LayerNorm(spatial_hidden_dims[-1])
        self.fusion = nn.Linear(signal_embed_dim + spatial_hidden_dims[-1], signal_embed_dim)
        self.fusion_norm = nn.LayerNorm(signal_embed_dim)

        self.spatial_encoder.apply(self._init_weights)
        self.spatial_norm.apply(self._init_weights)
        self.fusion.apply(self._init_weights)
        self.fusion_norm.apply(self._init_weights)

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaForceSpatialMLPProbe")
        if spatial_coords.shape[:-1] != z.shape[:-1]:
            raise ValueError(
                "spatial_coords and signal tokens must have matching batch, time, and sensor dimensions; "
                f"got {tuple(spatial_coords.shape)} and {tuple(z.shape)}"
            )
        if spatial_coords.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"expected {self.coordinate_dim} coordinate channels, got {spatial_coords.shape[-1]}"
            )

        spatial_embedding = self.spatial_norm(self.spatial_encoder(spatial_coords))
        fused = self.fusion(torch.cat([z, spatial_embedding], dim=-1))
        return self.fusion_norm(fused)


class XelaForceConcatEmbeddingBaselineProbe(XelaForceLinearProbe):
    """Force fusion baseline with a learned non-spatial embedding."""

    def __init__(self, *args, baseline_embedding_dim: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        signal_embed_dim = self.layer_norm.normalized_shape[0]
        self.baseline_fusion = LearnedConcatEmbeddingFusion(
            signal_embed_dim=signal_embed_dim,
            baseline_embedding_dim=baseline_embedding_dim,
            init_std=self.init_std,
        )

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        return self.baseline_fusion(z)


class XelaForceSpatialWLMLPProbe(XelaForceLinearProbe):
    """Force probe with physical-graph WL coordinate diffusion and supervised fusion."""

    supports_spatial_coords = True
    supports_spatial_graph = True

    def __init__(
        self,
        *args,
        spatial_hidden_dims: List[int],
        spatial_wl_layers: int = 2,
        coordinate_dim: int = 3,
        bridge_k: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        signal_embed_dim = self.layer_norm.normalized_shape[0]
        self.spatial_encoder = XelaSpatialWLMLPEncoder(
            spatial_hidden_dims=spatial_hidden_dims,
            wl_num_layers=spatial_wl_layers,
            coordinate_dim=coordinate_dim,
            bridge_k=bridge_k,
            init_std=self.init_std,
        )
        self.fusion = nn.Linear(signal_embed_dim + self.spatial_encoder.output_dim, signal_embed_dim)
        self.fusion_norm = nn.LayerNorm(signal_embed_dim)
        self.fusion.apply(self._init_weights)
        self.fusion_norm.apply(self._init_weights)

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaForceSpatialWLMLPProbe")
        if spatial_coords.shape[:-1] != z.shape[:-1]:
            raise ValueError(
                "spatial_coords and signal tokens must have matching batch, time, and sensor dimensions; "
                f"got {tuple(spatial_coords.shape)} and {tuple(z.shape)}"
            )

        batch_size, time_steps, num_nodes, coordinate_dim = spatial_coords.shape
        flat_coords = spatial_coords.reshape(batch_size * time_steps, num_nodes, coordinate_dim)
        spatial_embedding = self.spatial_encoder(flat_coords, graph_info)
        spatial_embedding = spatial_embedding.view(batch_size, time_steps, num_nodes, -1)
        fused = self.fusion(torch.cat([z, spatial_embedding], dim=-1))
        return self.fusion_norm(fused)


class XelaForceSpatialGATv2Probe(XelaForceLinearProbe):
    """Force probe with a supervised physical-graph GATv2 coordinate encoder."""

    supports_spatial_coords = True
    supports_spatial_graph = True

    def __init__(
        self,
        *args,
        spatial_hidden_dims: List[int],
        spatial_gat_heads: int = 4,
        spatial_gat_dropout: float = 0.0,
        coordinate_dim: int = 3,
        bridge_k: int = 4,
        edge_mode: str = "distance",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        signal_embed_dim = self.layer_norm.normalized_shape[0]
        self.spatial_encoder = XelaSpatialGATv2Encoder(
            spatial_hidden_dims=spatial_hidden_dims,
            gat_heads=spatial_gat_heads,
            gat_dropout=spatial_gat_dropout,
            coordinate_dim=coordinate_dim,
            bridge_k=bridge_k,
            edge_mode=edge_mode,
        )
        self.fusion = nn.Linear(signal_embed_dim + self.spatial_encoder.output_dim, signal_embed_dim)
        self.fusion_norm = nn.LayerNorm(signal_embed_dim)
        self.fusion.apply(self._init_weights)
        self.fusion_norm.apply(self._init_weights)

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaForceSpatialGATv2Probe")
        if spatial_coords.shape[:-1] != z.shape[:-1]:
            raise ValueError(
                "spatial_coords and signal tokens must have matching batch, time, and sensor dimensions; "
                f"got {tuple(spatial_coords.shape)} and {tuple(z.shape)}"
            )

        batch_size, time_steps, num_nodes, coordinate_dim = spatial_coords.shape
        flat_coords = spatial_coords.reshape(batch_size * time_steps, num_nodes, coordinate_dim)
        spatial_embedding = self.spatial_encoder(flat_coords, graph_info)
        spatial_embedding = spatial_embedding.view(batch_size, time_steps, num_nodes, -1)
        fused = self.fusion(torch.cat([z, spatial_embedding], dim=-1))
        return self.fusion_norm(fused)


class XelaForceSpatialAttentionProbe(XelaForceLinearProbe):
    """Force probe with supervised full self-attention over spatial sensor tokens."""

    supports_spatial_coords = True

    def __init__(
        self,
        *args,
        spatial_attention_layers: int = 1,
        spatial_attention_heads: int = 12,
        spatial_distance_hidden_dim: int = 16,
        spatial_distance_bias: bool = True,
        spatial_directional_bias: bool = False,
        coordinate_dim: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if spatial_attention_layers <= 0:
            raise ValueError("spatial_attention_layers must be positive")

        embed_dim = self.layer_norm.normalized_shape[0]
        self.spatial_distance_bias = spatial_distance_bias
        self.spatial_directional_bias = spatial_directional_bias
        self.coordinate_dim = coordinate_dim
        self.spatial_attention_blocks = nn.ModuleList(
            [
                XelaDistanceBiasedAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=spatial_attention_heads,
                    distance_hidden_dim=spatial_distance_hidden_dim,
                    coordinate_dim=coordinate_dim,
                    init_std=self.init_std,
                    use_distance_bias=spatial_distance_bias,
                    use_directional_bias=spatial_directional_bias,
                )
                for _ in range(spatial_attention_layers)
            ]
        )

    def _prepare_tokens(self, z, spatial_coords=None, graph_info=None):
        if self.spatial_distance_bias and spatial_coords is None:
            raise ValueError("spatial_coords are required for distance-biased spatial attention")
        if spatial_coords is not None:
            if spatial_coords.shape[:-1] != z.shape[:-1]:
                raise ValueError(
                    "spatial_coords and signal tokens must have matching batch, time, and sensor dimensions; "
                    f"got {tuple(spatial_coords.shape)} and {tuple(z.shape)}"
                )
            if spatial_coords.shape[-1] != self.coordinate_dim:
                raise ValueError(
                    f"expected {self.coordinate_dim} coordinate channels, got {spatial_coords.shape[-1]}"
                )

        batch_size, time_steps, num_nodes, embed_dim = z.shape
        tokens = z.reshape(batch_size * time_steps, num_nodes, embed_dim)
        coords = None
        if spatial_coords is not None:
            coords = spatial_coords.reshape(batch_size * time_steps, num_nodes, self.coordinate_dim)
        for block in self.spatial_attention_blocks:
            tokens = block(tokens, coords)
        return tokens.reshape(batch_size, time_steps, num_nodes, embed_dim)

    # def forward(self, x):
    #     x = self.pooler(x)
    #     x = self.probe(x)
    #     x = einops.rearrange(x, "b n (t c) -> b n t c", c=3 if not self.only_normal_force else 1)
    #     if self.only_normal_force:
    #         x = F.sigmoid(x) if self.with_last_activations else x
    #     else:
    #         if self.with_last_activations:
    #             x[..., -1] = F.sigmoid(x[..., -1])
    #             x[..., 0:2] = F.tanh(x[..., 0:2])
    #     x = einops.rearrange(x, "b n t c -> b (n t) c")
    #     return x


class XelaForceSLModule(ForceSLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert isinstance(self.model_encoder, XelaTransformer), "Model encoder must be a XelaTransformer"
        self.sequence_length, self.time_chunk_size = (
            self.model_encoder.sequence_length,
            self.model_encoder.time_chunk_size,
        )
        self.train_pred, self.train_gt = [], []
        self.val_pred = []
        self.val_gt = []
        self.test_pred, self.test_gt = [], []
        self.target_mean, self.target_std, self.target_max = None, None, None
        self.only_normal_force = self.model_task.only_normal_force

    def _graph_to_device(self, graph_info, device, repeats: int = 1):
        if graph_info is None:
            return None
        graph_info = {key: value.to(device) if hasattr(value, "to") else value for key, value in graph_info.items()}
        if hasattr(graph_info.get("edge_count"), "dim") and graph_info["edge_count"].dim() > 1:
            if repeats > 1 and graph_info["edge_count"].shape[1] != repeats:
                raise ValueError(
                    f"Graph chunks ({graph_info['edge_count'].shape[1]}) do not match encoder chunks ({repeats})"
                )
            return {
                key: value.reshape(-1, *value.shape[2:]) if hasattr(value, "reshape") and value.dim() > 1 else value
                for key, value in graph_info.items()
            }
        if repeats > 1:
            graph_info = {
                key: value.repeat_interleave(repeats, dim=0) if hasattr(value, "repeat_interleave") else value
                for key, value in graph_info.items()
            }
        return graph_info

    def _forward_encoder(self, sensor_data, graph_info=None):
        sensor_data = self._select_encoder_input_channels(sensor_data)
        if graph_info is not None and getattr(self.model_encoder, "supports_graph_info", False):
            return self.model_encoder.forward_features(sensor_data, graph_info=graph_info)
        return self.model_encoder.forward_features(sensor_data)

    @staticmethod
    def _global_rmse(forces_gt: torch.Tensor, forces_pred: torch.Tensor):
        squared_error = (forces_gt - forces_pred).double().square()
        squared_error_sum = squared_error.sum(dim=0)
        sample_count = torch.tensor(
            forces_gt.shape[0],
            device=forces_gt.device,
            dtype=torch.float64,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(squared_error_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sample_count, op=torch.distributed.ReduceOp.SUM)

        rmse_axes = torch.sqrt(squared_error_sum / sample_count)
        rmse = torch.sqrt(squared_error_sum.sum() / (sample_count * squared_error_sum.numel()))
        return rmse, rmse_axes
    
    def on_fit_start(self, train_dataloader=None, val_dataloader=None, trainer_instance=None):
        self.init_stats(train_dataloader, trainer_instance.fabric.device)

    def init_stats(self, dataloader, device):
        train_dset = dataloader.dataset

        n=4
        while not (isinstance(train_dset, data.Dataset) and hasattr(train_dset, "target_mean")):
            train_dset = train_dset.dataset
            n-=1
            if n==0:
                raise ValueError("train_dset is not a data.Dataset")

        target_mean = torch.tensor(train_dset.target_mean).float().to(device)
        target_std = torch.tensor(train_dset.target_std).float().to(device)
        target_max = torch.tensor(train_dset.target_max).float().to(device)
        self.model_task.update_target_stats(target_mean, target_std, target_max)

    def forward(self, batch, batch_idx):
        sensor_data = batch["sensor"]
        chunked_time = sensor_data.shape[1] // self.sequence_length
        spatial_coords = None
        if getattr(self.model_task, "supports_spatial_coords", False):
            if sensor_data.shape[-1] < 6:
                raise ValueError(
                    "The spatial force probe requires sensor data with three signal and three XYZ channels"
                )
            spatial_coords = einops.rearrange(
                sensor_data[..., 3:6],
                "b (l q k) n c -> b l q k n c",
                l=chunked_time,
                k=self.time_chunk_size,
            ).mean(dim=3)
            spatial_coords = einops.rearrange(spatial_coords, "b l q n c -> b l (q n) c")

        graph_info = self._graph_to_device(batch.get("graph"), sensor_data.device, repeats=chunked_time)
        sensor_data = einops.rearrange(sensor_data, "b (l k) n c -> (b l) k n c", k=self.sequence_length)
        z = self._forward_encoder(sensor_data, graph_info=graph_info)["x_norm_patchtokens"]  # pyright: ignore[reportCallIssue]
        z = F.layer_norm(z, (z.shape[-1],))
        z = einops.rearrange(z, "(b l) n c -> b l n c", l=chunked_time)

        z = self._encoder_output_for_task(z)
        y_pred = self.model_task(z, spatial_coords=spatial_coords, graph_info=graph_info)
        return y_pred.squeeze(1)

    def step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        y_pred = self.forward(batch, batch_idx)
        y_gt = batch["force"][:, -1].unsqueeze(-1) if self.only_normal_force else batch["force"]
        # y_gt_normalized = (y_gt - self.target_mean) / self.target_std
        y_gt_normalized = y_gt / self.target_max

        # if self.only_normal_force:
        #     y_gt = y_gt[..., 2].unsqueeze(-1)
        #     y_gt_normalized = y_gt_normalized[..., 2].unsqueeze(-1)
        
        
        loss = F.smooth_l1_loss(y_pred, y_gt_normalized)
        # y_pred_detached = y_pred.detach() * self.target_std + self.target_mean
        y_pred_detached = y_pred.detach() * self.target_max
        rmse = torch.sqrt(F.mse_loss(y_pred_detached, y_gt, reduction="none")).mean(dim=0)

        return {
            "loss": loss,
            "batch_rmse": rmse,
            "y_pred": y_pred_detached,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None or self.target_max is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
            self.target_max = self.model_task.target_max
            if self.only_normal_force:
                self.target_max = self.target_max[-1]
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None or self.target_max is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
            self.target_max = self.model_task.target_max
            if self.only_normal_force:
                self.target_max = self.target_max[-1]
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None or self.target_max is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
            self.target_max = self.model_task.target_max
            if self.only_normal_force:
                self.target_max = self.target_max[-1]
        return self.step(batch, batch_idx)

    def on_train_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.train_pred.append(outputs["y_pred"])
        self.train_gt.append(batch["force"])
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["force"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_test_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.test_pred.append(outputs["y_pred"])
        self.test_gt.append(batch["force"])
        self.collect_test_identifiers(batch)

    def on_train_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="train")

    def on_validation_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="val")

    def on_epoch_end(self, trainer_instance=None, stage="train"):
        target_gt = None
        target_pred = None

        if stage == "train":
            target_gt = torch.cat(self.train_gt, dim=0)
            target_pred = torch.cat(self.train_pred, dim=0)
        elif stage == "val":
            target_gt = torch.cat(self.val_gt, dim=0)
            target_pred = torch.cat(self.val_pred, dim=0)

        forces_gt = target_gt
        forces_pred = target_pred

        if self.only_normal_force:
            forces_pred = forces_pred.repeat(1, 3)
            forces_pred[:, 0:2] = 0.0
            
        rmse, rmse_axes = self._global_rmse(forces_gt, forces_pred)
        rmse_x, rmse_y, rmse_z = rmse_axes

        if self.only_normal_force:
            rmse_x = 1000.0
            rmse_y = 1000.0

        # im_corr = plot_correlation(forces_gt, forces_pred)
        # img_err, img_cone = plot_forces_error(forces_gt, forces_pred)

        step = trainer_instance.global_step if stage=="train" else trainer_instance.global_val_step
        epoch = trainer_instance.current_epoch

        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            for i, (rmse_val, axis) in enumerate(zip([rmse, rmse_x, rmse_y, rmse_z], ["", "_x", "_y", "_z"])):
                trainer_instance.writer.add_scalar(f"{stage}/rmse{axis}", rmse_val, epoch)
         
            # trainer_instance.wandb.log(
            #     {
            #         f"{stage}/correlation": trainer_instance.wandb.Image(im_corr),
            #         f"{stage}/error": trainer_instance.wandb.Image(img_err),
            #         f"{stage}/error_cone": trainer_instance.wandb.Image(img_cone),
            #     }
            # )
        
        if stage == "train":
            self.train_pred = []
            self.train_gt = []
        elif stage == "val":
            self.val_pred = []
            self.val_gt = []
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def on_test_end(self, trainer_instance=None, stage="test"):

        forces_gt = torch.cat(self.test_gt, dim=0)
        forces_pred = torch.cat(self.test_pred, dim=0)

        if self.only_normal_force:
            forces_pred = forces_pred.repeat(1, 3)
            forces_pred[:, 0:2] = 0.0
            
        rmse, rmse_axes = self._global_rmse(forces_gt, forces_pred)
        rmse_x, rmse_y, rmse_z = rmse_axes

        if self.only_normal_force:
            rmse_x = 1000.0
            rmse_y = 1000.0

        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            for i, (rmse_val, axis) in enumerate(zip([rmse, rmse_x, rmse_y, rmse_z], ["", "_x", "_y", "_z"])):
                trainer_instance.writer.add_scalar(f"{stage}/rmse{axis}", rmse_val, 0)
        self.save_test_artifact(
            trainer_instance,
            task="force",
            y_true=forces_gt,
            y_pred=forces_pred,
        )

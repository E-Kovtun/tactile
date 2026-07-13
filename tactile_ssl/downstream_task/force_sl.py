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

from typing import Any, Dict, Optional, List
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
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "jepa",
        encoder_normalization_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
            encoder_normalization_override=encoder_normalization_override,
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
    ):
        super().__init__()
        self.only_normal_force = only_normal_force
        self.n_outputs = n_outputs if not only_normal_force else 1
        self.init_std = init_std
        self.pad_id = pad_id

        if self.pad_id is not None:
            self.pad_range = get_pad_xela_indexes(pad_id)

        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.pooler = AttentivePooler(
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

    def forward(self, z):
        if self.pad_id is not None:
            z = z[:, :, self.pad_range[0]:self.pad_range[1], :]

        b, t, _, c = z.shape
        z = self.pooler(z.flatten(0, 1))
        z = z.view(b, t, c)
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
        graph_info = self._graph_to_device(batch.get("graph"), sensor_data.device, repeats=chunked_time)
        sensor_data = einops.rearrange(sensor_data, "b (l k) n c -> (b l) k n c", k=self.sequence_length)
        z = self._forward_encoder(sensor_data, graph_info=graph_info)["x_norm_patchtokens"]  # pyright: ignore[reportCallIssue]
        z = F.layer_norm(z, (z.shape[-1],))
        z = einops.rearrange(z, "(b l) n c -> b l n c", l=chunked_time)

        if self.train_encoder:
            y_pred = self.model_task(z)
        else:
            y_pred = self.model_task(z.detach())
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

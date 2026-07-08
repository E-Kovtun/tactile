import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
import einops
import wandb
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed
from tactile_ssl.model.signal_transformer import SignalTransformer
from tactile_ssl.model import VIT_EMBED_DIMS


log = get_pylogger(__name__)


class XelaJoystickProbe(nn.Module):
    def __init__(
        self,
        discretize: Optional[DictConfig] = None,
        input_dim: int = 3,
        embed_dim="base",
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.n_outputs = 3

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
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    drop_path=0.1,
                )
                for _ in range(depth)
            ]
        )
        self.init_std = init_std
        self.discretize = discretize
        if discretize is not None:
            self.num_bins = discretize.num_bins
            self.probe = nn.Linear(embed_dim, self.num_bins**3)
        else:
            self.probe = nn.Linear(embed_dim, self.n_outputs)

        self.pos_embed_fn = SinusoidalEmbed(100, 1, embed_dim)
        attn_bias = torch.ones(1, 1, 100, 100)
        attn_bias = attn_bias.tril()
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        self.register_buffer("attn_bias", attn_bias)
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

    def forward(self, z):
        b, t, _, c = z.shape
        z = self.pooler(z.flatten(0, 1))
        z = z.view(b, t, c)

        pos_embed = self.pos_embed_fn(z.device).float().unsqueeze(0)
        z = z + pos_embed

        for block in self.blocks:
            z = block(z, self.attn_bias)
        y = self.probe(z)
        return y


class XelaJoystickSLModule(SLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model_encoder, SignalTransformer), "Model encoder must be a SignalTransformer"
        self.sequence_length, self.time_chunk_size = (
            self.model_encoder.sequence_length,
            self.model_encoder.time_chunk_size,
        )
        self.train_pred, self.train_gt = [], []
        self.val_pred = []
        self.val_gt = []
        self.target_mean, self.target_std = None, None
    
    def on_fit_start(self, train_dataloader=None, val_dataloader=None, trainer_instance=None):
        if trainer_instance is not None:
            trainer_instance.wandb.define_metric("train/loss", summary="min")
            trainer_instance.wandb.define_metric("train/rmse", summary="min")
            trainer_instance.wandb.define_metric("train/rmse_x", summary="min")
            trainer_instance.wandb.define_metric("train/rmse_y", summary="min")
            trainer_instance.wandb.define_metric("train/rmse_z", summary="min")
            trainer_instance.wandb.define_metric("val/loss", summary="min")
            trainer_instance.wandb.define_metric("val/rmse", summary="min")
            trainer_instance.wandb.define_metric("val/rmse_x", summary="min")
            trainer_instance.wandb.define_metric("val/rmse_y", summary="min")
            trainer_instance.wandb.define_metric("val/rmse_z", summary="min")

            trainer_instance.wandb.define_metric("train/auc_x", summary="max")
            trainer_instance.wandb.define_metric("train/auc_y", summary="max")
            trainer_instance.wandb.define_metric("train/auc_z", summary="max")
            trainer_instance.wandb.define_metric("val/auc_x", summary="max")
            trainer_instance.wandb.define_metric("val/auc_y", summary="max")
            trainer_instance.wandb.define_metric("val/auc_z", summary="max")


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        cond = batch["sensor"]
        y_gt = batch["joystick_dir"]

        if self.target_mean is None or self.target_std is None:
            self.target_mean = batch["mean"][0]
            self.target_std = batch["std"][0]

        out = {}
        chunked_time = cond.shape[1] // self.sequence_length
        cond = einops.rearrange(cond, "b (l k) n c -> (b l) k n c", k=self.sequence_length)
        z = self.model_encoder.forward_features(cond)["x_norm_patchtokens"]  # pyright: ignore[reportCallIssue]
        z = F.layer_norm(z, (z.shape[-1],))
        z = einops.rearrange(z, "(b l) n c -> b l n c", l=chunked_time)
        y_pred = self.forward(z)
        if self.model_task.discretize is not None:
            loss = F.cross_entropy(y_pred.flatten(0, 1), y_gt.flatten(0, 1))
            y_pred = y_pred.argmax(dim=-1)
            accuracy = (y_pred == y_gt).float().mean(dim=(0, 1))
            out["batch_accuracy"] = accuracy
        else:
            y_pred = y_pred.squeeze()
            loss = F.mse_loss(y_pred, y_gt)
            y_pred_detached = y_pred.detach() * self.target_std + self.target_mean
            y_gt_detached = y_gt * self.target_std + self.target_mean
            rmse = torch.sqrt(F.mse_loss(y_pred_detached, y_gt_detached, reduction="none")).mean(dim=(0, 1))
            out["batch_rmse"] = rmse
        out["loss"] = loss
        out["y_pred"] = y_pred
        return out

    def forward(self, cond):
        if self.train_encoder:
            y_pred = self.model_task(cond)
        else:
            y_pred = self.model_task(cond.detach())
        return y_pred

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.wandb.log(
                {
                    f"{label}/loss": outputs["loss"],
                    f"global_{label}_step": step,
                }
            )
            if self.model_task.discretize is not None:
                metric = "batch_accuracy"
                trainer_instance.wandb.log(
                    {
                        f"{label}/{metric}": outputs[f"{metric}"].item(),
                        f"global_{label}_step": step,
                    }
                )
            else:
                metric = "batch_rmse"
                trainer_instance.wandb.log(
                    {
                        f"{label}/{metric}_x": outputs[f"{metric}"][0].item(),
                        f"global_{label}_step": step,
                    }
                )
                trainer_instance.wandb.log(
                    {
                        f"{label}/{metric}_y": outputs[f"{metric}"][1].item(),
                        f"global_{label}_step": step,
                    }
                )
                trainer_instance.wandb.log(
                    {
                        f"{label}/{metric}_z": outputs[f"{metric}"][-1].item(),
                        f"global_{label}_step": step,
                    }
                )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.train_pred.append(outputs["y_pred"])
        self.train_gt.append(batch["joystick_dir"])
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["joystick_dir"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_epoch_end(self, trainer_instance=None, stage="train"):
        joystick_gt = None
        joystick_pred = None
        if stage == "train":
            joystick_gt = torch.cat(self.train_gt, dim=0).cpu().numpy()
            joystick_pred = torch.cat(self.train_pred, dim=0).cpu().numpy()
        elif stage == "val":
            joystick_gt = torch.cat(self.val_gt, dim=0).cpu().numpy()
            joystick_pred = torch.cat(self.val_pred, dim=0).cpu().numpy()

        if self.model_task.discretize is not None:
            lower_bound = float(self.model_task.discretize.lower_bound)
            upper_bound = float(self.model_task.discretize.upper_bound)

            def flat_idx_to_xyz(idx, grid_size, voxel_size):
                x = idx // (grid_size**2)
                y = (idx - x * (grid_size**2)) // grid_size
                z = idx - x * (grid_size**2) - y * grid_size

                x = x * voxel_size * (upper_bound - lower_bound) + lower_bound
                y = y * voxel_size * (upper_bound - lower_bound) + lower_bound
                z = z * voxel_size * (upper_bound - lower_bound) + lower_bound
                return np.stack([x, y, z], axis=-1)

            voxel_size = (upper_bound - lower_bound) / float(self.model_task.num_bins)
            joystick_dir_gt = flat_idx_to_xyz(joystick_gt, self.model_task.num_bins, voxel_size)
            joystick_dir_pred = flat_idx_to_xyz(joystick_pred, self.model_task.num_bins, voxel_size)
            normalized_rmse = None
        else:
            target_mean = self.target_mean.cpu().numpy()
            target_std = self.target_std.cpu().numpy()
            joystick_dir_gt = joystick_gt * target_std + target_mean
            joystick_dir_pred = joystick_pred * target_std + target_mean
            normalized_rmse = np.sqrt(np.mean((joystick_gt - joystick_pred) ** 2))

        gt_mean = np.mean(joystick_dir_gt, axis=(0, 1))
        pred_mean = np.mean(joystick_dir_pred, axis=(0, 1))
        aligned_rmse = np.sqrt(
            np.mean(((joystick_dir_gt - gt_mean[None, None, :]) - (joystick_dir_pred - pred_mean[None, None, :])) ** 2)
        )
        rmse = np.sqrt(np.mean((joystick_dir_gt - joystick_dir_pred) ** 2))
        rmse_x = np.sqrt(np.mean((joystick_dir_gt[:, :, 0] - joystick_dir_pred[:, :, 0]) ** 2))
        rmse_y = np.sqrt(np.mean((joystick_dir_gt[:, :, 1] - joystick_dir_pred[:, :, 1]) ** 2))
        rmse_z = np.sqrt(np.mean((joystick_dir_gt[:, :, 2] - joystick_dir_pred[:, :, 2]) ** 2))

        norm_xyz_threshold = 0.1 # normalized theta threshold
        auc_x = np.mean(np.abs(joystick_dir_gt[..., 0] - joystick_dir_pred[..., 0]) < norm_xyz_threshold)
        auc_y = np.mean(np.abs(joystick_dir_gt[..., 1] - joystick_dir_pred[..., 1]) < norm_xyz_threshold)
        auc_z = np.mean(np.abs(joystick_dir_gt[..., 2] - joystick_dir_pred[..., 2]) < norm_xyz_threshold)  # 1 degree

        idxs = np.arange(0, len(joystick_dir_gt), len(joystick_dir_gt) // 10)
        joystick_dir_gt = joystick_dir_gt[idxs]
        joystick_dir_pred = joystick_dir_pred[idxs]
        figs = []
        for i in range(10):
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            curr_joy_gt = joystick_dir_gt[i]
            curr_joy_pred = joystick_dir_pred[i]
            time = np.arange(curr_joy_gt.shape[0])
            axs[0].plot(time, curr_joy_gt[:, 0], color="r", label="x", linestyle="--")
            axs[1].plot(time, curr_joy_gt[:, 1], color="g", label="y", linestyle="--")
            axs[2].plot(time, curr_joy_gt[:, 2], color="b", label="z", linestyle="--")

            axs[0].plot(time, curr_joy_pred[:, 0], color="r", label="x_pred")
            axs[1].plot(time, curr_joy_pred[:, 1], color="g", label="y_pred")
            axs[2].plot(time, curr_joy_pred[:, 2], color="b", label="z_pred")
            for ax in axs:
                # ax.set_ylim(-1, 1)
                ax.legend()
            figs.append(fig)

        step = trainer_instance.global_step if stage=="train" else trainer_instance.global_val_step

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    f"{stage}/outputs": [wandb.Image(fig) for fig in figs],
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{stage}/rmse": rmse,
                    f"global_{stage}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {f"{stage}/mean_aligned_rmse": aligned_rmse, f"global_{stage}_step": step}
            )
            if normalized_rmse is not None:
                trainer_instance.wandb.log(
                    {
                        f"{stage}/normalized_rmse": normalized_rmse,
                        f"global_{stage}_step": step,
                    }
                )
            for i, (rmse_val, axis) in enumerate(zip([rmse_x, rmse_y, rmse_z], ["x", "y", "z"])):
                trainer_instance.wandb.log(
                    {
                        f"{stage}/rmse_{axis}": rmse_val,
                        f"global_{stage}_step": step,
                    }
                )
            for i, (auc_val, axis) in enumerate(zip([auc_x, auc_y, auc_z], ["_x", "_y", "_z"])):
                trainer_instance.wandb.log(
                    {
                        f"{stage}/auc{axis}": auc_val,
                        f"global_{stage}_step": step,
                    }
                )

        for fig in figs:
            plt.close(fig)
        if stage == "train":
            self.train_pred = []
            self.train_gt = []
        elif stage == "val":
            self.val_pred = []
            self.val_gt = []
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def on_validation_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="val")

    def on_train_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="train")

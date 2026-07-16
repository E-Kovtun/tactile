import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import einops
import wandb
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule, gather_batch_tensor
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed
from tactile_ssl.model.signal_transformer import SignalTransformer
from tactile_ssl.model import VIT_EMBED_DIMS


log = get_pylogger(__name__)


class XelaRelativePoseDecoder(nn.Module):
    def __init__(
        self,
        discretize: Optional[DictConfig] = None,
        embed_dim="base",
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        n_outputs=3,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.init_std = init_std

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
        self.layer_norm = norm_layer(embed_dim)

        self.discretize = discretize
        if discretize is not None:
            assert isinstance(discretize, int), "Discretize must be an integer"
            assert discretize > 1, "Discretize must be greater than 1"
            self.num_bins = int(discretize)
            self.probe_x = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, self.num_bins),
            )
            self.probe_y = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, self.num_bins),
            )
            self.probe_z = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, self.num_bins),
            )
        else:
            self.probe = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, self.n_outputs),
            )

        self.pos_embed_fn = SinusoidalEmbed(10000, 1, embed_dim)

        attn_bias = torch.ones(1, 1, 1000, 1000)
        attn_bias = attn_bias.tril()
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        self.register_buffer("attn_bias", attn_bias)

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

    def update_target_stats(self, target_mean, target_std):
        print(f"Updating target stats in RelativePoseDecoder: {target_mean}, {target_std}")
        self.target_mean = target_mean
        self.target_std = target_std

    def _prepare_tokens(self, z, spatial_coords=None):
        return z

    def forward(self, z, spatial_coords=None):
        z = self._prepare_tokens(z, spatial_coords)
        b, t, _, c = z.shape
        z = self.pooler(z.flatten(0, 1))
        z = z.view(b, t, c)

        pos_embed = self.pos_embed_fn(z.device).float().unsqueeze(0)
        z += pos_embed[:, : z.shape[1]]

        for block in self.blocks:
            z = block(z, self.attn_bias[..., : z.shape[1], : z.shape[1]])
        z = self.layer_norm(z)

        if self.discretize is not None:
            y_tx = self.probe_x(z)
            y_ty = self.probe_y(z)
            y_tz = self.probe_z(z)
            y = torch.stack([y_tx, y_ty, y_tz], dim=-2)
        else:
            y = self.probe(z)
        return y


class XelaRelativePoseSpatialMLPDecoder(XelaRelativePoseDecoder):
    """Relative-pose decoder with supervised per-sensor XYZ embeddings."""

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

    def _prepare_tokens(self, z, spatial_coords=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaRelativePoseSpatialMLPDecoder")
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


class XelaRelativePoseModule(SLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model_encoder, SignalTransformer), "Model encoder must be a SignalTransformer"
        self.sequence_length, self.time_chunk_size = (
            self.model_encoder.sequence_length,
            self.model_encoder.time_chunk_size,
        )
        self.train_pred, self.train_gt = [], []
        self.val_pred, self.val_gt = [], []
        self.test_pred, self.test_gt = [], []
        self.target_mean, self.target_std = None, None

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
        self.model_task.update_target_stats(target_mean, target_std)

    def forward(self, batch, batch_idx):
        sensor_data = batch["sensor"]
        chunked_time = sensor_data.shape[1] // self.sequence_length
        spatial_coords = None
        if getattr(self.model_task, "supports_spatial_coords", False):
            if sensor_data.shape[-1] < 6:
                raise ValueError(
                    "The spatial relative-pose decoder requires sensor data with three signal and three XYZ channels"
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

        if self.train_encoder:
            y_pred = self.model_task(z, spatial_coords=spatial_coords)
        else:
            y_pred = self.model_task(z.detach(), spatial_coords=spatial_coords)
        return y_pred

    def step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        y_pred = self.forward(batch, batch_idx)
        y_gt = batch["relative_object_pose"]

        out = {}
        if self.model_task.discretize is not None:
            loss = 0
            for i in range(3):
                loss_i = F.cross_entropy(y_pred[..., i, :].flatten(0, 1), y_gt[..., i].flatten(0, 1))
                loss += loss_i
            y_pred = y_pred.argmax(dim=-1)
            accuracy = (y_pred == y_gt).float().mean(dim=(0, 1))
            out["batch_accuracy"] = accuracy
        else:
            y_gt_normalized = (y_gt - self.target_mean) / self.target_std
            loss = F.mse_loss(y_pred, y_gt_normalized)

            y_pred_detached = y_pred.detach() * self.target_std + self.target_mean
            rmse = torch.sqrt(F.mse_loss(y_pred_detached, y_gt, reduction="none")).mean(dim=(0, 1))
            out["batch_rmse"] = rmse

        out["loss"] = loss
        out["y_pred"] = y_pred_detached
        return out

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        if self.target_mean is None or self.target_std is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std
        return self.step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if (
            trainer_instance is not None
            and trainer_instance.fabric.is_global_zero
            and trainer_instance.should_log
        ):
            trainer_instance.writer.add_scalar(f"{label}/loss", outputs["loss"], step)
            metric = "batch_rmse"
            trainer_instance.writer.add_scalar(f"{label}/{metric}_x", outputs[f"{metric}"][0].item(), step)
            trainer_instance.writer.add_scalar(f"{label}/{metric}_y", outputs[f"{metric}"][1].item(), step)
            trainer_instance.writer.add_scalar(f"{label}/{metric}_theta", outputs[f"{metric}"][2].item(), step)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.train_pred.append(outputs["y_pred"])
        self.train_gt.append(batch["relative_object_pose"])
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["relative_object_pose"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_test_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.test_pred.append(outputs["y_pred"])
        self.test_gt.append(batch["relative_object_pose"])

    def on_train_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="train")

    def on_validation_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="val")

    def on_epoch_end(self, trainer_instance=None, stage="train"):
        target_gt = None
        target_pred = None

        target_mean = self.target_mean.cpu().numpy()
        target_std = self.target_std.cpu().numpy()

        if stage == "train":
            target_gt = torch.cat(self.train_gt, dim=0)
            target_pred = torch.cat(self.train_pred, dim=0)
        elif stage == "val":
            target_gt = torch.cat(self.val_gt, dim=0)
            target_pred = torch.cat(self.val_pred, dim=0)

        target_gt = gather_batch_tensor(target_gt).cpu().numpy()
        target_pred = gather_batch_tensor(target_pred).cpu().numpy()

        if self.model_task.discretize is not None:
            lower_bound = target_mean - 2 * target_std
            upper_bound = target_mean + 2 * target_std

            def idx_to_val(idx):
                return lower_bound + (upper_bound - lower_bound) * (idx / self.model_task.discretize)

            relative_pose_gt = idx_to_val(target_gt)
            relative_pose_pred = idx_to_val(target_pred)
        else:
            relative_pose_gt = target_gt
            relative_pose_pred = target_pred

        rmse = np.sqrt(np.mean((relative_pose_gt - relative_pose_pred) ** 2))
        rmse_x = np.sqrt(np.mean((relative_pose_gt[:, :, 0] - relative_pose_pred[:, :, 0]) ** 2))
        rmse_y = np.sqrt(np.mean((relative_pose_gt[:, :, 1] - relative_pose_pred[:, :, 1]) ** 2))
        rmse_theta = np.sqrt(np.mean((relative_pose_gt[:, :, 2] - relative_pose_pred[:, :, 2]) ** 2))

        xy_threshold = 0.02  # 1mm
        theta_threshold = 5.0
        auc_x_1mm = np.mean(np.abs(relative_pose_gt[..., 0] - relative_pose_pred[..., 0]) < xy_threshold)
        auc_y_1mm = np.mean(np.abs(relative_pose_gt[..., 1] - relative_pose_pred[..., 1]) < xy_threshold)
        auc_theta_1deg = np.mean(np.abs(relative_pose_gt[..., 2] - relative_pose_pred[..., 2]) < theta_threshold) 

        # idxs = np.arange(0, len(relative_pose_gt), len(relative_pose_gt) // 10)
        # relative_pose_gt = relative_pose_gt[idxs]
        # relative_pose_pred = relative_pose_pred[idxs]
        # figs = []
        # for i in range(10):
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        #     curr_relative_pose_gt = relative_pose_gt[i]
        #     curr_relative_pose_pred = relative_pose_pred[i]
        #     time = np.arange(curr_relative_pose_gt.shape[0])
        #     axs[0].plot(time, curr_relative_pose_gt[:, 0], color="r", label="x", linestyle="--")
        #     axs[1].plot(time, curr_relative_pose_gt[:, 1], color="g", label="y", linestyle="--")
        #     axs[2].plot(
        #         time,
        #         curr_relative_pose_gt[:, 2],
        #         color="b",
        #         label=r"$\theta$",
        #         linestyle="--",
        #     )

        #     axs[0].plot(time, curr_relative_pose_pred[:, 0], color="r", label="x_pred")
        #     axs[1].plot(time, curr_relative_pose_pred[:, 1], color="g", label="y_pred")
        #     axs[2].plot(time, curr_relative_pose_pred[:, 2], color="b", label=r"$\theta$_pred")
        #     for ax in axs:
        #         ax.legend()
        #     figs.append(fig)

        step = trainer_instance.global_step if stage=="train" else trainer_instance.global_val_step
        epoch = trainer_instance.current_epoch

        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            # trainer_instance.wandb.log(
            #     {
            #         f"{stage}/outputs": [wandb.Image(fig) for fig in figs],
            #     }
            # )
            for i, (rmse_val, axis) in enumerate(zip([rmse, rmse_x, rmse_y, rmse_theta], ["", "_x", "_y", "_theta"])):
                trainer_instance.writer.add_scalar(f"{stage}/rmse{axis}", rmse_val, epoch)

            for i, (auc_val, axis) in enumerate(zip([auc_x_1mm, auc_y_1mm, auc_theta_1deg], ["_x", "_y", "_theta"])):
                trainer_instance.writer.add_scalar(f"{stage}/acc{axis}", auc_val, epoch)
        # for fig in figs:
        #     plt.close(fig)
        if stage == "train":
            self.train_pred = []
            self.train_gt = []
        elif stage == "val":
            self.val_pred = []
            self.val_gt = []
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def on_test_end(self, trainer_instance=None, stage="test"):

        relative_pose_gt = gather_batch_tensor(torch.cat(self.test_gt, dim=0)).cpu().numpy()
        relative_pose_pred = gather_batch_tensor(torch.cat(self.test_pred, dim=0)).cpu().numpy()
            
        rmse = np.sqrt(np.mean((relative_pose_gt - relative_pose_pred) ** 2))
        rmse_x = np.sqrt(np.mean((relative_pose_gt[:, :, 0] - relative_pose_pred[:, :, 0]) ** 2))
        rmse_y = np.sqrt(np.mean((relative_pose_gt[:, :, 1] - relative_pose_pred[:, :, 1]) ** 2))
        rmse_theta = np.sqrt(np.mean((relative_pose_gt[:, :, 2] - relative_pose_pred[:, :, 2]) ** 2))

        xy_threshold = 0.02  # 1mm
        theta_threshold = 5.0
        auc_x_1mm = np.mean(np.abs(relative_pose_gt[..., 0] - relative_pose_pred[..., 0]) < xy_threshold)
        auc_y_1mm = np.mean(np.abs(relative_pose_gt[..., 1] - relative_pose_pred[..., 1]) < xy_threshold)
        auc_theta_1deg = np.mean(np.abs(relative_pose_gt[..., 2] - relative_pose_pred[..., 2]) < theta_threshold) 


        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            for i, (rmse_val, axis) in enumerate(zip([rmse, rmse_x, rmse_y, rmse_theta], ["", "_x", "_y", "_theta"])):
                trainer_instance.writer.add_scalar(f"{stage}/rmse{axis}", rmse_val, 0)

            for i, (auc_val, axis) in enumerate(zip([auc_x_1mm, auc_y_1mm, auc_theta_1deg], ["_x", "_y", "_theta"])):
                trainer_instance.writer.add_scalar(f"{stage}/acc{axis}", auc_val, 0)

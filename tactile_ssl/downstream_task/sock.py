from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.downstream_task.sl_module import SLModule, gather_batch_tensor
from tactile_ssl.downstream_task.xela_object import XelaObjectSLModule
from tactile_ssl.downstream_task.xela_relativepose import XelaRelativePoseModule


class SockTemporalClassifier(nn.Module):
    """Pool each five-frame JEPA embedding, then model the nine-step sequence."""

    expects_temporal_patch_tokens = True

    def __init__(
        self,
        input_embed_dim: int,
        classes: Sequence[str],
        class_weights: Optional[Sequence[float]] = None,
        num_heads: int = 3,
        temporal_depth: int = 2,
        dropout: float = 0.1,
        max_windows: int = 9,
    ) -> None:
        super().__init__()
        self.num_classes = len(classes)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.spatial_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            depth=1,
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, max_windows, input_embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=input_embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * input_embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, temporal_depth)
        self.norm = nn.LayerNorm(input_embed_dim)
        self.probe = nn.Linear(input_embed_dim, self.num_classes)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError("Expected [batch, windows, sensors, embedding]")
        batch_size, windows, sensors, channels = patch_tokens.shape
        if windows > self.temporal_pos.shape[1]:
            raise ValueError("More temporal windows than configured")
        pooled = self.spatial_pooler(
            patch_tokens.reshape(batch_size * windows, sensors, channels)
        ).squeeze(1)
        pooled = pooled.reshape(batch_size, windows, channels)
        pooled = pooled + self.temporal_pos[:, :windows]
        encoded = self.temporal_encoder(pooled)
        return self.probe(self.norm(encoded.mean(dim=1)))


class SockBaselineIdentity(nn.Identity):
    """Identity encoder carrying the attributes expected by SLModule wrappers."""

    sequence_length = 1
    time_chunk_size = 1


class SockCNNGRU(nn.Module):
    """Released SensTextile CNN+BiGRU action-classification baseline."""

    def __init__(
        self,
        classes: Sequence[str],
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = len(classes)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.conv1 = nn.Conv2d(2, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.gru = nn.GRU(
            32 * 4 * 4,
            120,
            num_layers=2,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(240, 84)
        self.fc2 = nn.Linear(84, self.num_classes)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 4:
            raise ValueError("Expected matching [batch, time, 32, 32] foot grids")
        batch, time, height, width = left.shape
        x = torch.stack([left, right], dim=2).reshape(batch * time, 2, height, width)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(batch, time, 32 * 4 * 4).transpose(0, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = torch.cat([x[-1, :, :120], x[0, :, 120:]], dim=1)
        return self.fc2(F.relu(self.fc1(x)))


class SockActionCNNGRUModule(XelaObjectSLModule):
    """Supervised released baseline evaluated on the shared action split."""

    def forward(self, batch, batch_idx):
        left = batch["left_grid"]
        right = batch["right_grid"]
        if self.training:
            # Exact distribution used by the released loader after z-scoring.
            left = left + (torch.randn_like(left) - 0.5) * 0.5
            right = right + (torch.randn_like(right) - 0.5) * 0.5
        return self.model_task(left, right)


class SockActionSLModule(XelaObjectSLModule):
    """Action probe over 45 raw frames represented by nine JEPA windows."""

    def forward(self, batch, batch_idx):
        sensor = batch["sensor"]
        if sensor.shape[1] % self.sequence_length:
            raise ValueError("Action window must be divisible by encoder sequence_length")
        windows = sensor.shape[1] // self.sequence_length
        sensor = einops.rearrange(
            sensor, "b (w t) n c -> (b w) t n c", t=self.sequence_length
        )
        encoded = self._forward_encoder(sensor)["x_norm_patchtokens"]
        encoded = F.layer_norm(encoded, (encoded.shape[-1],))
        encoded = einops.rearrange(encoded, "(b w) n c -> b w n c", w=windows)
        encoded = self._encoder_output_for_task(encoded)
        return self.model_task(encoded)

    def on_test_end(self, trainer_instance=None, stage="test"):
        target_gt = gather_batch_tensor(torch.cat(self.test_gt, dim=0)).cpu().numpy()
        target_pred = gather_batch_tensor(torch.cat(self.test_pred, dim=0)).cpu().numpy()
        recalls = []
        f1_scores = []
        for label in range(self.model_task.num_classes):
            true_positive = np.sum((target_gt == label) & (target_pred == label))
            false_positive = np.sum((target_gt != label) & (target_pred == label))
            false_negative = np.sum((target_gt == label) & (target_pred != label))
            recall_denominator = true_positive + false_negative
            precision_denominator = true_positive + false_positive
            recall = (
                true_positive / recall_denominator
                if recall_denominator
                else 0.0
            )
            precision = (
                true_positive / precision_denominator
                if precision_denominator
                else 0.0
            )
            recalls.append(recall)
            f1_scores.append(
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            trainer_instance.writer.add_scalar(
                f"{stage}/balanced_accuracy", float(np.mean(recalls)), 0
            )
            trainer_instance.writer.add_scalar(
                f"{stage}/macro_f1", float(np.mean(f1_scores)), 0
            )
        super().on_test_end(trainer_instance, stage)


class _SockPoseMetricsMixin:
    def _pose_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        y_pred = self.forward(batch, batch_idx)
        y_gt = batch["relative_object_pose"]
        y_gt_normalized = (y_gt - self.target_mean) / self.target_std
        loss = F.mse_loss(y_pred, y_gt_normalized)
        y_pred_real = y_pred.detach() * self.target_std + self.target_mean
        component_rmse = torch.sqrt(
            F.mse_loss(y_pred_real, y_gt, reduction="none").mean(dim=(0, 1))
        )
        return {
            "loss": loss,
            "y_pred": y_pred_real,
            "batch_rmse": component_rmse.mean(),
        }

    def training_step(self, batch, batch_idx):
        self._ensure_stats()
        return self._pose_step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self._ensure_stats()
        return self._pose_step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self._ensure_stats()
        return self._pose_step(batch, batch_idx)

    def _ensure_stats(self):
        if self.target_mean is None or self.target_std is None:
            self.target_mean = self.model_task.target_mean
            self.target_std = self.model_task.target_std

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if (
            trainer_instance is not None
            and trainer_instance.fabric.is_global_zero
            and trainer_instance.should_log
        ):
            trainer_instance.writer.add_scalar(f"{label}/loss", outputs["loss"], step)
            trainer_instance.writer.add_scalar(
                f"{label}/batch_rmse", outputs["batch_rmse"], step
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.train_pred.append(outputs["y_pred"])
        self.train_gt.append(batch["relative_object_pose"])
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["relative_object_pose"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_test_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.test_pred.append(outputs["y_pred"])
        self.test_gt.append(batch["relative_object_pose"])
        self.collect_test_identifiers(batch)

    def on_train_epoch_end(self, trainer_instance=None):
        self._pose_epoch_end(trainer_instance, "train")

    def on_validation_epoch_end(self, trainer_instance=None):
        self._pose_epoch_end(trainer_instance, "val")

    def _pose_epoch_end(self, trainer_instance, stage):
        pred_list = self.train_pred if stage == "train" else self.val_pred
        gt_list = self.train_gt if stage == "train" else self.val_gt
        prediction = gather_batch_tensor(torch.cat(pred_list)).cpu().numpy()
        target = gather_batch_tensor(torch.cat(gt_list)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        joint_rmse = np.sqrt(
            np.mean((prediction.reshape(-1, 23, 3) - target.reshape(-1, 23, 3)) ** 2,
                    axis=(0, 2))
        )
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            trainer_instance.writer.add_scalar(f"{stage}/rmse", rmse, trainer_instance.current_epoch)
            for joint, value in enumerate(joint_rmse):
                trainer_instance.writer.add_scalar(
                    f"{stage}/rmse_joint_{joint:02d}", float(value), trainer_instance.current_epoch
                )
        pred_list.clear()
        gt_list.clear()

    def on_test_end(self, trainer_instance=None, stage="test"):
        local_gt = torch.cat(self.test_gt)
        local_pred = torch.cat(self.test_pred)
        gt = gather_batch_tensor(local_gt).cpu().numpy()
        pred = gather_batch_tensor(local_pred).cpu().numpy()
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            trainer_instance.writer.add_scalar(f"{stage}/rmse", rmse, 0)
        self.save_test_artifact(
            trainer_instance, task="sock_pose", y_true=local_gt, y_pred=local_pred
        )


class SockPoseSLModule(_SockPoseMetricsMixin, XelaRelativePoseModule):
    """Pose at raw frame five from one five-frame JEPA embedding."""

    step = _SockPoseMetricsMixin._pose_step


class SockTemporalPoseRegressor(nn.Module):
    """Aggregate twelve frozen five-frame JEPA embeddings into one pose."""

    def __init__(
        self,
        input_embed_dim: int,
        n_outputs: int = 69,
        num_heads: int = 3,
        temporal_depth: int = 2,
        dropout: float = 0.1,
        max_windows: int = 12,
    ) -> None:
        super().__init__()
        self.spatial_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            depth=1,
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, max_windows, input_embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=input_embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * input_embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, temporal_depth)
        self.norm = nn.LayerNorm(input_embed_dim)
        self.probe = nn.Linear(input_embed_dim, n_outputs)
        self.register_buffer("target_mean", torch.zeros(n_outputs))
        self.register_buffer("target_std", torch.ones(n_outputs))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def update_target_stats(self, mean, std):
        self.target_mean = mean
        self.target_std = std

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError("Expected [batch, windows, sensors, embedding]")
        batch, windows, _, channels = patch_tokens.shape
        if windows > self.temporal_pos.shape[1]:
            raise ValueError("More temporal windows than configured")
        pooled = self.spatial_pooler(
            patch_tokens.reshape(-1, patch_tokens.shape[2], channels)
        ).squeeze(1)
        pooled = pooled.reshape(batch, windows, channels)
        pooled = pooled + self.temporal_pos[:, :windows]
        encoded = self.temporal_encoder(pooled)
        return self.probe(self.norm(encoded.mean(dim=1)))


class SockPoseOfficialJEPAModule(_SockPoseMetricsMixin, XelaRelativePoseModule):
    """JEPA pose probe on the baseline's centered 60-frame examples."""

    def forward(self, batch, batch_idx):
        sensor = batch["sensor"]
        if sensor.shape[1] % self.sequence_length:
            raise ValueError("Pose window must be divisible by encoder sequence_length")
        windows = sensor.shape[1] // self.sequence_length
        sensor = einops.rearrange(
            sensor, "b (w t) n c -> (b w) t n c", t=self.sequence_length
        )
        encoded = self._forward_encoder_patchtokens(sensor)
        encoded = F.layer_norm(encoded, (encoded.shape[-1],))
        encoded = einops.rearrange(encoded, "(b w) n c -> b w n c", w=windows)
        encoded = self._encoder_output_for_task(encoded)
        return self.model_task(encoded).unsqueeze(1)

    step = _SockPoseMetricsMixin._pose_step


class SockPoseCNN(nn.Module):
    """Corrected implementation of the official two-foot CNN baseline."""

    def __init__(self, window_size: int = 60, n_outputs: int = 69) -> None:
        super().__init__()

        def make_branch():
            return nn.Sequential(
                nn.Conv2d(window_size, 16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Conv2d(16, 8, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Flatten(),
            )

        # The released train.py requests symmetric=False. Its right-channel
        # declaration is inconsistent with the 60-frame loader; use the
        # intended asymmetric branches, both with the actual 60 channels.
        self.left_branch = make_branch()
        self.right_branch = make_branch()
        self.probe = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs),
        )
        self.register_buffer("target_mean", torch.zeros(n_outputs))
        self.register_buffer("target_std", torch.ones(n_outputs))

    def update_target_stats(self, mean, std):
        self.target_mean = mean
        self.target_std = std

    def forward(self, left, right):
        return self.probe(
            torch.cat([self.left_branch(left), self.right_branch(right)], dim=-1)
        )


class SockPoseCNNModule(_SockPoseMetricsMixin, SLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_pred, self.train_gt = [], []
        self.val_pred, self.val_gt = [], []
        self.test_pred, self.test_gt = [], []
        self.target_mean, self.target_std = None, None

    def forward(self, batch, batch_idx):
        prediction = self.model_task(batch["left_grid"], batch["right_grid"])
        return prediction.unsqueeze(1)

    def _pose_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        y_pred = self.forward(batch, batch_idx)
        y_gt = batch["relative_object_pose"]
        loss = F.mse_loss(y_pred, y_gt)
        component_rmse = torch.sqrt(
            F.mse_loss(y_pred.detach(), y_gt, reduction="none").mean(dim=(0, 1))
        )
        return {
            "loss": loss,
            "y_pred": y_pred.detach(),
            "batch_rmse": component_rmse.mean(),
        }

    step = _pose_step

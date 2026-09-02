from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.algorithm.module import Module


class DecoImageTokenizer(nn.Module):
    """ResNet-34 spatial image tokens, matching the public DECO backbone."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise ImportError("DecoImageTokenizer requires torchvision") from error
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, embed_dim, kernel_size=3, padding=1)
        self.camera_embedding = nn.Embedding(2, embed_dim)

    def project_backbone_features(self, backbone_features: torch.Tensor) -> torch.Tensor:
        if backbone_features.ndim != 5 or backbone_features.shape[1:3] != (2, 512):
            raise ValueError(
                "Expected cached image features [B,2,512,H,W], "
                f"got {tuple(backbone_features.shape)}"
            )
        batch, cameras = backbone_features.shape[:2]
        features = self.projection(backbone_features.flatten(0, 1))
        features = features.flatten(2).transpose(1, 2)
        features = features.reshape(batch, cameras, features.shape[1], features.shape[2])
        camera_ids = torch.arange(cameras, device=backbone_features.device)
        features = features + self.camera_embedding(camera_ids)[None, :, None]
        return features.flatten(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError(f"Expected images [B,2,3,H,W], got {tuple(images.shape)}")
        batch, cameras = images.shape[:2]
        flat_features = self.backbone(images.flatten(0, 1))
        backbone_features = flat_features.reshape(batch, cameras, *flat_features.shape[1:])
        return self.project_backbone_features(backbone_features)


class DecoFusionBlock(nn.Module):
    """DECO-style joint visual/action attention plus tactile cross-attention."""

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.joint_norm = nn.LayerNorm(dim)
        self.joint_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.tactile_query_norm = nn.LayerNorm(dim)
        self.tactile_norm = nn.LayerNorm(dim)
        self.tactile_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self,
        visual: torch.Tensor,
        action: torch.Tensor,
        tactile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_length = visual.shape[1]
        joint = torch.cat([visual, action], dim=1)
        normalized = self.joint_norm(joint)
        attended, _ = self.joint_attention(normalized, normalized, normalized, need_weights=False)
        joint = joint + attended
        tactile_update, _ = self.tactile_attention(
            self.tactile_query_norm(joint),
            self.tactile_norm(tactile),
            self.tactile_norm(tactile),
            need_weights=False,
        )
        joint = joint + tactile_update
        joint = joint + self.mlp(self.output_norm(joint))
        return joint[:, :visual_length], joint[:, visual_length:]


class DecoPolicyHead(nn.Module):
    """Fuse two cameras, proprioception and JEPA tactile tokens into 32x28 actions."""

    def __init__(
        self,
        tactile_embed_dim: int = 192,
        dim: int = 512,
        action_dim: int = 28,
        chunk_size: int = 32,
        num_conditions: int = 6,
        num_heads: int = 8,
        depth: int = 6,
        inference_steps: int = 10,
        image_pretrained: bool = False,
        freeze_image_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.inference_steps = int(inference_steps)
        self.image_tokenizer = DecoImageTokenizer(dim, image_pretrained)
        self.freeze_image_backbone = bool(freeze_image_backbone)
        if self.freeze_image_backbone:
            self.image_tokenizer.backbone.requires_grad_(False)
            self.image_tokenizer.backbone.eval()
        self.tactile_projection = nn.Sequential(
            nn.LayerNorm(tactile_embed_dim), nn.Linear(tactile_embed_dim, dim)
        )
        self.proprio_encoder = nn.Sequential(nn.Linear(action_dim, dim), nn.Mish(), nn.Linear(dim, dim))
        self.condition_encoder = nn.Embedding(num_conditions, dim)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, dim), nn.Mish(), nn.Linear(dim, dim))
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.blocks = nn.ModuleList(
            [DecoFusionBlock(dim=dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.action_output = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen ResNet must keep BatchNorm running statistics fixed as well
        # as parameters; Module.train() would otherwise put it back in train mode.
        if self.freeze_image_backbone:
            self.image_tokenizer.backbone.eval()
        return self

    def predict_velocity(
        self,
        tactile_tokens: torch.Tensor,
        images: Optional[torch.Tensor],
        proprio: torch.Tensor,
        condition: torch.Tensor,
        noisy_action: torch.Tensor,
        diffusion_time: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tactile_tokens.ndim != 3:
            raise ValueError(f"Expected tactile tokens [B,N,C], got {tuple(tactile_tokens.shape)}")
        if image_features is not None:
            visual = self.image_tokenizer.project_backbone_features(image_features)
        elif images is not None:
            visual = self.image_tokenizer(images)
        else:
            raise ValueError("Either images or cached image_features is required")
        tactile = self.tactile_projection(tactile_tokens)
        conditioning = (
            self.proprio_encoder(proprio)
            + self.condition_encoder(condition)
            + self.time_encoder(diffusion_time)
        )
        visual = visual + conditioning[:, None]
        action = (
            self.action_encoder(noisy_action)
            + self.action_queries.expand(tactile.shape[0], -1, -1)
            + conditioning[:, None]
        )
        for block in self.blocks:
            visual, action = block(visual, action, tactile)
        return self.action_output(self.output_norm(action))

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        images: Optional[torch.Tensor],
        proprio: torch.Tensor,
        condition: torch.Tensor,
        noisy_action: Optional[torch.Tensor] = None,
        diffusion_time: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noisy_action is not None:
            if diffusion_time is None:
                raise ValueError("diffusion_time is required with noisy_action")
            return self.predict_velocity(
                tactile_tokens, images, proprio, condition, noisy_action, diffusion_time,
                image_features=image_features,
            )
        sample = torch.randn(
            tactile_tokens.shape[0], self.chunk_size, self.action_dim,
            device=tactile_tokens.device, dtype=tactile_tokens.dtype,
        )
        schedule = torch.linspace(1.0, 0.0, self.inference_steps + 1, device=sample.device)
        for current, following in zip(schedule[:-1], schedule[1:]):
            time = current.expand(sample.shape[0])
            velocity = self.predict_velocity(
                tactile_tokens, images, proprio, condition, sample, time,
                image_features=image_features,
            )
            sample = sample + (following - current) * velocity
        return sample


class SparshStyleImageTokenizer(nn.Module):
    """One global image token per camera, following Sparsh-skin B.4."""

    def __init__(
        self,
        embed_dim: int = 192,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import (
                ResNet18_Weights,
                ResNet34_Weights,
                resnet18,
                resnet34,
            )
        except ImportError as error:
            raise ImportError("SparshStyleImageTokenizer requires torchvision") from error
        builders = {
            "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
            "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        }
        if backbone_name not in builders:
            raise ValueError(f"Unsupported image backbone {backbone_name!r}")
        builder, pretrained_weights = builders[backbone_name]
        backbone = builder(weights=pretrained_weights if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.projection = nn.Linear(512, embed_dim)
        self.camera_embedding = nn.Embedding(2, embed_dim)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def project_backbone_features(self, backbone_features: torch.Tensor) -> torch.Tensor:
        if backbone_features.ndim != 5 or backbone_features.shape[1:3] != (2, 512):
            raise ValueError(
                "Expected cached image features [B,2,512,H,W], "
                f"got {tuple(backbone_features.shape)}"
            )
        pooled = backbone_features.mean(dim=(-2, -1))
        pooled = pooled.to(dtype=self.projection.weight.dtype)
        tokens = self.projection(pooled)
        camera_ids = torch.arange(2, device=tokens.device)
        return tokens + self.camera_embedding(camera_ids)[None]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError(f"Expected images [B,2,3,H,W], got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
            images = (images - self.image_mean) / self.image_std
        batch = images.shape[0]
        features = self.backbone(images.flatten(0, 1))
        features = features.reshape(batch, 2, *features.shape[1:])
        return self.project_backbone_features(features)


class SparshStyleTactilePooler(nn.Module):
    """Pool all tactile patch tokens into one learned full-hand token."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 3) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tactile_tokens: torch.Tensor) -> torch.Tensor:
        if tactile_tokens.ndim != 3:
            raise ValueError(f"Expected tactile tokens [B,N,C], got {tuple(tactile_tokens.shape)}")
        normalized = self.input_norm(tactile_tokens)
        query = self.query.expand(tactile_tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, normalized, normalized, need_weights=False)
        return self.output_norm(query + pooled)


class DecoSparshPolicyHead(nn.Module):
    """Small Sparsh-style action-token decoder adapted to DECO observations."""

    def __init__(
        self,
        tactile_embed_dim: int = 192,
        dim: int = 192,
        action_dim: int = 28,
        chunk_size: int = 16,
        num_heads: int = 3,
        depth: int = 2,
        dropout: float = 0.1,
        image_backbone: str = "resnet18",
        image_pretrained: bool = False,
        freeze_image_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.image_tokenizer = SparshStyleImageTokenizer(
            embed_dim=dim,
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            freeze_backbone=freeze_image_backbone,
        )
        self.tactile_pooler = SparshStyleTactilePooler(tactile_embed_dim, num_heads)
        self.tactile_projection = (
            nn.Identity()
            if tactile_embed_dim == dim
            else nn.Linear(tactile_embed_dim, dim)
        )
        self.tactile_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.action_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.chunk_size * self.action_dim),
        )
        nn.init.trunc_normal_(self.tactile_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.action_token, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.image_tokenizer.freeze_backbone:
            self.image_tokenizer.backbone.eval()
        return self

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        images: Optional[torch.Tensor],
        proprio: torch.Tensor,
        condition: torch.Tensor,
        noisy_action: Optional[torch.Tensor] = None,
        diffusion_time: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del proprio, condition, diffusion_time
        if noisy_action is not None:
            raise ValueError("The Sparsh-style policy uses direct behavior cloning, not diffusion")
        if image_features is not None:
            visual = self.image_tokenizer.project_backbone_features(image_features)
        elif images is not None:
            visual = self.image_tokenizer(images)
        else:
            raise ValueError("Either images or cached image_features is required")
        tactile = self.tactile_projection(self.tactile_pooler(tactile_tokens))
        tactile = tactile + self.tactile_type_embedding
        action = self.action_token.expand(tactile.shape[0], -1, -1)
        fused = self.transformer(torch.cat([action, tactile, visual], dim=1))
        prediction = self.action_mlp(self.output_norm(fused[:, 0]))
        return prediction.reshape(-1, self.chunk_size, self.action_dim)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-torch.log(torch.tensor(10000.0, device=time.device, dtype=time.dtype)) / max(half - 1, 1))
        )
        angles = time[:, None] * frequency[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class DecoPolicyModel(nn.Module):
    def __init__(
        self,
        tactile_encoder: Optional[nn.Module],
        policy_head: nn.Module,
        train_encoder: bool = False,
        checkpoint_encoder: Optional[str] = None,
        encoder_checkpoint_key: str = "target_encoder",
        tactile_input_mode: str = "encoder",
        random_tactile_token_count: int = 528,
        random_tactile_embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if tactile_input_mode not in {"encoder", "random"}:
            raise ValueError("tactile_input_mode must be 'encoder' or 'random'")
        if tactile_input_mode == "encoder" and tactile_encoder is None:
            raise ValueError("tactile_encoder is required in encoder mode")
        if tactile_input_mode == "random" and checkpoint_encoder:
            raise ValueError("Random tactile mode must not load an encoder checkpoint")
        if random_tactile_token_count <= 0 or random_tactile_embed_dim <= 0:
            raise ValueError("Random tactile token dimensions must be positive")
        self.tactile_encoder = tactile_encoder
        self.policy_head = policy_head
        self.train_encoder = bool(train_encoder)
        self.tactile_input_mode = str(tactile_input_mode)
        self.random_tactile_token_count = int(random_tactile_token_count)
        self.random_tactile_embed_dim = int(random_tactile_embed_dim)
        if checkpoint_encoder:
            checkpoint = torch.load(checkpoint_encoder, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            prefix = f"{encoder_checkpoint_key}."
            encoder_state = {
                key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
            }
            if not encoder_state:
                raise KeyError(
                    f"No {encoder_checkpoint_key!r} weights found in {checkpoint_encoder}"
                )
            self.tactile_encoder.load_state_dict(encoder_state, strict=True)
        if self.tactile_encoder is not None and not self.train_encoder:
            self.tactile_encoder.requires_grad_(False)
            self.tactile_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tactile_encoder is not None and not self.train_encoder:
            self.tactile_encoder.eval()
        return self

    def forward(
        self,
        sensor,
        images,
        proprio,
        condition,
        noisy_action=None,
        diffusion_time=None,
        image_features=None,
    ):
        if self.tactile_input_mode == "random":
            # Fresh nuisance tokens preserve the fusion interface while carrying
            # no information from the tactile observation. The global training
            # seed still makes the ablation reproducible per policy-head run.
            tactile_tokens = torch.randn(
                sensor.shape[0],
                self.random_tactile_token_count,
                self.random_tactile_embed_dim,
                device=sensor.device,
                dtype=sensor.dtype,
            )
        elif self.train_encoder:
            tactile_tokens = self.tactile_encoder(sensor)
        else:
            with torch.no_grad():
                tactile_tokens = self.tactile_encoder(sensor)
        return self.policy_head(
            tactile_tokens, images, proprio, condition, noisy_action, diffusion_time,
            image_features=image_features,
        )


def masked_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    per_value = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    weights = valid_mask.to(dtype=per_value.dtype)
    return (per_value * weights).sum() / weights.sum().clamp_min(1)


class DecoPolicyModule(Module, nn.Module):
    """Minimal Trainer-compatible behavior-cloning module for Task 4."""

    def __init__(
        self,
        model: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.optim_partial = optim_cfg
        self.scheduler_partial = scheduler_cfg
        self.test_prediction: list[torch.Tensor] = []
        self.test_target: list[torch.Tensor] = []
        self.test_valid_mask: list[torch.Tensor] = []
        self.test_sample_id: list[torch.Tensor] = []
        self.test_group_id: list[torch.Tensor] = []
        self.test_condition: list[torch.Tensor] = []

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_action: Optional[torch.Tensor] = None,
        diffusion_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            sensor=batch["sensor"],
            images=batch.get("images"),
            proprio=batch["proprio"],
            condition=batch["condition"],
            noisy_action=noisy_action,
            diffusion_time=diffusion_time,
            image_features=batch.get("image_features"),
        )

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action = batch["action"]
        time = torch.sigmoid(torch.randn(action.shape[0], device=action.device, dtype=action.dtype))
        noise = torch.randn_like(action)
        noisy_action = (1.0 - time[:, None, None]) * action + time[:, None, None] * noise
        target_velocity = noise - action
        prediction = self.forward(batch, noisy_action, time)
        loss = masked_action_loss(prediction, target_velocity, batch["action_valid_mask"])
        mse = ((prediction.detach() - target_velocity) ** 2).mean(dim=-1)
        weights = batch["action_valid_mask"].to(mse.dtype)
        rmse = torch.sqrt((mse * weights).sum() / weights.sum().clamp_min(1))
        return {"loss": loss, "batch_rmse": rmse, "y_pred": prediction.detach()}

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = masked_action_loss(prediction, batch["action"], batch["action_valid_mask"])
        error = (prediction - batch["action"]).square().mean(dim=-1)
        weights = batch["action_valid_mask"].to(error.dtype)
        rmse = torch.sqrt((error * weights).sum() / weights.sum().clamp_min(1))
        return {"loss": loss, "batch_rmse": rmse, "y_pred": prediction}

    def on_test_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        # Policy evaluation is intentionally single-GPU. Keeping the accumulated
        # arrays on CPU avoids retaining a full Task-4 test split in GPU memory.
        self.test_prediction.append(outputs["y_pred"].detach().cpu())
        self.test_target.append(batch["action"].detach().cpu())
        self.test_valid_mask.append(batch["action_valid_mask"].detach().cpu())
        self.test_sample_id.append(batch["sample_id"].detach().cpu())
        self.test_group_id.append(batch["group_id"].detach().cpu())
        self.test_condition.append(batch["condition"].detach().cpu())

    def on_test_end(self, trainer_instance=None):
        if trainer_instance is None or not trainer_instance.fabric.is_global_zero:
            return
        if not self.test_prediction:
            raise RuntimeError("DECO policy evaluation produced no test batches")
        run_root = Path(trainer_instance.checkpoint_dir).resolve().parent
        output = run_root / "evaluation" / "test_predictions.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        # NumPy has no native bfloat16 dtype. Mixed-precision policy heads can
        # legitimately accumulate BF16 predictions, so serialize metrics and
        # artifacts in portable FP32 instead.
        prediction = torch.cat(self.test_prediction).float().numpy()
        target = torch.cat(self.test_target).float().numpy()
        valid_mask = torch.cat(self.test_valid_mask).numpy()
        per_step_mse = np.square(prediction - target).mean(axis=-1)
        rmse = float(np.sqrt(per_step_mse[valid_mask].mean()))
        np.savez_compressed(
            output,
            task=np.asarray("deco_task4_policy"),
            y_true=target,
            y_pred=prediction,
            action_valid_mask=valid_mask,
            sample_id=torch.cat(self.test_sample_id).numpy(),
            group_id=torch.cat(self.test_group_id).numpy(),
            condition=torch.cat(self.test_condition).numpy(),
            normalized_action_rmse=np.asarray(rmse, dtype=np.float64),
            checkpoint=np.asarray(getattr(trainer_instance, "evaluation_checkpoint_path", "")),
        )
        trainer_instance.writer.add_scalar("test/normalized_action_rmse", rmse, 0)

    def load_task(self, checkpoint_task: str) -> None:
        checkpoint = torch.load(checkpoint_task, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        self.load_state_dict(state, strict=True)

    def configure_optimizers(
        self, num_iterations_per_epoch: int, num_epochs: int
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        optimizer = self.optim_partial(p for p in self.parameters() if p.requires_grad)
        if self.scheduler_partial is None:
            return optimizer, None, None
        scheduler = self.scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        return optimizer, {"scheduler": scheduler, "interval": "step", "monitor": None}, None

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero and trainer_instance.should_log:
            trainer_instance.writer.add_scalar("train/loss", outputs["loss"], trainer_instance.global_step)
            trainer_instance.writer.add_scalar("train/batch_rmse", outputs["batch_rmse"], trainer_instance.global_step)

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero and trainer_instance.should_log:
            trainer_instance.writer.add_scalar("val/loss", outputs["loss"], trainer_instance.global_val_step)
            trainer_instance.writer.add_scalar("val/batch_rmse", outputs["batch_rmse"], trainer_instance.global_val_step)


class DecoSparshPolicyModule(DecoPolicyModule):
    """Direct behavior-cloning task module for the Sparsh-style DECO decoder."""

    @staticmethod
    def _aligned_target(batch: Dict[str, torch.Tensor], prediction: torch.Tensor):
        target = batch["action"]
        valid_mask = batch["action_valid_mask"]
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
            )
        return target, valid_mask

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prediction = self.forward(batch)
        target, valid_mask = self._aligned_target(batch, prediction)
        loss = masked_action_loss(prediction, target, valid_mask)
        error = (prediction.detach() - target).square().mean(dim=-1)
        weights = valid_mask.to(error.dtype)
        rmse = torch.sqrt((error * weights).sum() / weights.sum().clamp_min(1))
        return {"loss": loss, "batch_rmse": rmse, "y_pred": prediction.detach()}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def on_test_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        target, valid_mask = self._aligned_target(batch, outputs["y_pred"])
        self.test_prediction.append(outputs["y_pred"].detach().cpu())
        self.test_target.append(target.detach().cpu())
        self.test_valid_mask.append(valid_mask.detach().cpu())
        self.test_sample_id.append(batch["sample_id"].detach().cpu())
        self.test_group_id.append(batch["group_id"].detach().cpu())
        self.test_condition.append(batch["condition"].detach().cpu())

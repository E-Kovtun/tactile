from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from tactile_ssl.downstream_task.deco_policy import (
    DecoSparshPolicyHead,
    DecoSparshPolicyModule,
    SparshStyleImageTokenizer,
)


class FrozenResNet18ImageTokenizer(SparshStyleImageTokenizer):
    """Frozen standard ResNet18 with a trainable 512-to-policy projection."""

    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__(
            embed_dim=embed_dim,
            backbone_name="resnet18",
            pretrained=True,
            freeze_backbone=True,
        )

    def project_backbone_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 5 and features.shape[1:3] == (2, 512):
            pooled = features.mean(dim=(-2, -1))
        elif features.ndim == 3 and features.shape[1:] == (2, 512):
            pooled = features
        else:
            raise ValueError(
                "Expected cached ResNet18 features [B,2,512] or [B,2,512,H,W], "
                f"got {tuple(features.shape)}"
            )
        pooled = pooled.to(dtype=self.projection.weight.dtype)
        camera_ids = torch.arange(2, device=pooled.device)
        return self.projection(pooled) + self.camera_embedding(camera_ids)[None]


class FinalDecoSparshPolicyHead(DecoSparshPolicyHead):
    """Common action-token head for tactile and controlled vision-only runs."""

    def __init__(
        self,
        tactile_embed_dim: int = 192,
        dim: int = 192,
        action_dim: int = 12,
        chunk_size: int = 16,
        num_heads: int = 3,
        depth: int = 2,
        dropout: float = 0.1,
        use_tactile: bool = True,
    ) -> None:
        super().__init__(
            tactile_embed_dim=tactile_embed_dim,
            dim=dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            image_backbone="resnet18",
            image_pretrained=True,
            freeze_image_backbone=True,
        )
        self.image_tokenizer = FrozenResNet18ImageTokenizer(dim)
        self.use_tactile = bool(use_tactile)
        if not self.use_tactile:
            self.tactile_pooler = None
            self.tactile_projection = None
            self.tactile_type_embedding = None

    def forward(
        self,
        tactile_tokens: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        proprio: torch.Tensor,
        condition: torch.Tensor,
        noisy_action: Optional[torch.Tensor] = None,
        diffusion_time: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del proprio, condition, diffusion_time
        if noisy_action is not None:
            raise ValueError("The final DECO policy uses direct behavior cloning")
        if image_features is not None:
            visual = self.image_tokenizer.project_backbone_features(image_features)
        elif images is not None:
            visual = self.image_tokenizer(images)
        else:
            raise ValueError("Either images or cached image_features is required")
        tokens = [self.action_token.expand(visual.shape[0], -1, -1)]
        if self.use_tactile:
            if tactile_tokens is None:
                raise ValueError("Tactile tokens are required for this policy variant")
            tactile = self.tactile_projection(self.tactile_pooler(tactile_tokens))
            tokens.append(tactile + self.tactile_type_embedding)
        elif tactile_tokens is not None:
            raise ValueError("Vision-only policy must not receive tactile tokens")
        tokens.append(visual)
        fused = self.transformer(torch.cat(tokens, dim=1))
        prediction = self.action_mlp(self.output_norm(fused[:, 0]))
        return prediction.reshape(-1, self.chunk_size, self.action_dim)


class FinalDecoPolicyModel(nn.Module):
    """Frozen, absent, or end-to-end direct-4T tactile encoder wrapper."""

    def __init__(
        self,
        tactile_encoder: Optional[nn.Module],
        policy_head: FinalDecoSparshPolicyHead,
        train_encoder: bool = False,
        checkpoint_encoder: Optional[str] = None,
        encoder_checkpoint_key: str = "target_encoder",
        tactile_input_mode: str = "encoder",
    ) -> None:
        super().__init__()
        if tactile_input_mode not in {"encoder", "none"}:
            raise ValueError("tactile_input_mode must be 'encoder' or 'none'")
        if tactile_input_mode == "encoder" and tactile_encoder is None:
            raise ValueError("Encoder mode requires a tactile encoder")
        if tactile_input_mode == "none" and (tactile_encoder is not None or checkpoint_encoder):
            raise ValueError("Vision-only mode must not construct or load a tactile encoder")
        if policy_head.use_tactile != (tactile_input_mode == "encoder"):
            raise ValueError("Policy-head tactile interface does not match tactile_input_mode")
        self.tactile_encoder = tactile_encoder
        self.policy_head = policy_head
        self.train_encoder = bool(train_encoder)
        self.tactile_input_mode = tactile_input_mode
        if checkpoint_encoder:
            checkpoint = torch.load(checkpoint_encoder, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            prefix = f"{encoder_checkpoint_key}."
            encoder_state = {
                key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
            }
            if not encoder_state:
                raise KeyError(f"No {encoder_checkpoint_key!r} weights found in {checkpoint_encoder}")
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
        if self.tactile_input_mode == "none":
            tactile_tokens = None
        elif self.train_encoder:
            tactile_tokens = self.tactile_encoder(sensor)
        else:
            with torch.no_grad():
                tactile_tokens = self.tactile_encoder(sensor)
        return self.policy_head(
            tactile_tokens,
            images,
            proprio,
            condition,
            noisy_action,
            diffusion_time,
            image_features=image_features,
        )


class FinalDecoSparshPolicyModule(DecoSparshPolicyModule):
    """Behavior-cloning module with an explicit hands-only evaluation artifact."""

    def on_test_end(self, trainer_instance=None):
        if trainer_instance is None or not trainer_instance.fabric.is_global_zero:
            return
        if not self.test_prediction:
            raise RuntimeError("DECO hands-only policy evaluation produced no test batches")
        run_root = Path(trainer_instance.checkpoint_dir).resolve().parent
        output = run_root / "evaluation" / "test_predictions.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction = torch.cat(self.test_prediction).float().numpy()
        target = torch.cat(self.test_target).float().numpy()
        if prediction.shape[-1] != 12 or target.shape[-1] != 12:
            raise ValueError("Hands-only evaluation requires 12 action dimensions")
        valid_mask = torch.cat(self.test_valid_mask).numpy()
        per_step_mse = np.square(prediction - target).mean(axis=-1)
        rmse = float(np.sqrt(per_step_mse[valid_mask].mean()))
        np.savez_compressed(
            output,
            task=np.asarray("deco_task4_hands_policy"),
            action_schema=np.asarray(
                [*(f"left_hand_qpos_{i}" for i in range(6)),
                 *(f"right_hand_qpos_{i}" for i in range(6))]
            ),
            source_action_indices=np.asarray(
                [*range(7, 13), *range(20, 26)], dtype=np.int64
            ),
            y_true=target,
            y_pred=prediction,
            action_valid_mask=valid_mask,
            sample_id=torch.cat(self.test_sample_id).numpy(),
            group_id=torch.cat(self.test_group_id).numpy(),
            condition=torch.cat(self.test_condition).numpy(),
            normalized_action_rmse=np.asarray(rmse, dtype=np.float64),
            checkpoint=np.asarray(
                getattr(trainer_instance, "evaluation_checkpoint_path", "")
            ),
        )
        trainer_instance.writer.add_scalar("test/normalized_hand_action_rmse", rmse, 0)

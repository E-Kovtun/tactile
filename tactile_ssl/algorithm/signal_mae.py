from __future__ import annotations

from typing import Any, Dict, Literal

import einops
import torch
import torch.nn.functional as F

from tactile_ssl.algorithm.mae import MAEModule


class SignalMAEModule(MAEModule):
    """Canonical visible-token MAE for tactile sensor and hypertaxel tokens."""

    def __init__(self, *args, **kwargs) -> None:
        # Signal encoders tokenize sensors through either patch_embed (Xela,
        # Socks) or a dedicated hypertaxel tokenizer (DECO).
        kwargs["require_patch_embed"] = False
        super().__init__(*args, **kwargs)
        if self.encoder.sequence_length != self.encoder.time_chunk_size:
            raise ValueError(
                "SignalMAEModule currently requires one temporal chunk; "
                "sequence_length must equal time_chunk_size"
            )
        if self.mask_type != "random":
            raise ValueError("SignalMAEModule currently supports random masking only")

    def sample_masks(self, sensor: torch.Tensor):
        batch, _, num_sensors, _ = sensor.shape
        keep_count = max(1, int(num_sensors * (1.0 - self.mask_ratio)))
        noise = torch.rand(batch, num_sensors, device=sensor.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]
        mask = torch.ones(batch, num_sensors, device=sensor.device, dtype=torch.bool)
        mask[:, :keep_count] = False
        mask = torch.gather(mask, 1, ids_restore)
        return ids_keep, mask, ids_restore

    def forward_encoder(self, sensor: torch.Tensor, ids_keep: torch.Tensor):
        return self.encoder.forward_features(
            sensor,
            masks=[ids_keep],
            mask_type="tubelet",
        )

    def forward(self, sensor: torch.Tensor):
        ids_keep, mask, ids_restore = self.sample_masks(sensor)
        encoded = self.forward_encoder(sensor, ids_keep)
        prediction = self.decoder(encoded["x_norm_patchtokens"], ids_restore)
        return prediction, mask

    def reconstruction_target(self, sensor: torch.Tensor) -> torch.Tensor:
        normalize = getattr(self.encoder, "normalize", None)
        target = normalize(sensor) if callable(normalize) else sensor
        return einops.rearrange(
            target,
            "b (t k) n c -> b (t n) (c k)",
            k=self.encoder.time_chunk_size,
        )

    def element_validity(self, target: torch.Tensor) -> torch.Tensor:
        tokenizer = getattr(self.encoder, "tokenizer", None)
        member_mask = getattr(tokenizer, "member_mask", None)
        if member_mask is None:
            return torch.ones_like(target, dtype=torch.bool)
        validity = einops.repeat(
            member_mask,
            "n c -> 1 n (c k)",
            k=self.encoder.time_chunk_size,
        )
        return validity.to(device=target.device).expand(target.shape[0], -1, -1)

    def compute_loss(self, sensor, prediction, mask):
        target = self.reconstruction_target(sensor)
        validity = self.element_validity(target)
        squared_error = F.mse_loss(prediction, target, reduction="none")
        valid_count = validity.sum(dim=-1).clamp_min(1)
        token_loss = (squared_error * validity).sum(dim=-1) / valid_count
        return (token_loss * mask).sum() / mask.sum().clamp_min(1)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        sensor = batch["sensor"]
        prediction, mask = self.forward(sensor)
        loss = self.compute_loss(sensor, prediction, mask)
        return {
            "loss": loss,
            "ssl_loss": loss.detach().item(),
            "reconstruction": prediction.detach(),
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self.training_step(batch, batch_idx)

    def _log(self, outputs, stage: Literal["train", "val"], trainer_instance) -> None:
        if trainer_instance is None or not trainer_instance.should_log:
            return
        step = trainer_instance.global_step if stage == "train" else trainer_instance.global_val_step
        trainer_instance.writer.add_scalar(f"{stage}/loss", outputs["loss"], step)
        trainer_instance.writer.add_scalar(f"{stage}/ssl_loss", outputs["ssl_loss"], step)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self._log(outputs, "train", trainer_instance)

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self._log(outputs, "val", trainer_instance)

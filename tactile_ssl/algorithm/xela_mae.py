import random
from typing import Any, Dict, List, Optional, Literal

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from xformers.ops import fmha

from tactile_ssl.algorithm import MAEModule
from tactile_ssl.data.xela.utils import xela_sensor_layout
from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.utils.masking import sample_block_mask, sample_block_size_1d

log = get_pylogger(__name__)


class XelaMAEModule(MAEModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: Load this in a different way
        # This is valid only when the baseline is subtracted in the xela dataset

    def log_on_batch_end(self, outputs, stage: Literal["train", "val"] = "train", trainer_instance=None):
        loss = outputs["loss"]
        ssl_loss = outputs["ssl_loss"]
        if trainer_instance is not None and trainer_instance.should_log:
            step = trainer_instance.step
            trainer_instance.writer.add_scalar(f"{stage}/loss", loss, step)
            trainer_instance.writer.add_scalar(f"{stage}/ssl_loss", ssl_loss, step)

            for probe in self.online_probes:
                outputs_probe = {k: v for k, v in outputs.items() if k.startswith(probe.probe_name)}
                for k, v in outputs_probe.items():
                    trainer_instance.writer.add_scalar(f"{stage}/{k}", v, step)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)
        # Plot online probe predictions
        if trainer_instance is not None:
            step = trainer_instance.global_val_step
            if step is None:
                return
            if (step % self.log_freq_img == 0) and "reconstruction_img" in outputs.keys():
                X_pred = outputs["reconstruction_img"]
                encoder = self.encoder
                X_pred = einops.rearrange(
                    X_pred,
                    "b (t n) (c k) ->b (t k) n c",
                    k=encoder.time_chunk_size,
                    n=encoder.in_dim,
                )
                # in_dims corresponds to num sensors
                X_pred = einops.rearrange(X_pred, "b t n (c l) -> b t n c l", n=encoder.in_dim, l=encoder.in_chans)
                X_orig = batch["sensor"]
                X_orig = einops.rearrange(X_orig, "b t n (c l) -> b t n c l", n=encoder.in_dim, l=encoder.in_chans)
                X_pred = X_pred[0].cpu().numpy()[..., 0, :3]
                X_orig = X_orig[0].cpu().numpy()[..., 0, :3]
                # print(f"x_pred: {X_pred}")
                # print(f"x_orig: {X_orig}")
                xela_mean = self.encoder.xela_mean.detach().cpu().numpy()
                xela_std = self.encoder.xela_std.detach().cpu().numpy()
                X_pred = xela_sensor_layout(X_pred, xela_mean, xela_std)
                X_orig = xela_sensor_layout(X_orig)

                trainer_instance.writer.add_video("val/pred_signal", torch.from_numpy(X_pred).unsqueeze(0), step, fps=5)
                trainer_instance.writer.add_video("val/target_signal", torch.from_numpy(X_orig).unsqueeze(0), step, fps=5)

    def sample_masks(self, x):
        batch_size, _, num_sensors, _ = x.shape

        len_keep = int(num_sensors * (1 - self.mask_ratio))

        noise = torch.rand((batch_size, num_sensors), device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones((batch_size, num_sensors), device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        ids_keep = ids_keep.unsqueeze(1)
        mask = mask.unsqueeze(1)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, ids_mask_visible: Optional[torch.Tensor] = None):
        embedding = self.encoder.forward_features(x, masktoken_masks=ids_mask_visible)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
    ):
        ids_mask_visible, mask, ids_restore_mask = self.sample_masks(x)
        embedding_dict = self.forward_encoder(x, mask)
        patch_tokens = embedding_dict["x_norm_patchtokens"]

        x_pred = self.decoder(patch_tokens, ids_restore_mask)
        return x_pred, mask

    def compute_loss(self, x, x_pred, mask):
        target = self.encoder.normalize(x)
        target = einops.rearrange(target, "b (t k) n c -> b (t n) (c k)", k=self.encoder.time_chunk_size)
        loss = F.mse_loss(x_pred, target, reduction="none")
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["sensor"]
        x_pred, mask = self.forward(x)

        loss = self.compute_loss(x, x_pred, mask.squeeze(1))

        output = {
            "ssl_loss": loss.item(),
            "reconstruction_img": x_pred.detach(),
        }

        # online probes
        embedding = None
        cls_embedding = None
        if len(self.online_probes) > 0:
            with torch.no_grad():
                output_dict = self.encoder.forward_features(x)
                cls_embedding = output_dict["x_norm_regtokens"].squeeze(1)
                embedding = output_dict["x_norm_patchtokens"]
                embedding = F.layer_norm(embedding, (embedding.size(-1),))
                target = self.encoder.normalize(x)

        online_probes_loss = 0.0
        for probe in self.online_probes:
            probe_name: str = str(probe.probe_name)
            if probe_name == "reconstruction":
                target = einops.rearrange(target, "b (t k) n c -> b (t n) (c k)", k=self.encoder.time_chunk_size)
                probe_loss, decoded_x = probe(embedding, target=target)
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                output[f"{probe_name}_img"] = decoded_x.detach()
            elif "classification" in probe_name:
                gt_labels = batch[probe_name]
                probe_loss, pred_logits = probe(cls_embedding, target=gt_labels)
                pred_labels = torch.argmax(pred_logits, dim=1)
                accuracy = (pred_labels == gt_labels).float().mean()
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                output[f"{probe_name}_accuracy"] = accuracy
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss
        output["loss"] = loss  # type: ignore
        output["online_probes_loss"] = online_probes_loss

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

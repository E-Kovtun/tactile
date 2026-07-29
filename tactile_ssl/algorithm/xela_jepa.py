import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import copy 
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import random
from omegaconf import ListConfig

from tactile_ssl.algorithm.module import Module
from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.utils.ema import update_moving_average
from tactile_ssl.utils.masking import sample_block_mask, sample_block_size_1d
from tactile_ssl.utils.jepa_masking import sample_multiblock_graph_masks
from tactile_ssl.model.signal_transformer import SignalDecoder


log = get_pylogger(__name__)


class SignalJEPAPredictor(SignalDecoder):
    def __init__(self, zero_init_mask_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if zero_init_mask_tokens:
            nn.init.zeros_(self.mask_token)

    def prepare_tokens_with_mask(self, x, context_masks, masks, context_pos_embed):
        b, chunked_t, _, _ = x.shape

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(x.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        elif self.pos_embed_fn == "given":
            pos_embed = context_pos_embed
        else:
            raise NotImplementedError("Unknown position embedding function")
        pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", t=chunked_t)
        pos_embed = einops.repeat(pos_embed, "1 t n c -> b t n c", b=b)

        # context_masked_pos_embed = self.apply_tubelet_masks(pos_embed, context_masks)
        # x = x + context_masked_pos_embed

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

    def forward(self, x, context_masks, masks, context_pos_embed, *args, **kwargs):
        assert context_masks is not None, "JEPA Predictor requires context masks"
        assert masks is not None, "JEPA Predictor requires masks"

        x = self.pre_embed(x, *args, **kwargs)
        _, t, n, _ = x.shape
        num_context_tokens = t * n
        x = self.prepare_tokens_with_mask(x, context_masks, masks, context_pos_embed)
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


class XelaJEPAModule(Module, nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        context_mask_scale: Tuple[float, float] = (0.85, 1.0),
        target_mask_scale: Tuple[float, float] = (0.15, 0.2),
        num_context_masks: int = 1,
        num_target_masks: int = 4,
        moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
        use_momentum: bool = True,
        masking: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum

        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = SignalJEPAPredictor(input_dim=encoder.embed_dim, in_dim=encoder.in_dim, in_chans=encoder.in_chans, 
            sequence_length=encoder.sequence_length, time_chunk_size=encoder.time_chunk_size, 
            num_heads=encoder.num_heads, embed_dim=encoder.embed_dim, with_masktoken=True, pos_embed_fn="given",
            zero_init_mask_tokens=True, depth=4)

        self.jepa_loss = nn.MSELoss()

        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(moving_average_decay, ListConfig)
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay

        self.context_mask_scale = context_mask_scale
        self.target_mask_scale = target_mask_scale
        self.num_context_masks = num_context_masks
        self.num_target_masks = num_target_masks
        self.masking = masking

        self.generator = torch.Generator()
        self.step = -1
        self._schedule_step = 0
        self._schedule_total_steps = None

    def log_on_batch_end(self, outputs, stage: Literal["train", "val"] = "train", trainer_instance=None):
        ssl_loss = outputs["ssl_loss"]
        if trainer_instance is not None and trainer_instance.should_log:
            step = trainer_instance.step
            trainer_instance.writer.add_scalar(f"{stage}/ssl_loss", ssl_loss, step)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        if self.use_momentum:
            moving_average_decay = (
                next(self.momentum_scheduler) if self.momentum_scheduler is not None else self.moving_average_decay
            )
            with torch.no_grad():
                update_moving_average(
                    self.target_encoder,
                    self.context_encoder,
                    moving_average_decay,
                )
        trainer_instance.writer.add_scalar("train/moving_average_decay", moving_average_decay, trainer_instance.step)
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)
        self._schedule_step += 1

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)

    def sample_jepa_masks(self, x, graph_info: Optional[Dict[str, torch.Tensor]] = None):
        batch_size, _, num_sensors, _ = x.shape

        context_maskblock_size = sample_block_size_1d(num_sensors, self.context_mask_scale)[0]
        target_maskblock_size = sample_block_size_1d(num_sensors, self.target_mask_scale)[0]

        if self.masking is not None and str(self.masking.get("mode", "legacy")) != "legacy":
            mode = str(self.masking.get("mode"))
            if mode != "multiblock_graph":
                raise ValueError(f"Unsupported JEPA masking mode {mode!r}")
            if graph_info is None:
                raise ValueError("multiblock_graph masking requires batch['graph']")

            context_cfg = self.masking.get("context", {})
            target_cfg = self.masking.get("target", {})
            overlap_cfg = self.masking.get("overlap", {})
            expected_overlap = {
                "target_target": "allow",
                "context_target": "subtract_from_context",
                "context_context": "allow",
            }
            actual_overlap = {
                key: str(overlap_cfg.get(key, value)) for key, value in expected_overlap.items()
            }
            if actual_overlap != expected_overlap:
                raise ValueError(
                    "multiblock_graph currently supports only I-JEPA overlap semantics: "
                    f"{expected_overlap}; got {actual_overlap}"
                )

            return sample_multiblock_graph_masks(
                graph_info=graph_info,
                batch_size=batch_size,
                num_nodes=num_sensors,
                context_size=context_maskblock_size,
                target_size=target_maskblock_size,
                num_context_masks=self.num_context_masks,
                num_target_masks=self.num_target_masks,
                context_strategy=str(context_cfg.get("strategy", "connected_region")),
                target_strategy=str(target_cfg.get("strategy", "connected_region")),
                context_growth=str(context_cfg.get("growth", "dijkstra")),
                target_growth=str(target_cfg.get("growth", "dijkstra")),
                target_groups=target_cfg.get("groups"),
                min_context_keep_tokens=int(self.masking.get("min_context_keep_tokens", 32)),
                min_context_keep_ratio=float(self.masking.get("min_context_keep_ratio", 0.15)),
                max_resample_attempts=int(self.masking.get("max_resample_attempts", 32)),
                device=x.device,
                generator=self.generator,
            )

        context_masks, target_masks = self.sample_context_target_masks(batch_size=batch_size, orig_shape=num_sensors, 
            context_mask_shape=context_maskblock_size, num_context_masks=self.num_context_masks, 
            target_mask_shape=target_maskblock_size, num_target_masks=self.num_target_masks, 
            device=x.device, generator=self.generator
        )
        return context_masks, target_masks

    def sample_jepa_target_mask(
        self,
        orig_shape: int,
        mask_shape: int,
        generator: Optional[torch.Generator] = None,
    ):

        mask_indices = torch.randperm(orig_shape, generator=generator)[:mask_shape]
        mask = torch.zeros(orig_shape, dtype=torch.int32)
        mask[mask_indices] = 1

        mask_complement = torch.ones(orig_shape, dtype=torch.int32)
        mask_complement[mask_indices] = 0

        return mask_indices, mask_complement


    def sample_jepa_context_mask(
        self, 
        orig_shape: int,
        mask_shape: int,
        acceptable_regions: List[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ):

        acceptable_mask = torch.ones(orig_shape, dtype=torch.int32)

        for k in range(len(acceptable_regions)):
            acceptable_mask *= acceptable_regions[k]

        acceptable_indices = torch.nonzero(acceptable_mask).flatten()

        perm = torch.randperm(len(acceptable_indices), generator=generator)
        mask_indices = acceptable_indices[perm[:mask_shape]]

        mask = torch.zeros(orig_shape, dtype=torch.int32)
        mask[mask_indices] = 1

        mask_complement = torch.ones(orig_shape, dtype=torch.int32)
        mask_complement[mask_indices] = 0

        return mask_indices, mask_complement


    def sample_context_target_masks(
        self, 
        batch_size: int,
        orig_shape: int, # num_sensors
        context_mask_shape: int,
        num_context_masks: int,
        target_mask_shape: int,
        num_target_masks: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        
        min_keep_context_patches, min_keep_target_patches = orig_shape, orig_shape
        collated_context_masks, collated_target_masks = [], []

        for _ in range(batch_size):
            target_masks_cached = []
            target_masks_complement = []
            for _ in range(num_target_masks):
                mask, mask_complement = self.sample_jepa_target_mask(
                    orig_shape=orig_shape, 
                    mask_shape=target_mask_shape, 
                    generator=generator
                )
                target_masks_cached.append(mask)
                target_masks_complement.append(mask_complement)
                min_keep_target_patches = min(min_keep_target_patches, len(mask))
            collated_target_masks.append(target_masks_cached)

            context_masks_cached = []
            for _ in range(num_context_masks):
                mask, _ = self.sample_jepa_context_mask(
                    orig_shape=orig_shape,
                    mask_shape=context_mask_shape,
                    acceptable_regions=target_masks_complement,
                    generator=generator,
                )
                context_masks_cached.append(mask)
                min_keep_context_patches = min(min_keep_context_patches, len(mask))
            collated_context_masks.append(context_masks_cached)

        collated_context_masks = [[cm[:min_keep_context_patches] for cm in masks] for masks in collated_context_masks]
        collated_target_masks = [[cm[:min_keep_target_patches] for cm in masks] for masks in collated_target_masks]

        context_masks = torch.stack([torch.stack(sample_masks, dim=0) for sample_masks in collated_context_masks], dim=0) # b x k x n1
        context_masks = context_masks.permute(1, 0, 2).to(device)
        target_masks = torch.stack([torch.stack(sample_masks, dim=0) for sample_masks in collated_target_masks], dim=0) # b x p x n2
        target_masks = target_masks.permute(1, 0, 2).to(device)

        return context_masks, target_masks

    def forward(
        self,
        xs: torch.Tensor,
        context_masks: torch.Tensor,
        target_masks: Union[torch.Tensor, List[torch.Tensor]],
    ):
        assert context_masks is not None and target_masks is not None, "Masks are required for JEPAModule during training"

        # len(context_masks) = k, len(target_masks) = p
        # context_masks k x b x n1
        # target_masks p x b x n2, or a list of b x n_i tensors for
        # group-specific target scales.
        k = context_masks.shape[0]
        b = context_masks.shape[1]
        context_out = self.context_encoder.forward_features(xs, masks=context_masks, mask_type='tubelet')        # do we need the same or separate pos_embed compared to jepa decoder
        context_patch_tokens = context_out["x_norm_patchtokens"] # (b k) x (t n1) x c
        
        context_patch_tokens = einops.rearrange(
            context_patch_tokens,
            "b (t n) c -> b t n c",
            n=context_masks.shape[-1],
        )

        with torch.no_grad():
            target_out = self.target_encoder.forward_features(xs) 
        target_patch_tokens = target_out["x_norm_patchtokens"] # b x (t n) x c

        target_patch_tokens = einops.rearrange(
            target_patch_tokens,
            "b (t n) c -> b t n c",
            n=xs.shape[-2],
        )

        def predict_mask_group(mask_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            p = mask_group.shape[0]
            predictor_out = self.predictor(
                context_patch_tokens,
                context_masks=context_masks,
                masks=mask_group,
                context_pos_embed=self.target_encoder.pos_embed,
            )
            predictor_out = torch.cat(predictor_out, dim=0)
            predictor_out = einops.rearrange(
                predictor_out,
                "(p k b) t n c -> k p b t n c",
                k=k,
                p=p,
                b=b,
                n=mask_group.shape[-1],
            )

            target_masked = self.target_encoder.apply_tubelet_masks(
                target_patch_tokens,
                masks=mask_group,
            )
            target_masked = einops.rearrange(
                target_masked,
                "(p b) t n c -> p b t n c",
                b=b,
                p=p,
            )
            target_masked = einops.repeat(
                target_masked,
                "p b t n c -> k p b t n c",
                k=k,
            )
            return predictor_out, target_masked.detach()

        if isinstance(target_masks, torch.Tensor):
            predictor_out, target_masked = predict_mask_group(target_masks)
            return self.jepa_loss(predictor_out, target_masked)

        # Predictor batches masks with the same token count. Loss is first
        # reduced per target, then averaged across targets, so large global
        # masks do not outweigh small local masks merely by containing more
        # sensor tokens.
        masks_by_size: Dict[int, List[torch.Tensor]] = {}
        for target_mask in target_masks:
            masks_by_size.setdefault(target_mask.shape[-1], []).append(target_mask)

        per_target_losses = []
        for same_size_masks in masks_by_size.values():
            mask_group = torch.stack(same_size_masks, dim=0)
            predictor_out, target_masked = predict_mask_group(mask_group)
            per_target_losses.append(
                F.mse_loss(predictor_out, target_masked, reduction="none").mean(
                    dim=(0, 2, 3, 4, 5)
                )
            )
        return torch.cat(per_target_losses).mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step = self.step + 1
        self.generator.manual_seed(self.step)
        x = batch["sensor"]
        context_masks = batch.get("context_masks")
        target_masks = batch.get("target_masks")
        if (context_masks is None) != (target_masks is None):
            raise ValueError("A batch must contain both context_masks and target_masks, or neither")
        if context_masks is None:
            context_masks, target_masks = self.sample_jepa_masks(
                x,
                graph_info=batch.get("graph"),
            )
        context_masks = context_masks.to(x.device)
        if isinstance(target_masks, torch.Tensor):
            target_masks = target_masks.to(x.device)
        else:
            target_masks = [target_mask.to(x.device) for target_mask in target_masks]

        ssl_loss = self.forward(x, context_masks, target_masks)

        output = {
            "ssl_loss": ssl_loss.item(),
            "loss": ssl_loss
        }
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("online_probes")}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        log.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        log.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = self.optim_partial(optim_groups)
        if self.lr_scheduler_partial is None:
            return optimizer, None, None

        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        self._schedule_total_steps = int(num_epochs * num_iterations_per_epoch)
        self._reset_momentum_schedule(0)

        if self.wd_scheduler_partial is None:
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        wd_scheduler = self.wd_scheduler_partial(
            optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
        )
        return (
            optimizer,
            {"scheduler": lr_scheduler, "interval": "step", "monitor": None},
            {"wd_scheduler": wd_scheduler, "interval": "step", "frequency": 1},
        )

    def _reset_momentum_schedule(self, start_step: int) -> None:
        if self._schedule_total_steps is None:
            raise RuntimeError("Momentum schedule has not been configured")
        if not 0 <= start_step <= self._schedule_total_steps:
            raise RuntimeError("Invalid JEPA momentum schedule position")
        self._schedule_step = start_step
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i * (self.moving_average_decay[1] - self.moving_average_decay[0]) / self._schedule_total_steps
                for i in range(start_step, self._schedule_total_steps + 1)
            )

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "mask_step": self.step,
            "schedule_step": self._schedule_step,
            "schedule_total_steps": self._schedule_total_steps,
        }

    def load_checkpoint_state(self, state, global_step: int, current_epoch: int) -> None:
        if state is None:
            self.step = global_step - 1
            start_step = global_step
        else:
            if int(state["schedule_total_steps"]) != self._schedule_total_steps:
                raise RuntimeError("JEPA total schedule length changed since the checkpoint was created")
            self.step = int(state["mask_step"])
            start_step = int(state["schedule_step"])
        self._reset_momentum_schedule(start_step)

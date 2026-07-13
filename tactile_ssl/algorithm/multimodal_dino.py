# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functools import partial
import einops
from omegaconf import ListConfig

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.algorithm import Module
from tactile_ssl.loss.dino_loss import DINOLoss
from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.utils.ema import update_moving_average
from tactile_ssl.model import MultimodalTransformer

from tactile_ssl.utils.masking import sample_masks

log = get_pylogger(__name__)


class MultimodalDINOModule(Module, nn.Module):
    def __init__(
        self,
        encoder: MultimodalTransformer,
        dino_head: partial,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        global_mask_scales: Dict[str, Tuple[float, float]],
        local_mask_scales: Dict[str, Tuple[float, float]],
        num_global_masks: int = 1,
        num_local_masks: int = 4,
        min_keep_num_patches: List[int] = [4],
        allow_mask_overlap: bool = False,
        online_probes: Optional[List[nn.Module]] = None,
        online_probes_lrs: List[float] = [],
        moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
        teacher_temp: Union[float, Tuple[float, ...]] = (0.04, 0.07),
        teacher_warmup_epochs: int = 10,
        use_momentum=True,
        log_freq_reconstruction: int = 1000,
    ):
        super().__init__()
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum
        self.global_mask_scales = global_mask_scales
        self.local_mask_scales = local_mask_scales
        self.num_global_masks = num_global_masks
        self.num_local_masks = num_local_masks
        self.min_keeps = min_keep_num_patches
        self.allow_mask_overlap = allow_mask_overlap
        self.log_freq_img = log_freq_reconstruction
        self.modals = encoder.modals
        self.num_modals = encoder.num_modals

        assert len(self.min_keeps) == self.num_modals
        assert len(global_mask_scales) == self.num_modals
        assert len(local_mask_scales) == self.num_modals

        self.generator = torch.Generator()
        self.step = -1

        # Encoders
        dino_head = partial(dino_head, in_dim=encoder.embed_dim)

        self.student_encoder_dict, self.teacher_encoder_dict = dict(), dict()
        self.student_encoder_dict["backbone"] = encoder
        self.student_encoder_dict["dino_head"] = dino_head()
        self.student_encoder = nn.ModuleDict(self.student_encoder_dict)

        self.teacher_encoder_dict["backbone"] = copy.deepcopy(encoder)
        self.teacher_encoder_dict["dino_head"] = dino_head()
        self.teacher_encoder = nn.ModuleDict(self.teacher_encoder_dict)
        self.teacher_encoder.requires_grad_(False)

        self.dino_losses: Dict[str, DINOLoss] = nn.ModuleDict(
            {
                modal: DINOLoss(out_dim=self.student_encoder_dict["dino_head"].last_layer.out_features)
                for modal in self.modals
            }
        )

        # Online probes
        self.online_probes = [] if online_probes is None else nn.ModuleList(online_probes)
        self.online_probes_lrs = [] if online_probes_lrs is None else online_probes_lrs

        assert len(self.online_probes) == len(
            self.online_probes_lrs
        ), "Number of online probes should match the number of learning rates"

        # Momentum scheduler if moving average decay is a tuple
        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(moving_average_decay, ListConfig)
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay

        self.teacher_temp_scheduler = None
        if not isinstance(teacher_temp, float):
            assert isinstance(teacher_temp, list) or isinstance(teacher_temp, ListConfig)
            assert len(teacher_temp) == 2
            teacher_temp = tuple(teacher_temp)
        self.teacher_temp = teacher_temp
        self.teacher_warmup_epochs = teacher_warmup_epochs
        self._schedule_step = 0
        self._schedule_total_steps = None
        self._schedule_steps_per_epoch = None

    @abstractmethod
    def prepare_data(self, batch: Dict[str, Any], *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self, losses: Dict[str, torch.Tensor], *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def get_global_mask_shapes(self, *args, **kwargs) -> Dict[str, List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_local_mask_shapes(self, *args, **kwargs) -> Dict[str, List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_embed_shapes(self, *args, **kwargs) -> Dict[str, List[int]]:
        raise NotImplementedError

    @abstractmethod
    def log_on_batch_end(
        self, outputs: Dict[str, torch.Tensor], stage: Literal["train", "val"] = "train", trainer_instance=None
    ):
        raise NotImplementedError

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        batch_idx: int,
        trainer_instance=None,
    ):
        assert self.teacher_encoder is not None, "target encoder has not been created"
        self.current_teacher_temp = (
            next(self.teacher_temp_scheduler) if self.teacher_temp_scheduler is not None else self.teacher_temp
        )
        if self.use_momentum:
            moving_average_decay = (
                next(self.momentum_scheduler) if self.momentum_scheduler is not None else self.moving_average_decay
            )
            with torch.no_grad():
                update_moving_average(
                    self.teacher_encoder,
                    self.student_encoder,
                    moving_average_decay,
                )
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)
        self._schedule_step += 1

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        batch_idx: int,
        trainer_instance=None,
    ):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)

    def forward(
        self,
        xs: Dict[str, torch.Tensor],
        global_masks: Dict[str, List[torch.Tensor]],
        local_masks: Dict[str, List[torch.Tensor]],
    ):
        assert global_masks is not None and local_masks is not None, "Masks are required for DINOModule during training"
        student_global_dict = self.student_encoder_dict["backbone"].forward_features(xs, global_masks)
        assert "x_norm_regtokens" in student_global_dict.keys(), "Dino requires backbone to contain 1 register token"
        student_global_cls_tokens = student_global_dict["x_norm_regtokens"]
        student_global_cls_tokens = {
            modal: einops.rearrange(cls_tokens, "(p b) 1 c -> b p c", p=len(global_masks[modal]))
            for modal, cls_tokens in student_global_cls_tokens.items()
        }

        student_local_dict = self.student_encoder_dict["backbone"].forward_features(xs, local_masks)
        student_local_cls_tokens = student_local_dict["x_norm_regtokens"]
        student_local_cls_tokens = {
            modal: einops.rearrange(cls_tokens, "(p b) 1 c -> b p c", p=len(local_masks[modal]))
            for modal, cls_tokens in student_local_cls_tokens.items()
        }

        student_cls_tokens = [
            torch.cat([student_global_cls_tokens[modal], student_local_cls_tokens[modal]], dim=-2)
            for modal in self.modals
        ]

        student_cls_token_split_sizes = [student_cls_token.shape[-2] for student_cls_token in student_cls_tokens]
        student_cls_tokens_cat = torch.cat(student_cls_tokens, dim=-2)
        student_cls_tokens_cat_after_head = self.student_encoder_dict["dino_head"](student_cls_tokens_cat)
        student_cls_tokens_after_head = torch.split(
            student_cls_tokens_cat_after_head, student_cls_token_split_sizes, dim=-2
        )
        student_cls_tokens_after_head = {
            modal: einops.rearrange(cls_token, "b p c -> p b 1 c")
            for modal, cls_token in zip(self.modals, student_cls_tokens_after_head)
        }

        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict["backbone"].forward_features(xs, global_masks)
            teacher_global_cls_tokens = teacher_global_dict["x_norm_regtokens"]
            teacher_global_cls_tokens = [teacher_global_cls_tokens[modal] for modal in self.modals]
            teacher_global_cls_token_split_sizes = [
                teacher_global_cls_token.shape[-2] for teacher_global_cls_token in teacher_global_cls_tokens
            ]
            teacher_global_cls_tokens_cat = torch.cat(teacher_global_cls_tokens, dim=-2)
            teacher_cls_tokens_cat_after_head = self.teacher_encoder_dict["dino_head"](teacher_global_cls_tokens_cat)
            teacher_cls_tokens_cat_after_head = teacher_cls_tokens_cat_after_head.detach()
            teacher_cls_tokens_after_head = torch.split(
                teacher_cls_tokens_cat_after_head, teacher_global_cls_token_split_sizes, dim=-2
            )
            teacher_cls_tokens_after_head = {
                modal: teacher_cls_token_after_head
                for modal, teacher_cls_token_after_head in zip(self.modals, teacher_cls_tokens_after_head)
            }

            teacher_dino_softmax_centered_lists = {
                modal: self.dino_losses[modal]
                .softmax_center_teacher(teacher_cls_tokens_after_head[modal], teacher_temp=self.current_teacher_temp)
                .view(self.num_global_masks, -1, *teacher_cls_tokens_after_head[modal].shape[1:])
                for modal in self.modals
            }

            for modal, dino_loss in self.dino_losses.items():
                dino_loss.update_center(teacher_cls_tokens_after_head[modal])

        losses = {
            modal: self.dino_losses[modal](
                list(student_cls_tokens_after_head[modal]), list(teacher_dino_softmax_centered_lists[modal])
            )
            for modal in self.modals
        }

        return self.postprocess(losses)

    @abstractmethod
    def online_probe(
        self, xs_gt: Dict[str, Dict[str, torch.Tensor]], embeddings: Dict[str, Dict[str, torch.Tensor]], *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        def get_embeddings(xs):
            embeddings = self.teacher_encoder_dict["backbone"].forward_features(xs)["x_norm_patchtokens"]
            return {modal: F.layer_norm(embedding, (embedding.size(-1),)) for modal, embedding in embeddings.items()}

        self.step = self.step + 1
        self.generator.manual_seed(self.step)

        if "main" not in batch:
            batch = {"main": batch}

        xs = {}
        xs_gt = {}
        xs["main"], xs_gt["main"] = self.prepare_data(batch["main"])
        embed_shapes = self.get_embed_shapes()
        global_mask_shapes = self.get_global_mask_shapes()
        local_mask_shapes = self.get_local_mask_shapes()

        global_masks = {}
        local_masks = {}
        for modal in self.modals:
            x = xs["main"][modal]
            embed_shape = embed_shapes[modal]
            global_mask_shape = global_mask_shapes[modal]
            local_mask_shape = local_mask_shapes[modal]
            min_keep = self.min_keeps[modal]

            global_mask, local_mask = sample_masks(
                x.shape[0],
                embed_shape,
                global_mask_shape,
                self.num_global_masks,
                local_mask_shape,
                self.num_local_masks,
                min_keep,
                self.allow_mask_overlap,
                x.device,
            )
            global_masks[modal] = global_mask
            local_masks[modal] = local_mask

        ssl_loss, ssl_outputs = self.forward(xs["main"], global_masks, local_masks)

        if self.online_probes is not None:
            with torch.no_grad():
                if "supp" in batch:
                    xs["supp"], xs_gt["supp"] = self.prepare_data(batch["supp"])
                embeddings = {k: get_embeddings(v) for k, v in xs.items()}
            online_probe_loss, online_probe_outputs = self.online_probe(xs_gt, embeddings)
        else:
            online_probe_loss, online_probe_outputs = torch.tensor(0), {}

        output = {}
        if isinstance(ssl_outputs, dict):
            output.update(ssl_outputs)

        loss = ssl_loss + online_probe_loss
        output.update(
            {
                "loss": loss,
                "ssl_loss": ssl_loss.item(),
                "online_probe_loss": online_probe_loss.item(),
            }
        )
        output.update(online_probe_outputs)

        return output

    def validation_step(self, batch: Union[Dict[str, Any], List[Any]], batch_idx: int) -> Dict[str, torch.Tensor]:
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

        for probe, lr in zip(self.online_probes, self.online_probes_lrs):
            trainable_probe_params = {pn: p for pn, p in probe.named_parameters() if p.requires_grad}
            optim_groups.append({"params": trainable_probe_params.values(), "lr": lr})

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
        self._schedule_steps_per_epoch = num_iterations_per_epoch
        self._reset_training_schedules(0)

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

    def _reset_training_schedules(self, start_step: int) -> None:
        if self._schedule_total_steps is None or self._schedule_steps_per_epoch is None:
            raise RuntimeError("Training schedules have not been configured")
        if not 0 <= start_step <= self._schedule_total_steps:
            raise RuntimeError("Invalid DINO training schedule position")
        self._schedule_step = start_step
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i * (self.moving_average_decay[1] - self.moving_average_decay[0]) / self._schedule_total_steps
                for i in range(start_step, self._schedule_total_steps + 1)
            )
        self.current_teacher_temp = self.teacher_temp
        if isinstance(self.teacher_temp, tuple):
            self.teacher_temp_scheduler = self.teacher_temp_schedule(
                self._schedule_total_steps,
                self._schedule_steps_per_epoch,
                start_step=start_step,
            )
            self.current_teacher_temp = self._teacher_temp_at_step(max(start_step - 1, 0))

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "mask_step": self.step,
            "schedule_step": self._schedule_step,
            "schedule_total_steps": self._schedule_total_steps,
            "schedule_steps_per_epoch": self._schedule_steps_per_epoch,
        }

    def load_checkpoint_state(self, state, global_step: int, current_epoch: int) -> None:
        if state is None:
            self.step = global_step - 1
            start_step = global_step
        else:
            if int(state["schedule_total_steps"]) != self._schedule_total_steps:
                raise RuntimeError("DINO total schedule length changed since the checkpoint was created")
            if int(state["schedule_steps_per_epoch"]) != self._schedule_steps_per_epoch:
                raise RuntimeError("DINO steps per epoch changed since the checkpoint was created")
            self.step = int(state["mask_step"])
            start_step = int(state["schedule_step"])
        self._reset_training_schedules(start_step)

    def _teacher_temp_at_step(self, step: int):
        warmup_steps = self.teacher_warmup_epochs * self._schedule_steps_per_epoch
        if warmup_steps == 0:
            return self.teacher_temp[1]
        if step > warmup_steps:
            return self.teacher_temp[1]
        return self.teacher_temp[0] + step * (self.teacher_temp[1] - self.teacher_temp[0]) / warmup_steps

    def teacher_temp_schedule(self, total_steps, num_iterations_per_epoch, start_step=0):
        assert isinstance(self.teacher_temp, tuple), "Teacher temp must be a tuple if this function is called"
        for i in range(start_step, total_steps + 1):
            teacher_temp = None
            if i > (self.teacher_warmup_epochs * num_iterations_per_epoch):
                teacher_temp = self.teacher_temp[1]
            else:
                teacher_temp = self.teacher_temp[0] + i * (self.teacher_temp[1] - self.teacher_temp[0]) / (
                    self.teacher_warmup_epochs * num_iterations_per_epoch
                )
            yield teacher_temp

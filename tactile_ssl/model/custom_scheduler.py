# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        steps_per_epoch,
        start_lr,
        T_max,
        warmup_epochs=10,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.T_max = T_max - self.warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self._step_count < self.warmup_steps:
                progress = float(self._step_count) / float(max(1, self.warmup_steps))
                new_lr = self.start_lr + progress * (base_lr - self.start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step_count - self.warmup_steps) / float(max(1, self.T_max))
                new_lr = max(
                    self.final_lr,
                    self.final_lr + (base_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            lrs.append(new_lr)
        return lrs


class CosineWDSchedule(object):
    def __init__(self, optimizer, ref_weight_decay, T_max, final_weight_decay=0.0):
        self.optimizer = optimizer
        self.ref_weight_decay = ref_weight_decay
        self.final_weight_decay = final_weight_decay
        self.T_max = T_max
        self._step = 0.0

    def state_dict(self):
        return {
            "ref_weight_decay": self.ref_weight_decay,
            "final_weight_decay": self.final_weight_decay,
            "T_max": self.T_max,
            "_step": self._step,
        }

    def load_state_dict(self, state_dict):
        self.ref_weight_decay = state_dict["ref_weight_decay"]
        self.final_weight_decay = state_dict["final_weight_decay"]
        self.T_max = state_dict["T_max"]
        self._step = state_dict["_step"]
        self.apply_current_value()

    def get_current_value(self):
        progress = self._step / self.T_max
        new_wd = self.final_weight_decay + (self.ref_weight_decay - self.final_weight_decay) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_weight_decay <= self.ref_weight_decay:
            return max(self.final_weight_decay, new_wd)
        return min(self.final_weight_decay, new_wd)

    def apply_current_value(self):
        new_wd = self.get_current_value()
        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd

    def step(self):
        self._step += 1
        return self.apply_current_value()

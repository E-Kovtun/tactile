import math

import pytest
import torch

from tactile_ssl.model.custom_scheduler import CosineWDSchedule, WarmupCosineScheduler
from tactile_ssl.trainer.trainer import CHECKPOINT_FORMAT_VERSION, Trainer


def _make_schedulers():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.04)
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=2,
        start_lr=1e-5,
        T_max=20,
        warmup_epochs=2,
        final_lr=1e-6,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_weight_decay=0.04,
        final_weight_decay=0.4,
        T_max=20,
    )
    return optimizer, lr_scheduler, wd_scheduler


def _advance(lr_scheduler, wd_scheduler, steps):
    for _ in range(steps):
        lr_scheduler.step()
        wd_scheduler.step()


@pytest.mark.parametrize("legacy", [False, True])
def test_restore_scheduler_keeps_current_optimizer_and_schedule_position(legacy):
    _, saved_lr_scheduler, saved_wd_scheduler = _make_schedulers()
    _advance(saved_lr_scheduler, saved_wd_scheduler, steps=7)

    saved_lr = saved_lr_scheduler.get_last_lr()[0]
    saved_wd = saved_wd_scheduler.get_current_value()
    if legacy:
        lr_payload = {"scheduler": saved_lr_scheduler, "interval": "step", "monitor": None}
        wd_payload = {"wd_scheduler": saved_wd_scheduler, "interval": "step", "frequency": 1}
        version = 1
    else:
        lr_payload = saved_lr_scheduler.state_dict()
        wd_payload = saved_wd_scheduler.state_dict()
        version = CHECKPOINT_FORMAT_VERSION

    optimizer, lr_scheduler, wd_scheduler = _make_schedulers()
    if not legacy:
        optimizer.param_groups[0]["lr"] = saved_lr
        optimizer.param_groups[0]["weight_decay"] = saved_wd
    Trainer._restore_scheduler(
        {"scheduler": lr_scheduler},
        lr_payload,
        object_key="scheduler",
        checkpoint_version=version,
    )
    Trainer._restore_scheduler(
        {"wd_scheduler": wd_scheduler},
        wd_payload,
        object_key="wd_scheduler",
        checkpoint_version=version,
    )

    assert lr_scheduler.optimizer is optimizer
    assert wd_scheduler.optimizer is optimizer
    assert optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(saved_wd)

    lr_scheduler.step()
    wd_scheduler.step()
    saved_lr_scheduler.step()
    saved_wd_scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr_scheduler.get_last_lr()[0])
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(saved_wd_scheduler.get_current_value())


def test_scheduler_state_does_not_capture_optimizer():
    _, lr_scheduler, wd_scheduler = _make_schedulers()
    _advance(lr_scheduler, wd_scheduler, steps=3)

    assert "optimizer" not in lr_scheduler.state_dict()
    assert "optimizer" not in wd_scheduler.state_dict()
    assert wd_scheduler.state_dict()["_step"] == 3
    expected_wd = 0.4 + (0.04 - 0.4) * 0.5 * (1.0 + math.cos(math.pi * 3 / 20))
    assert wd_scheduler.get_current_value() == pytest.approx(expected_wd)


def test_new_checkpoint_rejects_optimizer_scheduler_mismatch():
    _, saved_lr_scheduler, _ = _make_schedulers()
    for _ in range(4):
        saved_lr_scheduler.step()
    _, current_lr_scheduler, _ = _make_schedulers()

    with pytest.raises(RuntimeError, match="do not match optimizer values"):
        Trainer._restore_scheduler(
            {"scheduler": current_lr_scheduler},
            saved_lr_scheduler.state_dict(),
            object_key="scheduler",
            checkpoint_version=CHECKPOINT_FORMAT_VERSION,
        )

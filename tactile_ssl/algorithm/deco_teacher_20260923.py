"""Teacher-only stochastic-depth ablation; EMA and student remain unchanged."""
from tactile_ssl.algorithm.xela_jepa import XelaJEPAModule
import json
import torch


class DecoTeacherModeJEPAModule(XelaJEPAModule):
    def __init__(self, *, deterministic_teacher=False, **kwargs):
        super().__init__(**kwargs)
        self.deterministic_teacher = bool(deterministic_teacher)
        if self.deterministic_teacher:
            self.target_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.deterministic_teacher:
            self.target_encoder.eval()
        return self

    def on_fit_start(self, train_dataloader, val_dataloader, trainer_instance=None):
        super().on_fit_start(train_dataloader, val_dataloader, trainer_instance)
        sampler = train_dataloader.sampler
        if not (getattr(sampler, "shuffle", False) or isinstance(sampler, torch.utils.data.RandomSampler)):
            raise RuntimeError("This ablation requires a shuffled distributed sampler")
        print("DECO_ABLATION_SAMPLER", type(sampler).__name__, "shuffle=True", flush=True)

    def on_train_batch_start(self, batch, batch_idx):
        if self.deterministic_teacher and self.target_encoder.training:
            raise RuntimeError("Deterministic teacher unexpectedly entered train mode")
        if not getattr(self, "_batch_audited", False):
            groups = batch["group_id"].detach().cpu().tolist()
            gathered = [groups]
            if torch.distributed.is_initialized():
                gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, groups)
            print("DECO_ABLATION_BATCH", json.dumps({
                "global_batch": sum(len(g) for g in gathered),
                "distinct_episodes": len(set(x for g in gathered for x in g)),
                "teacher_training": self.target_encoder.training,
                "student_training": self.context_encoder.training,
                "context_tokens": batch["context_masks"].shape[-1],
            }), flush=True)
            self._batch_audited = True

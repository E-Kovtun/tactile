from typing import Any, Dict, Optional, List
from functools import partial
import einops

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule, gather_batch_tensor
from tactile_ssl.downstream_task.d360_sl import D360SLModule
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed
from tactile_ssl.model.xela_transformer import XelaTransformer
from tactile_ssl.data.xela.utils import get_pad_xela_indexes

from tactile_ssl.utils.plotting_forces import plot_correlation, plot_forces_error
from tactile_ssl.model import VIT_EMBED_DIMS

from tactile_ssl.model.d360_transformer import D360Transformer

log = get_pylogger(__name__)



class XelaObjectSLModule(SLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert isinstance(self.model_encoder, XelaTransformer), "Model encoder must be a XelaTransformer"
        self.sequence_length, self.time_chunk_size = (
            self.model_encoder.sequence_length,
            self.model_encoder.time_chunk_size,
        )
        self.train_pred, self.train_gt = [], []
        self.val_pred, self.val_gt = [], []
        self.test_pred, self.test_gt = [], []

    def _graph_to_device(self, graph_info: Optional[Dict[str, torch.Tensor]], device: torch.device):
        if graph_info is None:
            return None
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in graph_info.items()}

    def _forward_encoder(self, sensor_data, graph_info=None):
        if graph_info is not None and getattr(self.model_encoder, "supports_graph_info", False):
            return self.model_encoder.forward_features(sensor_data, graph_info=graph_info)
        return self.model_encoder.forward_features(sensor_data)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if (
            trainer_instance is not None
            and trainer_instance.fabric.is_global_zero
            and trainer_instance.should_log
        ):
            trainer_instance.writer.add_scalar(f"{label}/loss", outputs["loss"].item(), step)

            metric = "batch_accuracy"
            trainer_instance.writer.add_scalar(f"{label}/{metric}", outputs[f"{metric}"].item(), step)

    def forward(self, batch, batch_idx):
        sensor_data = batch["sensor"]
        graph_info = self._graph_to_device(batch.get("graph"), sensor_data.device)
        encoder_output = self._forward_encoder(sensor_data, graph_info=graph_info)
        if self.model_encoder.num_register_tokens > 0:
            cls_embedding = encoder_output["x_norm_regtokens"].squeeze(1)
        else:
            assert self.model_encoder.num_register_tokens == 0
            cls_embedding = torch.mean(encoder_output["x_norm_patchtokens"], dim=1)
        if self.train_encoder:
            pred_logits = self.model_task(cls_embedding)
        else:
            pred_logits = self.model_task(cls_embedding.detach())
        return pred_logits

    def step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        output = {}
        pred_logits = self.forward(batch, batch_idx)
        gt_labels = batch["object_classification"]

        if self.model_task.class_weights is not None:
            loss = torch.nn.CrossEntropyLoss(weight=self.model_task.class_weights.to(gt_labels.device))(pred_logits, gt_labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(pred_logits, gt_labels)
        output["loss"] = loss

        pred_labels = torch.argmax(pred_logits, dim=1).detach()
        output["pred_labels"] = pred_labels.detach()

        batch_accuracy = (pred_labels == gt_labels).float().mean()
        output["batch_accuracy"] = batch_accuracy

        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.step(batch, batch_idx)

    def on_train_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.train_pred.append(outputs["pred_labels"])
        self.train_gt.append(batch["object_classification"])
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.val_pred.append(outputs["pred_labels"])
        self.val_gt.append(batch["object_classification"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_test_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.test_pred.append(outputs["pred_labels"])
        self.test_gt.append(batch["object_classification"])

    def on_train_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="train")

    def on_validation_epoch_end(self, trainer_instance=None):
        return self.on_epoch_end(trainer_instance, stage="val")

    def on_epoch_end(self, trainer_instance=None, stage="train"):

        if stage == "train":
            target_gt = torch.cat(self.train_gt, dim=0)
            target_pred = torch.cat(self.train_pred, dim=0)
        elif stage == "val":
            target_gt = torch.cat(self.val_gt, dim=0)
            target_pred = torch.cat(self.val_pred, dim=0)

        target_gt = gather_batch_tensor(target_gt).cpu().numpy()
        target_pred = gather_batch_tensor(target_pred).cpu().numpy()

        epoch_accuracy = (target_pred == target_gt).mean()

        step = trainer_instance.global_step if stage=="train" else trainer_instance.global_val_step
        epoch = trainer_instance.current_epoch

        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            trainer_instance.writer.add_scalar(f"{stage}/accuracy", epoch_accuracy, epoch)
         
        if stage == "train":
            self.train_pred = []
            self.train_gt = []
        elif stage == "val":
            self.val_pred = []
            self.val_gt = []
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def on_test_end(self, trainer_instance=None, stage="test"):

        target_gt = gather_batch_tensor(torch.cat(self.test_gt, dim=0)).cpu().numpy()
        target_pred = gather_batch_tensor(torch.cat(self.test_pred, dim=0)).cpu().numpy()

        test_accuracy = (target_pred == target_gt).mean()
        if trainer_instance is not None and trainer_instance.fabric.is_global_zero:
            trainer_instance.writer.add_scalar(f"{stage}/accuracy", test_accuracy, 0)

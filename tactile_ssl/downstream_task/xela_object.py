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
from tactile_ssl.downstream_task.concat_embedding_baseline import LearnedConcatEmbeddingFusion
from tactile_ssl.downstream_task.spatial_gatv2 import XelaSpatialGATv2Encoder
from tactile_ssl.downstream_task.spatial_wl_mlp import XelaSpatialWLMLPEncoder
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed
from tactile_ssl.model.xela_transformer import XelaTransformer
from tactile_ssl.data.xela.utils import get_pad_xela_indexes

from tactile_ssl.utils.plotting_forces import plot_correlation, plot_forces_error
from tactile_ssl.model import VIT_EMBED_DIMS

from tactile_ssl.model.d360_transformer import D360Transformer

log = get_pylogger(__name__)


class XelaObjectSpatialMLPClassifier(nn.Module):
    """Object classifier with a supervised global embedding of sensor XYZ coordinates."""

    supports_spatial_coords = True

    def __init__(
        self,
        input_embed_dim: int,
        classes: List[str],
        spatial_hidden_dims: List[int],
        class_weights: Optional[List[float]] = None,
        coordinate_dim: int = 3,
    ):
        super().__init__()
        if coordinate_dim <= 0:
            raise ValueError("coordinate_dim must be positive")
        if not spatial_hidden_dims:
            raise ValueError("spatial_hidden_dims must contain at least the output embedding dimension")
        if any(dim <= 0 for dim in spatial_hidden_dims):
            raise ValueError("all spatial_hidden_dims values must be positive")

        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.coordinate_dim = coordinate_dim
        self.spatial_hidden_dims = tuple(spatial_hidden_dims)

        spatial_dims = [coordinate_dim, *spatial_hidden_dims]
        spatial_layers = []
        for layer_idx, (in_dim, out_dim) in enumerate(zip(spatial_dims[:-1], spatial_dims[1:])):
            spatial_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < len(spatial_dims) - 2:
                spatial_layers.append(nn.GELU())

        self.spatial_encoder = nn.Sequential(*spatial_layers)
        self.spatial_norm = nn.LayerNorm(spatial_hidden_dims[-1])
        self.fusion = nn.Linear(input_embed_dim + spatial_hidden_dims[-1], input_embed_dim)
        self.fusion_norm = nn.LayerNorm(input_embed_dim)
        self.probe = nn.Linear(input_embed_dim, self.num_classes)

    def forward(self, signal_embedding, spatial_coords=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaObjectSpatialMLPClassifier")
        if spatial_coords.ndim != 4:
            raise ValueError(
                "spatial_coords must have shape [batch, time, sensors, coordinates]; "
                f"got {tuple(spatial_coords.shape)}"
            )
        if spatial_coords.shape[0] != signal_embedding.shape[0]:
            raise ValueError("spatial_coords and signal_embedding must have matching batch dimensions")
        if spatial_coords.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"expected {self.coordinate_dim} coordinate channels, got {spatial_coords.shape[-1]}"
            )

        spatial_embedding = self.spatial_norm(self.spatial_encoder(spatial_coords))
        spatial_embedding = spatial_embedding.mean(dim=(1, 2))
        fused = self.fusion(torch.cat([signal_embedding, spatial_embedding], dim=-1))
        return self.probe(self.fusion_norm(fused))


class XelaObjectConcatEmbeddingBaselineClassifier(nn.Module):
    """Object-classification fusion baseline with a learned non-spatial embedding."""

    def __init__(
        self,
        input_embed_dim: int,
        classes: List[str],
        class_weights: Optional[List[float]] = None,
        baseline_embedding_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.baseline_fusion = LearnedConcatEmbeddingFusion(
            signal_embed_dim=input_embed_dim,
            baseline_embedding_dim=baseline_embedding_dim,
        )
        self.probe = nn.Linear(input_embed_dim, self.num_classes)

    def forward(self, signal_embedding):
        return self.probe(self.baseline_fusion(signal_embedding))


class XelaObjectSpatialWLMLPClassifier(nn.Module):
    """Object classifier with a physical-graph WL embedding of sensor coordinates."""

    supports_spatial_coords = True
    supports_spatial_graph = True

    def __init__(
        self,
        input_embed_dim: int,
        classes: List[str],
        spatial_hidden_dims: List[int],
        class_weights: Optional[List[float]] = None,
        spatial_wl_layers: int = 2,
        coordinate_dim: int = 3,
        bridge_k: int = 4,
    ):
        super().__init__()
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.spatial_encoder = XelaSpatialWLMLPEncoder(
            spatial_hidden_dims=spatial_hidden_dims,
            wl_num_layers=spatial_wl_layers,
            coordinate_dim=coordinate_dim,
            bridge_k=bridge_k,
        )
        self.fusion = nn.Linear(input_embed_dim + self.spatial_encoder.output_dim, input_embed_dim)
        self.fusion_norm = nn.LayerNorm(input_embed_dim)
        self.probe = nn.Linear(input_embed_dim, self.num_classes)

    def forward(self, signal_embedding, spatial_coords=None, graph_info=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaObjectSpatialWLMLPClassifier")
        if spatial_coords.ndim != 4:
            raise ValueError(
                "spatial_coords must have shape [batch, time, sensors, coordinates]; "
                f"got {tuple(spatial_coords.shape)}"
            )
        if spatial_coords.shape[0] != signal_embedding.shape[0]:
            raise ValueError("spatial_coords and signal_embedding must have matching batch dimensions")

        mean_coords = spatial_coords.mean(dim=1)
        spatial_embedding = self.spatial_encoder(mean_coords, graph_info).mean(dim=1)
        fused = self.fusion(torch.cat([signal_embedding, spatial_embedding], dim=-1))
        return self.probe(self.fusion_norm(fused))


class XelaObjectSpatialGATv2Classifier(nn.Module):
    """Object classifier with a supervised physical-graph GATv2 coordinate encoder."""

    supports_spatial_coords = True
    supports_spatial_graph = True

    def __init__(
        self,
        input_embed_dim: int,
        classes: List[str],
        spatial_hidden_dims: List[int],
        class_weights: Optional[List[float]] = None,
        spatial_gat_heads: int = 4,
        spatial_gat_dropout: float = 0.0,
        coordinate_dim: int = 3,
        bridge_k: int = 4,
        edge_mode: str = "distance",
    ):
        super().__init__()
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.spatial_encoder = XelaSpatialGATv2Encoder(
            spatial_hidden_dims=spatial_hidden_dims,
            gat_heads=spatial_gat_heads,
            gat_dropout=spatial_gat_dropout,
            coordinate_dim=coordinate_dim,
            bridge_k=bridge_k,
            edge_mode=edge_mode,
        )
        self.fusion = nn.Linear(input_embed_dim + self.spatial_encoder.output_dim, input_embed_dim)
        self.fusion_norm = nn.LayerNorm(input_embed_dim)
        self.probe = nn.Linear(input_embed_dim, self.num_classes)

    def forward(self, signal_embedding, spatial_coords=None, graph_info=None):
        if spatial_coords is None:
            raise ValueError("spatial_coords are required for XelaObjectSpatialGATv2Classifier")
        if spatial_coords.ndim != 4:
            raise ValueError(
                "spatial_coords must have shape [batch, time, sensors, coordinates]; "
                f"got {tuple(spatial_coords.shape)}"
            )
        if spatial_coords.shape[0] != signal_embedding.shape[0]:
            raise ValueError("spatial_coords and signal_embedding must have matching batch dimensions")

        mean_coords = spatial_coords.mean(dim=1)
        spatial_embedding = self.spatial_encoder(mean_coords, graph_info).mean(dim=1)
        fused = self.fusion(torch.cat([signal_embedding, spatial_embedding], dim=-1))
        return self.probe(self.fusion_norm(fused))



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
        spatial_coords = None
        if getattr(self.model_task, "supports_spatial_coords", False):
            if sensor_data.shape[-1] < 6:
                raise ValueError(
                    "The spatial object classifier requires sensor data with three signal and three XYZ channels"
                )
            spatial_coords = sensor_data[..., 3:6]

        graph_info = self._graph_to_device(batch.get("graph"), sensor_data.device)
        encoder_output = self._forward_encoder(sensor_data, graph_info=graph_info)
        if self.model_encoder.num_register_tokens > 0:
            cls_embedding = encoder_output["x_norm_regtokens"].squeeze(1)
        else:
            assert self.model_encoder.num_register_tokens == 0
            cls_embedding = torch.mean(encoder_output["x_norm_patchtokens"], dim=1)
        task_input = cls_embedding if self.train_encoder else cls_embedding.detach()
        if getattr(self.model_task, "supports_spatial_graph", False):
            pred_logits = self.model_task(task_input, spatial_coords=spatial_coords, graph_info=graph_info)
        elif getattr(self.model_task, "supports_spatial_coords", False):
            pred_logits = self.model_task(task_input, spatial_coords=spatial_coords)
        else:
            pred_logits = self.model_task(task_input)
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

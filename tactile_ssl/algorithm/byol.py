# Taken from https://github.com/lucidrains/byol-pytorch and got modified
from typing import Any, Dict, Tuple, Optional, List, Literal, Union
from functools import partial
import copy
import random
from functools import wraps
import einops
from omegaconf import ListConfig

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.transforms import functional as TF

from tactile_ssl.algorithm.module import Module
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.data.xela_tdex.utils import XELA_IMG_ORDER
from tactile_ssl.utils.ema import update_moving_average
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class PerSampleTactileImageAugment(nn.Module):
    """Apply the original T-Dex crop/blur recipe independently per image.

    torchvision's tensor transforms draw one set of random parameters when a
    whole ``[B, C, H, W]`` batch is passed in.  The original dataset-level
    augmentation was invoked per sample, so preserve that sampling semantics
    when augmentations run inside BYOL.
    """

    def __init__(self, img_size=(224, 224)):
        super().__init__()
        self.img_size = tuple(int(value) for value in img_size)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if images.ndim != 4:
            raise ValueError(f"Expected image or image batch, got {tuple(images.shape)}")
        batch, _, height, width = images.shape
        device, dtype = images.device, images.dtype

        # Independent near-full random crops, evaluated in one grid_sample.
        apply_crop = torch.rand(batch, device=device) < 0.5
        scale = torch.empty(batch, device=device).uniform_(0.9, 1.0)
        log_ratio = torch.empty(batch, device=device).uniform_(
            float(torch.log(torch.tensor(0.75))),
            float(torch.log(torch.tensor(4.0 / 3.0))),
        )
        ratio = log_ratio.exp()
        crop_h = (scale / ratio).sqrt().clamp(max=1.0)
        crop_w = (scale * ratio).sqrt().clamp(max=1.0)
        crop_h = torch.where(apply_crop, crop_h, torch.ones_like(crop_h))
        crop_w = torch.where(apply_crop, crop_w, torch.ones_like(crop_w))
        center_y = crop_h / 2 + torch.rand(batch, device=device) * (1 - crop_h)
        center_x = crop_w / 2 + torch.rand(batch, device=device) * (1 - crop_w)
        theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = crop_w
        theta[:, 1, 1] = crop_h
        theta[:, 0, 2] = 2 * center_x - 1
        theta[:, 1, 2] = 2 * center_y - 1
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # Blur is vectorized; its Bernoulli decision remains independent per image.
        sigma = float(torch.empty((), device=device).uniform_(1.0, 2.0).item())
        blurred = TF.gaussian_blur(images, kernel_size=[3, 3], sigma=[sigma, sigma])
        blur_mask = (torch.rand(batch, device=device) < 0.5).view(batch, 1, 1, 1)
        images = torch.where(blur_mask, blurred, images)
        images = (images - self.mean.to(dtype=dtype)) / self.std.to(dtype=dtype)
        return images.squeeze(0) if squeeze else images


# MLP class for projector and predictor


def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False),
    )


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4096):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReconstructionDecoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 368*3),
        )
        self.probe_type = "reconstruction"

    def forward(self, embedding):
        values = self.net(embedding)
        values = einops.rearrange(values, "b (s c) -> b s c", s=368, c=3)
        return values

    def visualize(self, decoded_x, tactile_values):
        from tactile_ssl.data.xela_tdex_tactile import dump_tactile_state
        import matplotlib.pyplot as plt
        import numpy as np

        decoded_x = decoded_x.detach().cpu().numpy()
        fig1 = dump_tactile_state(decoded_x[0])
        fig2 = dump_tactile_state(tactile_values[0].detach().cpu().numpy())
        fig1.suptitle("Decoded XELA Measurement")
        fig1.canvas.draw()
        fig1_image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        fig1_image = fig1_image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
        fig2.suptitle("Target XELA measurement")

        fig2.canvas.draw()
        fig2_image = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        fig2_image = fig2_image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        fig_image = np.concatenate([fig1_image, fig2_image], axis=1)

        plt.close(fig1)
        plt.close(fig2)
        plt.imshow(fig_image)
        plt.axis("off")
        plt.show()

class ClassificationDecoder(nn.Module):
    def __init__(self, input_dim: int = 512, classes: List[str] = None, class_weights: List[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = classes
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.probe = nn.Sequential(nn.Linear(input_dim, self.num_classes))
        self.probe_type = "classification"

    def forward(self, x):
        x = self.probe(x)
        return x


class BYOLModule(Module, nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        image_size: int,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        augment_fn: torch.nn.Sequential = None,
        augment_fn2: torch.nn.Sequential = None,
        decoders: Optional[List[nn.Module]] = None,
        decoder_lrs: List[float] = [1e-4],
        moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
        use_momentum=True,
        in_channels=3,
        use_precomputed_views: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        # self.scheduler_partial = scheduler_cfg
        self.use_momentum = use_momentum
        self.use_precomputed_views = bool(use_precomputed_views)

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            T.RandomResizedCrop((image_size, image_size), antialias=True),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )
        # TODO(@akashsharma02): Move augmentations to the dataloader collation level similar to I-JEPA
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # Encoders
        self.online_encoder = nn.Sequential(backbone, projector)
        # self.projector = projector
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        self.decoders = [] if decoders is None else nn.ModuleList(decoders)
        if self.decoders:
            assert len(decoder_lrs) == len(self.decoders), "You must provide a learning rate for each decoder"
            self.decoder_lrs = decoder_lrs
        # Makes prediction from projected representation
        self.online_predictor = MLP(
            projector.output_dim,
            projector.output_dim,
        )
        # self.moving_average_decay = moving_average_decay

        # Momentum scheduler if moving average decay is a tuple
        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(moving_average_decay, ListConfig)
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay
        self._schedule_step = 0
        self._schedule_total_steps = None

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def log_on_batch_end(self, outputs, stage: Literal["train", "val"] = "train", trainer_instance=None):
        step = trainer_instance.global_step if stage=="train" else trainer_instance.global_val_step
        for key, value in outputs.items():
            trainer_instance.writer.add_scalar(f"{stage}/{key}", value, step)


    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        if self.use_momentum:
            moving_average_decay = (
                next(self.momentum_scheduler) if self.momentum_scheduler is not None else self.moving_average_decay
            )
            with torch.no_grad():
                update_moving_average(self.target_encoder, self.online_encoder, moving_average_decay)
        trainer_instance.writer.add_scalar("train/moving_average_decay", moving_average_decay, trainer_instance.global_step)
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)
        self._schedule_step += 1

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)

    # def on_train_epoch_end(self, trainer_instance=None):
    #     trainer_instance.fabric.barrier()
    #     assert (
    #         self.use_momentum
    #     ), "you do not need to update the moving average, since you have turned off momentum for the target encoder"
    #     assert self.target_encoder is not None, "target encoder has not been created yet"
    #     update_moving_average(self.target_encoder, self.online_encoder, self.moving_average_decay)
    #     trainer_instance.fabric.barrier()

    def forward(self, x, return_embedding=False, return_projection=True, image_one=None, image_two=None):
        assert not (
            self.training and x.shape[0] == 1
        ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        if return_embedding:
            return self.target_encoder(x)

        if image_one is None or image_two is None:
            image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self.target_encoder if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        tactile_values = batch["tactile_values"]
        if self.use_precomputed_views:
            ssl_loss = self.forward(
                x,
                image_one=batch["aug_image"],
                image_two=batch["aug_image2"],
            )
        else:
            ssl_loss = self.forward(x)

        with torch.no_grad():
            embedding = self.target_encoder(x)

        output = {
            "ssl_loss": ssl_loss.item(),
        }

        online_probes_loss = 0
        for decoder in self.decoders:
            decoded_x = decoder(embedding)
            if decoder.probe_type == "reconstruction":
                probe_loss = F.l1_loss(decoded_x, tactile_values, reduction="mean")
                output[f"{decoder.probe_type}_loss"] = probe_loss.item()
            elif decoder.probe_type == "classification":
                gt_labels = batch["object_classification"]
                probe_loss = F.cross_entropy(decoded_x, gt_labels, weight=decoder.class_weights.to(decoded_x.device))
                pred_labels = torch.argmax(decoded_x, dim=1)
                accuracy = (pred_labels == gt_labels).float().mean()
                output[f"{decoder.probe_type}_loss"] = probe_loss.item()
                output[f"{decoder.probe_type}_accuracy"] = accuracy
                
            else:
                raise NotImplementedError(f"Probe type {decoder.probe_type} not implemented")
            
            online_probes_loss += probe_loss

        loss = ssl_loss + online_probes_loss
        output["loss"] = loss  # type: ignore
        output["online_probes_loss"] = online_probes_loss

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    # def configure_optimizers(
    #     self, num_iterations_per_epoch, num_epochs
    # ) -> Tuple[torch.optim.Optimizer, Optional[Dict]]:
    #     trainable_encoder_params = [param for param in self.online_encoder.parameters() if param.requires_grad]

    #     optim_groups = [
    #         {"params": trainable_encoder_params},
    #     ]
    #     for decoder, lr in zip(self.decoders, self.decoder_lrs):
    #         trainable_decoder_params = [param for param in decoder.parameters() if param.requires_grad]
    #         optim_groups.append({"params": trainable_decoder_params, "lr": lr})

    #     optimizer = self.optim_partial(optim_groups)
    #     if self.scheduler_partial is None:
    #         return optimizer, None
    #     scheduler = self.scheduler_partial(optimizer=optimizer)
    #     return optimizer, {"scheduler": scheduler, "interval": "epoch"}
    

    def configure_optimizers(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("decoders")}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        for probe, lr in zip(self.decoders, self.decoder_lrs):
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
            raise RuntimeError("Invalid BYOL momentum schedule position")
        self._schedule_step = start_step
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i * (self.moving_average_decay[1] - self.moving_average_decay[0]) / self._schedule_total_steps
                for i in range(start_step, self._schedule_total_steps + 1)
            )

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "schedule_step": self._schedule_step,
            "schedule_total_steps": self._schedule_total_steps,
        }

    def load_checkpoint_state(self, state, global_step: int, current_epoch: int) -> None:
        if state is None:
            start_step = global_step
        else:
            if int(state["schedule_total_steps"]) != self._schedule_total_steps:
                raise RuntimeError("BYOL total schedule length changed since the checkpoint was created")
            start_step = int(state["schedule_step"])
        self._reset_momentum_schedule(start_step)

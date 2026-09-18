from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.models import AlexNet_Weights, alexnet

from tactile_ssl.data.deco_geometry import build_deco_geometry
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.data.xela_tdex.utils import XELA_IMG_ORDER


class AlexnetWrapper(nn.Module):
    """ImageNet AlexNet convolutional trunk used by the original BYOL baseline."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = alexnet(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(image)), 1)


class InstantaneousTactileAlexNet(AlexnetWrapper):
    """Apply an image BYOL AlexNet to one fixed frame of a tactile window.

    Keeping ``features`` and ``avgpool`` directly on this module deliberately
    matches :class:`AlexnetWrapper`'s state-dict layout.  Thus an instantaneous
    BYOL checkpoint can be loaded without inventing or freezing an untrained
    temporal-convolution/classifier layer for downstream evaluation.
    """

    def __init__(
        self,
        rasterizer: nn.Module,
        sequence_length: int,
        frame_index: int = 5,
        pretrained: bool = False,
        image_size: int = 224,
        **_: object,
    ) -> None:
        super().__init__(pretrained=pretrained)
        self.rasterizer = rasterizer
        self.sequence_length = int(sequence_length)
        self.time_chunk_size = self.sequence_length
        self.frame_index = int(frame_index)
        if not 0 <= self.frame_index < self.sequence_length:
            raise ValueError("frame_index must be in [0, sequence_length)")
        self.image_size = int(image_size)
        self.num_chunks = 1
        self.num_register_tokens = 0
        self.embed_dim = 256 * 6 * 6
        self.in_dim = int(rasterizer.num_sensors)
        self.in_chans = int(rasterizer.channels)

        # AlexNet is initialized from its RGB ImageNet weights.  Pressure-only
        # sensors produce a single atlas channel, so collapse the RGB filters
        # without adding a new state-dict prefix (downstream loading relies on
        # the original ``features.*`` layout).
        if self.in_chans == 1:
            original = self.features[0]
            replacement = nn.Conv2d(
                1,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(original.weight.sum(dim=1, keepdim=True))
                if original.bias is not None:
                    replacement.bias.copy_(original.bias)
            self.features[0] = replacement

    def rasterize(self, sensor: torch.Tensor) -> torch.Tensor:
        if sensor.ndim != 4 or sensor.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected tactile window [B, {self.sequence_length}, N, C], got {tuple(sensor.shape)}"
            )
        # Select before rasterization and augmentation.  For a large physical
        # batch this avoids materializing the two frames that this encoder is
        # explicitly defined to discard.
        video = self.rasterizer(sensor[:, self.frame_index : self.frame_index + 1])
        batch, channels, time, height, width = video.shape
        resized = F.interpolate(
            video.reshape(batch, channels * time, height, width),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, channels, time, self.image_size, self.image_size)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != self.in_chans:
            raise ValueError(
                f"Expected raster video [B, {self.in_chans}, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[2] != 1:
            raise ValueError(f"Expected one preselected frame, got {video.shape[2]}")
        image = video[:, :, 0]
        return super().forward(image).unsqueeze(1)

    def forward_features(self, sensor: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        patch_tokens = self.forward_video(self.rasterize(sensor))
        return {
            "x_norm_regtokens": patch_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens,
        }

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        return self.forward_features(sensor)["x_norm_patchtokens"]


class XelaTactileRasterizer(nn.Module):
    """Lay the canonical 368 Xela taxels out on the physical hand atlas."""

    channels = 3
    num_sensors = 368

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        if sensor.ndim != 4 or sensor.shape[2:] != (self.num_sensors, self.channels):
            raise ValueError(
                "Xela rasterizer expects [batch, time, 368, 3], "
                f"got {tuple(sensor.shape)}"
            )
        batch, time = sensor.shape[:2]
        canvas = sensor.new_zeros(batch, self.channels, time, 28, 24)
        start = 0
        for name, count in XELA_FLATTEN_ORDER.items():
            values = sensor[:, :, start : start + count]
            row, col = XELA_IMG_ORDER[name]
            if "aftc" in name:
                grid = sensor.new_zeros(batch, time, 6, 6, self.channels)
                grid[:, :, 2:] = values[:, :, 6:].reshape(batch, time, 4, 6, self.channels)
                grid[:, :, 0, 2:4] = values[:, :, :2]
                grid[:, :, 1, 1:5] = values[:, :, 2:6]
                height, width = 6, 6
            elif "4x4" in name:
                grid = values.reshape(batch, time, 4, 4, self.channels)
                col += 2
                height, width = 4, 4
            elif "4x6" in name:
                grid = values.reshape(batch, time, 4, 6, self.channels)
                height, width = 4, 6
            else:  # pragma: no cover - guarded by the canonical layout
                raise ValueError(f"Unknown Xela pad type {name!r}")
            canvas[:, :, :, row : row + height, col : col + width] = grid.permute(0, 4, 1, 2, 3)
            start += count
        return canvas


class SockTactileRasterizer(nn.Module):
    """Restore the two official physical pressure grids side by side."""

    channels = 1
    num_sensors = 453

    def __init__(self, sensor_map_path: str, gap: int = 2) -> None:
        super().__init__()
        rows = list(csv.DictReader(open(Path(sensor_map_path), newline="")))
        rows.sort(key=lambda row: int(row["global_node_id"]))
        if [int(row["global_node_id"]) for row in rows] != list(range(self.num_sensors)):
            raise ValueError("Sock map must contain contiguous global node ids 0..452")

        side_extents = {}
        for side in ("left", "right"):
            selected = [row for row in rows if row["side"] == side]
            min_row = min(int(row["grid_row"]) for row in selected)
            min_col = min(int(row["grid_col"]) for row in selected)
            max_row = max(int(row["grid_row"]) for row in selected)
            max_col = max(int(row["grid_col"]) for row in selected)
            side_extents[side] = (min_row, min_col, max_row, max_col)
        left_width = side_extents["left"][3] - side_extents["left"][1] + 1
        height = max(ext[2] - ext[0] + 1 for ext in side_extents.values())
        width = left_width + int(gap) + side_extents["right"][3] - side_extents["right"][1] + 1

        flat_indices = []
        for row in rows:
            side = row["side"]
            min_row, min_col, _, _ = side_extents[side]
            grid_row = int(row["grid_row"]) - min_row
            grid_col = int(row["grid_col"]) - min_col
            if side == "right":
                grid_col += left_width + int(gap)
            flat_indices.append(grid_row * width + grid_col)
        if len(set(flat_indices)) != self.num_sensors:
            raise ValueError("Sock rasterizer maps multiple sensors to the same pixel")
        self.height = height
        self.width = width
        self.register_buffer("flat_indices", torch.tensor(flat_indices, dtype=torch.long), persistent=True)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        if sensor.ndim != 4 or sensor.shape[2:] != (self.num_sensors, 1):
            raise ValueError(
                "Sock rasterizer expects [batch, time, 453, 1], "
                f"got {tuple(sensor.shape)}"
            )
        values = sensor.permute(0, 3, 1, 2)
        indices = self.flat_indices.view(1, 1, 1, -1).expand_as(values)
        canvas = values.new_zeros(*values.shape[:-1], self.height * self.width)
        return canvas.scatter(-1, indices, values).reshape(*values.shape[:-1], self.height, self.width)


class DecoTactileRasterizer(nn.Module):
    """Scatter grouped DECO values back to the canonical two-hand atlas."""

    channels = 1
    num_sensors = 528

    def __init__(self, height: int = 256, width: int = 512) -> None:
        super().__init__()
        geometry = build_deco_geometry()
        positions = geometry.raw_positions
        scaled = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0))
        cols = torch.from_numpy((scaled[:, 0] * (width - 1)).round().astype("int64"))
        rows = torch.from_numpy((scaled[:, 1] * (height - 1)).round().astype("int64"))
        flat_indices = rows * width + cols
        if torch.unique(flat_indices).numel() != geometry.num_raw_taxels:
            raise ValueError("DECO atlas resolution is too small to preserve every raw taxel")

        member_mask = torch.from_numpy(geometry.member_mask.copy()).bool()
        group_index, member_index = torch.nonzero(member_mask, as_tuple=True)
        raw_index = torch.from_numpy(geometry.member_indices.copy()).long()[group_index, member_index]
        order = torch.argsort(raw_index)
        self.height = int(height)
        self.width = int(width)
        self.register_buffer("group_index", group_index[order], persistent=True)
        self.register_buffer("member_index", member_index[order], persistent=True)
        self.register_buffer("flat_indices", flat_indices[raw_index[order]], persistent=True)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        if sensor.ndim != 4 or sensor.shape[2:] != (self.num_sensors, 5):
            raise ValueError(
                "DECO rasterizer expects [batch, time, 528, 5], "
                f"got {tuple(sensor.shape)}"
            )
        raw = sensor[:, :, self.group_index, self.member_index].unsqueeze(1)
        indices = self.flat_indices.view(1, 1, 1, -1).expand_as(raw)
        canvas = raw.new_zeros(*raw.shape[:-1], self.height * self.width)
        return canvas.scatter(-1, indices, raw).reshape(*raw.shape[:-1], self.height, self.width)


class TemporalTactileAlexNet(nn.Module):
    """Tactile rasterizer + learned temporal reduction + AlexNet backbone.

    The depthwise temporal convolution is initialized as an average and is part
    of the encoder, so the BYOL target copy updates it through EMA as well.
    """

    def __init__(
        self,
        rasterizer: nn.Module,
        sequence_length: int,
        time_chunk_size: int,
        embedding_dim: int = 1024,
        pretrained: bool = True,
        image_size: int = 224,
        normalization: Optional[DictConfig] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if sequence_length % time_chunk_size:
            raise ValueError("sequence_length must be divisible by time_chunk_size")
        self.rasterizer = rasterizer
        self.sequence_length = int(sequence_length)
        self.time_chunk_size = int(time_chunk_size)
        self.num_chunks = self.sequence_length // self.time_chunk_size
        # Downstream modules use this attribute to decide whether to consume a
        # register token or mean-pool patch tokens. AlexNet has no registers.
        self.num_register_tokens = 0
        self.embed_dim = int(embedding_dim)
        self.in_dim = int(rasterizer.num_sensors)
        self.in_chans = 3 if isinstance(rasterizer, XelaTactileRasterizer) else 5 if isinstance(rasterizer, DecoTactileRasterizer) else 1
        self.image_channels = int(rasterizer.channels)
        self.image_size = int(image_size)

        self.temporal_conv = nn.Conv3d(
            self.image_channels,
            self.image_channels,
            kernel_size=(self.time_chunk_size, 1, 1),
            stride=(self.time_chunk_size, 1, 1),
            groups=self.image_channels,
            bias=False,
        )
        nn.init.constant_(self.temporal_conv.weight, 1.0 / self.time_chunk_size)

        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = alexnet(weights=weights)
        if self.image_channels == 1:
            original = self.backbone.features[0]
            replacement = nn.Conv2d(
                1,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(original.weight.sum(dim=1, keepdim=True))
                if original.bias is not None:
                    replacement.bias.copy_(original.bias)
            self.backbone.features[0] = replacement
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.embed_dim),
        )

        if normalization is None or normalization.get("mean") is None:
            mean, std = torch.zeros(self.in_chans), torch.ones(self.in_chans)
        else:
            mean = torch.as_tensor(normalization.mean, dtype=torch.float32)
            std = torch.as_tensor(normalization.std, dtype=torch.float32)
        self.register_buffer("sensor_mean", mean, persistent=True)
        self.register_buffer("sensor_std", std, persistent=True)

    def normalize(self, sensor: torch.Tensor) -> torch.Tensor:
        if self.sensor_mean.numel() == 1 or self.sensor_mean.numel() == sensor.shape[-1]:
            return (sensor - self.sensor_mean) / self.sensor_std.clamp_min(1e-6)
        return sensor

    def rasterize(self, sensor: torch.Tensor) -> torch.Tensor:
        sensor = self.normalize(sensor)
        video = self.rasterizer(sensor)
        batch, channels, time, height, width = video.shape
        resized = F.interpolate(
            video.reshape(batch, channels * time, height, width),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, channels, time, self.image_size, self.image_size)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected raster video [B, {self.image_channels}, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[2] != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {video.shape[2]}")
        chunks = self.aggregate_video(video)
        return self.forward_aggregated(chunks)

    def aggregate_video(self, video: torch.Tensor) -> torch.Tensor:
        """Reduce each temporal chunk while keeping the atlas representation."""
        if video.ndim != 5 or video.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected raster video [B, {self.image_channels}, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[2] != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {video.shape[2]}")
        return self.temporal_conv(video)

    def forward_aggregated(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode already temporally reduced atlases."""
        if chunks.ndim != 5 or chunks.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected aggregated video [B, {self.image_channels}, chunks, H, W], "
                f"got {tuple(chunks.shape)}"
            )
        batch, channels, num_chunks, height, width = chunks.shape
        images = chunks.permute(0, 2, 1, 3, 4).reshape(batch * num_chunks, channels, height, width)
        embeddings = self.backbone(images)
        return embeddings.reshape(batch, num_chunks, self.embed_dim)

    def forward_features(self, sensor: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        patch_tokens = self.forward_video(self.rasterize(sensor))
        register = patch_tokens.mean(dim=1, keepdim=True)
        return {
            "x_norm_regtokens": register,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens,
        }

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        return self.forward_features(sensor)["x_norm_patchtokens"]

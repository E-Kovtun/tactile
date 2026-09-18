"""Image-style JEPA on fixed sparse Socks/DECO atlases.

The rectangle sampler follows XelaIJEPA2DMaskCollator: desired token fraction,
occupancy-corrected rectangle area, 64 placement attempts, target union removed
from context, then common batch lengths. NumPy batches the placement candidates.
"""
import csv
import math
from functools import lru_cache

import numpy as np
import torch

from tactile_ssl.utils.xela_ijepa_masking import XelaIJEPA2DMaskCollator


@lru_cache(maxsize=4)
def atlas(sensor, sensor_map_path=None):
    if sensor == 'deco':
        from tactile_ssl.data.deco_geometry import build_deco_geometry
        xy = build_deco_geometry().hypertaxel_positions.copy()
        xy -= xy.min(axis=0)
        # Fixed canonical 2D layout, quantized to 1/8 layout unit. No 3D input.
        coordinates = np.rint(xy[:, ::-1] * 8).astype(np.int64)
        expected = 528
    elif sensor == 'sock':
        if sensor_map_path is None:
            raise ValueError('Socks atlas requires the official physical sensor map')
        with open(sensor_map_path, newline='') as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: int(row['global_node_id']))
        assert [int(row['global_node_id']) for row in rows] == list(range(453))
        # Keep native grid orientation and all gaps; place right beside left.
        width = max(int(row['grid_col']) for row in rows) + 1
        coordinates = np.asarray([(int(row['grid_row']), int(row['grid_col']) + (width + 2 if row['side'] == 'right' else 0)) for row in rows])
        coordinates -= coordinates.min(axis=0)
        expected = 453
    else:
        raise ValueError(sensor)
    assert coordinates.shape == (expected, 2)
    assert len(np.unique(coordinates, axis=0)) == expected, 'Atlas collision'
    return coordinates


def fixed_embedding(coordinates, embed_dim):
    if embed_dim % 4:
        raise ValueError('2D sin/cos dimension must be divisible by four')
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    half = embed_dim // 2
    frequencies = 1.0 / (10000.0 ** (torch.arange(half // 2).float() / (half / 2)))
    def encode(position):
        angles = position[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)
    return torch.cat((encode(coordinates[:, 1]), encode(coordinates[:, 0])), dim=1)[None]


class SensorIJEPA2DMaskCollator(XelaIJEPA2DMaskCollator):
    def __init__(self, sensor, sensor_map_path=None, **kwargs):
        super().__init__(**kwargs)
        self.coordinates = torch.from_numpy(atlas(sensor, sensor_map_path).copy())
        self.num_nodes = len(self.coordinates)
        self.height, self.width = (self.coordinates.max(dim=0).values + 1).tolist()
        self._rectangles = {}

    def _block_shape(self, scale, generator):
        desired = max(1, round(self.num_nodes * self._uniform(scale, generator)))
        ratio = self._uniform(self.aspect_ratio, generator)
        area = desired * self.height * self.width / self.num_nodes
        return (max(1, min(self.height, round(math.sqrt(area * ratio)))),
                max(1, min(self.width, round(math.sqrt(area / ratio)))), desired)

    def _sample_rectangle(self, shape, generator, exclude=None, min_keep=1):
        height, width, desired = shape
        def rectangle_matrix(rect_height, rect_width):
            key = (rect_height, rect_width)
            if key in self._rectangles:
                return self._rectangles[key]
            if len(self._rectangles) >= 4:
                self._rectangles.clear()
            yy, xx = np.meshgrid(np.arange(self.height-rect_height+1), np.arange(self.width-rect_width+1), indexing='ij')
            coords = self.coordinates.numpy()
            y, x = yy.ravel()[:, None], xx.ravel()[:, None]
            self._rectangles[key] = ((coords[:, 0] >= y) & (coords[:, 0] < y+rect_height) & (coords[:, 1] >= x) & (coords[:, 1] < x+rect_width))
            return self._rectangles[key]

        matrix = rectangle_matrix(height, width)
        top = torch.randint(self.height-height+1, (self.max_resample_attempts,), generator=generator).numpy()
        left = torch.randint(self.width-width+1, (self.max_resample_attempts,), generator=generator).numpy()
        candidates = matrix[top * (self.width-width+1) + left].copy()
        if exclude is not None:
            candidates[:, exclude.numpy()] = False
        counts = candidates.sum(axis=1)
        error = np.where(counts >= min_keep, np.abs(counts-desired), self.num_nodes+1)
        best = int(error.argmin())
        if counts[best] < min_keep:
            # A sparse atlas can make 64 otherwise valid placements miss the
            # post-target floor. Search every placement before changing shape.
            for expansion in range(0, max(self.height-height, self.width-width) + 1):
                fallback_height = min(self.height, height + expansion)
                fallback_width = min(self.width, width + expansion)
                fallback = rectangle_matrix(fallback_height, fallback_width).copy()
                if exclude is not None:
                    fallback[:, exclude.numpy()] = False
                fallback_counts = fallback.sum(axis=1)
                valid = fallback_counts >= min_keep
                if valid.any():
                    fallback_error = np.where(valid, np.abs(fallback_counts-desired), self.num_nodes+1)
                    return torch.from_numpy(np.flatnonzero(fallback[int(fallback_error.argmin())]))
            raise RuntimeError(f'Atlas cannot retain context floor {min_keep}; shape={shape}')
        return torch.from_numpy(np.flatnonzero(candidates[best]))

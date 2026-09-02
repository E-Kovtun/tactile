from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset


CACHE_FORMAT_VERSION = 2
CACHE_TENSOR_KEYS = (
    "sensor",
    "images",
    "proprio",
    "action",
    "action_valid_mask",
    "condition",
    "sample_id",
    "group_id",
)


class CachedDecoPolicyDataset(Dataset):
    """Memory-mapped DECO policy shards with frozen ResNet features."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        manifest_path: str | Path | None = None,
        max_open_shards: int = 2,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.cache_root / "manifest.json"
        )
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported DECO policy cache format in {self.manifest_path}")
        if split not in payload.get("splits", {}):
            raise KeyError(f"Split {split!r} is missing from {self.manifest_path}")
        self.split = split
        if int(payload.get("action_chunk_size", -1)) != 16:
            raise ValueError(f"Expected a 16-step policy cache in {self.manifest_path}")
        if payload.get("image_storage") != "uint8-rgb-256":
            raise ValueError(f"Expected uint8 RGB images in {self.manifest_path}")
        self.shards = list(payload["splits"][split]["shards"])
        self.length = int(payload["splits"][split]["length"])
        self.stops = [int(shard["stop"]) for shard in self.shards]
        if self.stops and self.stops[-1] != self.length:
            raise ValueError(f"Shard index for {split} stops at {self.stops[-1]}, expected {self.length}")
        self.max_open_shards = int(max_open_shards)
        self._open: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return self.length

    def _load_shard(self, relative_path: str) -> dict[str, torch.Tensor]:
        cached = self._open.pop(relative_path, None)
        if cached is None:
            path = self.cache_root / relative_path
            try:
                cached = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except TypeError:  # torch < 2.1 has no mmap keyword
                cached = torch.load(path, map_location="cpu", weights_only=True)
            missing = sorted(set(CACHE_TENSOR_KEYS) - set(cached))
            if missing:
                raise KeyError(f"Cached shard {path} is missing tensors {missing}")
        self._open[relative_path] = cached
        while len(self._open) > self.max_open_shards:
            self._open.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.stops, index)
        shard_info = self.shards[shard_index]
        local_index = index - int(shard_info["start"])
        shard = self._load_shard(shard_info["file"])
        sample = {key: shard[key][local_index] for key in CACHE_TENSOR_KEYS}
        if sample["images"].dtype != torch.uint8:
            raise TypeError(f"Cached images must be uint8, got {sample['images'].dtype}")
        if sample["action"].shape != (16, 28):
            raise ValueError(f"Expected cached action [16,28], got {tuple(sample['action'].shape)}")
        if sample["action_valid_mask"].shape != (16,):
            raise ValueError("Expected cached action_valid_mask [16]")
        return sample


def evenly_spaced_subset(dataset: Dataset, max_samples: int | None):
    """Return a deterministic subset spread across the complete dataset order."""
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    # Pick bin centers rather than a prefix. The cache is episode-ordered, so
    # this preserves coverage across the whole validation split and its Task-4
    # variants without scanning 89k samples just to construct the subset.
    sample_ids = torch.arange(max_samples, dtype=torch.int64)
    indices = ((2 * sample_ids + 1) * len(dataset) // (2 * max_samples)).tolist()
    return Subset(dataset, indices)


def create_cached_deco_policy_datasets(
    cache_root: str,
    cache_manifest_path: str | None = None,
    max_open_shards: int = 2,
    validation_max_samples: int | None = None,
    **_: Any,
):
    datasets = {
        split: CachedDecoPolicyDataset(
            cache_root,
            split,
            manifest_path=cache_manifest_path,
            max_open_shards=max_open_shards,
        )
        for split in ("train", "val", "test")
    }
    datasets["train"].test_dataset = datasets["test"]
    validation = evenly_spaced_subset(datasets["val"], validation_max_samples)
    return datasets["train"], validation

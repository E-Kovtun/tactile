from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset


COMMON_KEYS = (
    "sensor",
    "proprio",
    "action",
    "action_valid_mask",
    "condition",
    "sample_id",
    "group_id",
)
VISION_CACHE_FORMAT_VERSION = 1
HAND_ACTION_INDICES = (*range(7, 13), *range(20, 26))


class FinalCachedDecoPolicyDataset(Dataset):
    """Join the existing policy cache with aligned frozen-ResNet sidecars."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        cache_manifest_path: str | Path,
        vision_cache_root: str | Path,
        vision_cache_manifest_path: str | Path,
        max_open_shards: int = 2,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.vision_cache_root = Path(vision_cache_root).expanduser().resolve()
        source = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))
        vision = json.loads(Path(vision_cache_manifest_path).read_text(encoding="utf-8"))
        if source.get("format_version") != 2 or source.get("image_storage") != "uint8-rgb-256":
            raise ValueError("Expected the DECO uint8 policy cache")
        if vision.get("format_version") != VISION_CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported frozen-ResNet feature cache")
        if vision.get("feature_storage") != "resnet18-imagenet1k-v1-gap-fp16":
            raise ValueError("The final policy requires standard ResNet18 GAP features")
        self.shards = list(source["splits"][split]["shards"])
        self.vision_shards = list(vision["splits"][split]["shards"])
        self.length = int(source["splits"][split]["length"])
        if int(vision["splits"][split]["length"]) != self.length:
            raise ValueError("Vision and policy split lengths differ")
        source_ranges = [(int(x["start"]), int(x["stop"])) for x in self.shards]
        vision_ranges = [(int(x["start"]), int(x["stop"])) for x in self.vision_shards]
        if source_ranges != vision_ranges:
            raise ValueError("Vision and policy shard ranges differ")
        self.stops = [stop for _, stop in source_ranges]
        self.max_open_shards = int(max_open_shards)
        self._open_policy: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._open_vision: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_policy"] = OrderedDict()
        state["_open_vision"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return self.length

    def _load(self, root: Path, relative: str, cache: OrderedDict):
        payload = cache.pop(relative, None)
        if payload is None:
            try:
                payload = torch.load(root / relative, map_location="cpu", weights_only=True, mmap=True)
            except TypeError:
                payload = torch.load(root / relative, map_location="cpu", weights_only=True)
        cache[relative] = payload
        while len(cache) > self.max_open_shards:
            cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.stops, index)
        local_index = index - int(self.shards[shard_index]["start"])
        policy = self._load(
            self.cache_root, self.shards[shard_index]["file"], self._open_policy
        )
        vision = self._load(
            self.vision_cache_root,
            self.vision_shards[shard_index]["file"],
            self._open_vision,
        )
        sample = {key: policy[key][local_index] for key in COMMON_KEYS}
        for key in ("sample_id", "group_id"):
            if not torch.equal(sample[key], vision[key][local_index]):
                raise ValueError(f"Frozen-ResNet sidecar {key} is misaligned")
        sample["image_features"] = vision["image_features"][local_index]
        if sample["image_features"].shape != (2, 512):
            raise ValueError("Expected image_features [2,512]")
        if sample["action"].shape != (16, 28):
            raise ValueError("Expected source action [16,28]")
        # DECO stores [left arm 7, left hand 6, right arm 7, right hand 6,
        # active camera 2]. The hands-only protocol predicts the two 6-DoF
        # Inspire hand commands and excludes both arms and the active camera.
        sample["action"] = sample["action"][..., HAND_ACTION_INDICES]
        if sample["action"].shape != (16, 12):
            raise AssertionError("Expected hands-only action [16,12]")
        return sample


def evenly_spaced_subset(dataset: Dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    max_samples = int(max_samples)
    ids = torch.arange(max_samples, dtype=torch.int64)
    return Subset(dataset, ((2 * ids + 1) * len(dataset) // (2 * max_samples)).tolist())


def create_final_cached_deco_policy_datasets(
    cache_root: str,
    cache_manifest_path: str,
    vision_cache_root: str,
    vision_cache_manifest_path: str,
    max_open_shards: int = 2,
    validation_max_samples: int | None = None,
    **_: Any,
):
    datasets = {
        split: FinalCachedDecoPolicyDataset(
            cache_root,
            split,
            cache_manifest_path,
            vision_cache_root,
            vision_cache_manifest_path,
            max_open_shards,
        )
        for split in ("train", "val", "test")
    }
    datasets["train"].test_dataset = datasets["test"]
    return datasets["train"], evenly_spaced_subset(datasets["val"], validation_max_samples)

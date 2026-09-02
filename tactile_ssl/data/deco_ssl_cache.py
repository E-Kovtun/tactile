from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_FORMAT_VERSION = 1
_REQUIRED_ARRAYS = {
    "sensor": ((3, 528, 5), np.dtype("float32")),
    "sample_id": ((), np.dtype("int64")),
    "group_id": ((), np.dtype("int64")),
}


class CachedDecoSSLDataset(Dataset):
    """Random-access tactile-only view of the fixed DECO Task-4 split."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.cache_root / "manifest.json"
        )
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported DECO SSL cache format in {self.manifest_path}")
        if int(payload.get("split_seed", -1)) != 42:
            raise ValueError("DECO SSL cache must use the fixed episode split seed 42")
        if int(payload.get("window_size", -1)) != 3 or int(payload.get("stride", -1)) != 3:
            raise ValueError("DECO SSL cache must contain 3-frame, stride-3 windows")
        split_info = payload.get("splits", {}).get(split)
        if split_info is None:
            raise KeyError(f"Split {split!r} is missing from {self.manifest_path}")

        self.split = split
        self.length = int(split_info["length"])
        self._paths = {
            key: self.cache_root / split_info["arrays"][key]
            for key in _REQUIRED_ARRAYS
        }
        self._arrays: dict[str, np.ndarray] | None = None
        self._validate_arrays()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                key: np.load(path, mmap_mode="r", allow_pickle=False)
                for key, path in self._paths.items()
            }
        return self._arrays

    def _validate_arrays(self) -> None:
        arrays = self._open_arrays()
        for key, (sample_shape, dtype) in _REQUIRED_ARRAYS.items():
            array = arrays[key]
            expected_shape = (self.length, *sample_shape)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Cached {key} has shape {array.shape}, expected {expected_shape}"
                )
            if array.dtype != dtype:
                raise TypeError(f"Cached {key} has dtype {array.dtype}, expected {dtype}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        arrays = self._open_arrays()
        # Copy one 31 KiB window out of the read-only mmap. This avoids writable
        # NumPy warnings and lets pinned-memory workers transfer it asynchronously.
        sensor = np.array(arrays["sensor"][index], copy=True)
        return {
            "sensor": torch.from_numpy(sensor),
            "sample_id": torch.tensor(int(arrays["sample_id"][index]), dtype=torch.long),
            "group_id": torch.tensor(int(arrays["group_id"][index]), dtype=torch.long),
        }


def create_cached_deco_ssl_datasets(
    cache_root: str,
    cache_manifest_path: str | None = None,
    **_: Any,
):
    datasets = {
        split: CachedDecoSSLDataset(
            cache_root,
            split,
            manifest_path=cache_manifest_path,
        )
        for split in ("train", "val", "test")
    }
    datasets["train"].test_dataset = datasets["test"]
    return datasets["train"], datasets["val"]

from __future__ import annotations

import json
import heapq
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


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

    @property
    def group_ids(self) -> np.ndarray:
        """Episode identifier for every cached, already-contiguous window."""
        return self._open_arrays()["group_id"]

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


class EpisodeDiverseBatchSampler(Sampler[list[int]]):
    """Build distributed batches with at most one window per DECO episode.

    Cached samples already contain one contiguous three-frame window. This
    sampler only changes which window starts are selected together: it shuffles
    starts independently inside every episode, then constructs each global DDP
    step from distinct episodes. Apart from the incomplete epoch tail, every
    cached window is visited exactly once per epoch.
    """

    handles_distributed = True

    def __init__(
        self,
        dataset: CachedDecoSSLDataset | Sampler[int],
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not drop_last:
            raise ValueError(
                "EpisodeDiverseBatchSampler requires drop_last=true so every "
                "DDP rank receives an equally sized episode-diverse batch"
            )
        if (num_replicas is None) != (rank is None):
            raise ValueError("num_replicas and rank must be provided together")
        if num_replicas is not None:
            if num_replicas <= 0:
                raise ValueError("num_replicas must be positive")
            if rank < 0 or rank >= num_replicas:
                raise ValueError("rank must lie in [0, num_replicas)")

        # Lightning Fabric reconstructs custom batch samplers while wrapping a
        # DataLoader and passes the loader's ordinary sampler as the first
        # positional argument.  Recover the underlying dataset in that case;
        # our sampler remains responsible for DDP sharding itself.
        if not hasattr(dataset, "group_ids") and hasattr(dataset, "data_source"):
            dataset = dataset.data_source
        if not hasattr(dataset, "group_ids"):
            raise TypeError(
                "EpisodeDiverseBatchSampler requires a dataset with group_ids "
                "or a sampler whose data_source provides group_ids"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = True
        self.seed = int(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        group_ids = np.asarray(dataset.group_ids)
        if group_ids.ndim != 1 or len(group_ids) != len(dataset):
            raise ValueError("dataset.group_ids must be one-dimensional and match dataset length")
        order = np.argsort(group_ids, kind="stable")
        _, starts, counts = np.unique(
            group_ids[order], return_index=True, return_counts=True
        )
        self._episode_indices = [
            order[start : start + count].astype(np.int64, copy=True)
            for start, count in zip(starts.tolist(), counts.tolist())
        ]

    def _distributed_context(self) -> tuple[int, int]:
        if self.num_replicas is not None:
            return int(self.num_replicas), int(self.rank)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def __len__(self) -> int:
        world_size, _ = self._distributed_context()
        return len(self.dataset) // (self.batch_size * world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        world_size, rank = self._distributed_context()
        global_batch_size = self.batch_size * world_size
        num_batches = len(self.dataset) // global_batch_size
        if num_batches == 0:
            raise ValueError(
                f"Dataset has {len(self.dataset)} windows, fewer than one global "
                f"batch of {global_batch_size}"
            )

        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        pools = [rng.permutation(indices) for indices in self._episode_indices]
        counts = np.asarray([len(pool) for pool in pools], dtype=np.int64)

        # Remove only the incomplete global tail. Dropping from the currently
        # longest episodes improves scheduling feasibility without reweighting
        # any complete batch.
        tail = len(self.dataset) - num_batches * global_batch_size
        for _ in range(tail):
            largest = np.flatnonzero(counts == counts.max())
            counts[int(rng.choice(largest))] -= 1

        active_groups = int(np.count_nonzero(counts))
        if active_groups < global_batch_size or int(counts.max()) > num_batches:
            raise ValueError(
                "Cannot form episode-diverse distributed batches: need at least "
                f"{global_batch_size} active episodes and no episode may contain "
                f"more than {num_batches} retained windows; got {active_groups} "
                f"episodes and maximum {int(counts.max())} windows"
            )

        positions = np.zeros(len(pools), dtype=np.int64)
        heap = [
            (-int(count), float(rng.random()), group)
            for group, count in enumerate(counts)
            if count
        ]
        heapq.heapify(heap)

        for _ in range(num_batches):
            if len(heap) < global_batch_size:
                raise RuntimeError("Episode-diverse scheduling became infeasible")
            selected = [heapq.heappop(heap) for _ in range(global_batch_size)]
            global_batch: list[int] = []
            for negative_remaining, _, group in selected:
                position = int(positions[group])
                global_batch.append(int(pools[group][position]))
                positions[group] += 1
                remaining = -negative_remaining - 1
                if remaining:
                    heapq.heappush(
                        heap, (-remaining, float(rng.random()), group)
                    )

            rng.shuffle(global_batch)
            start = rank * self.batch_size
            yield global_batch[start : start + self.batch_size]


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

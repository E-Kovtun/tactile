from __future__ import annotations

import bisect
import csv
import glob
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from tactile_ssl.evaluation.ids import stable_int64_id


SOCK_CLASSES = (
    "downstairs",
    "jump",
    "lean_left",
    "lean_right",
    "stand",
    "stand_toes",
    "upstairs",
    "walk",
    "walk_fast",
)


@dataclass(frozen=True)
class SockGeometry:
    left_flat_indices: np.ndarray
    right_flat_indices: np.ndarray
    node_group_id: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(len(self.left_flat_indices) + len(self.right_flat_indices))

    @classmethod
    def from_csv(cls, sensor_map_path: str, edge_path: str) -> "SockGeometry":
        rows = list(csv.DictReader(open(sensor_map_path, newline="")))
        rows.sort(key=lambda row: int(row["global_node_id"]))
        global_ids = [int(row["global_node_id"]) for row in rows]
        if global_ids != list(range(len(rows))):
            raise ValueError("Sock sensor map must have contiguous global_node_id values")

        left = np.asarray(
            [int(row["flat_index"]) for row in rows if row["side"] == "left"],
            dtype=np.int64,
        )
        right = np.asarray(
            [int(row["flat_index"]) for row in rows if row["side"] == "right"],
            dtype=np.int64,
        )
        node_group_id = torch.tensor(
            [0 if row["side"] == "left" else 1 for row in rows],
            dtype=torch.long,
        )

        edge_rows = list(csv.DictReader(open(edge_path, newline="")))
        edge_index = torch.tensor(
            [
                [int(row["src_global_node_id"]) for row in edge_rows],
                [int(row["dst_global_node_id"]) for row in edge_rows],
            ],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [float(row["visual_distance"]) for row in edge_rows],
            dtype=torch.float32,
        )
        if left.size != 237 or right.size != 216 or len(rows) != 453:
            raise ValueError(
                "Expected the official 237-left/216-right physical sensor map; "
                f"got {left.size}/{right.size}"
            )
        return cls(left, right, node_group_id, edge_index, edge_attr)

    def select_physical(self, left_grid: np.ndarray, right_grid: np.ndarray) -> np.ndarray:
        left = left_grid.reshape(left_grid.shape[0], -1)[:, self.left_flat_indices]
        right = right_grid.reshape(right_grid.shape[0], -1)[:, self.right_flat_indices]
        return np.concatenate([left, right], axis=1)[..., None].astype(np.float32, copy=False)

    def graph_dict(self) -> dict[str, torch.Tensor]:
        return {
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "edge_count": torch.tensor(self.edge_index.shape[1], dtype=torch.long),
            "node_group_id": self.node_group_id,
        }


def _nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Map every target timestamp to the closest source timestamp."""
    source = np.asarray(source_timestamps, dtype=np.int64)
    target = np.asarray(target_timestamps, dtype=np.int64)
    positions = np.searchsorted(source, target, side="left")
    positions = np.clip(positions, 0, len(source) - 1)
    previous = np.clip(positions - 1, 0, len(source) - 1)
    choose_previous = np.abs(target - source[previous]) <= np.abs(source[positions] - target)
    return np.where(choose_previous, previous, positions).astype(np.int64)


def _read_h5_frames(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary (possibly repeated) frame indices through h5py safely."""
    unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


@dataclass(frozen=True)
class _HDFRecording:
    label: int
    name: str
    left_path: str
    right_path: str
    right_to_left: np.ndarray
    left_count: int
    right_count: int
    first_right_index: int


class SockHDFWindowDataset(Dataset):
    """Timestamp-aligned windows from the raw sock classification recordings."""

    def __init__(
        self,
        root: str,
        geometry: SockGeometry,
        window_size: int,
        stride: int,
        classes: Sequence[str] = SOCK_CLASSES,
        entries: Optional[Sequence[tuple[int, int]]] = None,
        include_labels: bool = False,
        include_graph: bool = True,
        include_grids: bool = False,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.root = str(root)
        self.geometry = geometry
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.classes = tuple(classes)
        self.include_labels = bool(include_labels)
        self.include_graph = bool(include_graph)
        self.include_grids = bool(include_grids)
        self.sensor_mean: Optional[np.ndarray] = None
        self.sensor_std: Optional[np.ndarray] = None
        self.recordings = self._discover_recordings()
        self.entries = list(entries) if entries is not None else self._build_entries()
        self._handles: dict[str, h5py.File] = {}

    def _discover_recordings(self) -> list[_HDFRecording]:
        recordings: list[_HDFRecording] = []
        for label, class_name in enumerate(self.classes):
            pattern = str(Path(self.root) / class_name / "*" / "touch_0.hdf5")
            # The release code visits rounds in numeric order (1, 2, ..., 24).
            # A plain lexical sort would place round 10 before round 2 and would
            # therefore change the official chronological tail split.
            def natural_key(path: str):
                return [
                    int(token) if token.isdigit() else token
                    for token in re.split(r"(\d+)", path)
                ]

            for left_path in sorted(glob.glob(pattern), key=natural_key):
                right_path = str(Path(left_path).with_name("touch_1.hdf5"))
                if not Path(right_path).exists():
                    raise FileNotFoundError(right_path)
                with h5py.File(left_path, "r") as left_file, h5py.File(right_path, "r") as right_file:
                    left_count = int(left_file["frame_count"][0])
                    right_count = int(right_file["frame_count"][0])
                    left_ts = left_file["ts"][:left_count]
                    right_ts = right_file["ts"][:right_count]
                recordings.append(
                    _HDFRecording(
                        label=label,
                        name=f"{class_name}/{Path(left_path).parent.name}",
                        left_path=left_path,
                        right_path=right_path,
                        # The released synchronizer chooses the first left
                        # timestamp at or after each right timestamp. It skips
                        # right frames preceding the first left timestamp and
                        # stops once the left stream is exhausted.
                        right_to_left=np.searchsorted(
                            left_ts, right_ts, side="left"
                        ).astype(np.int64),
                        left_count=left_count,
                        right_count=right_count,
                        first_right_index=int(
                            np.searchsorted(right_ts, left_ts[0], side="left")
                        ),
                    )
                )
        if not recordings:
            raise FileNotFoundError(f"No paired sock recordings found below {self.root}")
        return recordings

    def _build_entries(self) -> list[tuple[int, int]]:
        entries = []
        for recording_id, recording in enumerate(self.recordings):
            for right_start in range(
                recording.first_right_index,
                recording.right_count,
                self.stride,
            ):
                right_stop = right_start + self.window_size
                if right_stop > recording.right_count:
                    break
                left_indices = recording.right_to_left[right_start:right_stop]
                if (
                    left_indices.size != self.window_size
                    or int(left_indices[-1]) >= recording.left_count
                ):
                    break
                entries.append((recording_id, right_start))
        return entries

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _file(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        recording_id, start = self.entries[index]
        recording = self.recordings[recording_id]
        right_indices = np.arange(start, start + self.window_size, dtype=np.int64)
        # Align every right-foot frame independently. The streams contain
        # occasional duplicate/dropped frames, so advancing the left index by
        # one after aligning only the window start introduces temporal drift.
        left_indices = recording.right_to_left[right_indices]
        if int(left_indices[-1]) >= recording.left_count:
            raise IndexError("Sock window maps beyond the left recording")
        left_grid = _read_h5_frames(
            self._file(recording.left_path)["pressure"], left_indices
        )
        right_grid = _read_h5_frames(
            self._file(recording.right_path)["pressure"], right_indices
        )
        sensor_values = self.geometry.select_physical(left_grid, right_grid)
        if self.sensor_mean is not None and self.sensor_std is not None:
            sensor_values = (sensor_values - self.sensor_mean) / self.sensor_std
        sensor = torch.from_numpy(sensor_values)
        group_id = stable_int64_id("sock-recording", recording.name)
        sample = {
            "sensor": sensor,
            "group_id": torch.tensor(group_id, dtype=torch.long),
            "sample_id": torch.tensor(
                stable_int64_id("sock-window", recording.name, start, self.window_size),
                dtype=torch.long,
            ),
        }
        if self.include_labels:
            sample["object_classification"] = torch.tensor(recording.label, dtype=torch.long)
        if self.include_grids:
            left_grid = left_grid.astype(np.float32, copy=False)
            right_grid = right_grid.astype(np.float32, copy=False)
            if self.sensor_mean is not None and self.sensor_std is not None:
                left_grid = (left_grid - self.sensor_mean) / self.sensor_std
                right_grid = (right_grid - self.sensor_mean) / self.sensor_std
            sample["left_grid"] = torch.from_numpy(left_grid)
            sample["right_grid"] = torch.from_numpy(right_grid)
        if self.include_graph:
            sample["graph"] = self.geometry.graph_dict()
        return sample


class SockPoseDataset(Dataset):
    """Official tactile2pose split with configurable temporal target alignment.

    The downloaded ``*.p`` files are trusted artifacts from the official
    SensTextile release. They contain ``[left, right, pose]`` NumPy arrays.
    """

    def __init__(
        self,
        split_path: str,
        geometry: SockGeometry,
        window_size: int,
        target_index: int,
        stride: int = 1,
        include_pose: bool = True,
        include_grids: bool = False,
        include_graph: bool = False,
        drop_root_orientation: bool = True,
        centered: bool = False,
        centered_non_overlapping: bool = False,
        arrays: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        if not 0 <= target_index < window_size:
            raise ValueError("target_index must lie inside the temporal window")
        self.split_path = str(split_path)
        self.geometry = geometry
        self.window_size = int(window_size)
        self.target_index = int(target_index)
        self.stride = int(stride)
        self.include_pose = bool(include_pose)
        self.include_grids = bool(include_grids)
        self.include_graph = bool(include_graph)
        self.centered = bool(centered)
        self.centered_non_overlapping = bool(centered_non_overlapping)
        self.sensor_mean: Optional[np.ndarray] = None
        self.sensor_std: Optional[np.ndarray] = None
        if arrays is None:
            with open(split_path, "rb") as stream:
                arrays = pickle.load(stream)
        if len(arrays) != 3:
            raise ValueError("Pose pickle must contain [left_grid, right_grid, pose]")
        self.left = np.asarray(arrays[0])
        self.right = np.asarray(arrays[1])
        self.pose = np.asarray(arrays[2], dtype=np.float32)
        common_length = min(len(self.left), len(self.right), len(self.pose))
        self.left = self.left[:common_length]
        self.right = self.right[:common_length]
        self.pose = self.pose[:common_length]
        if drop_root_orientation and self.pose.shape[-1] == 72:
            self.pose = self.pose[:, 3:]
        if self.pose.shape[-1] != 69:
            raise ValueError(f"Expected 69 pose values after preprocessing, got {self.pose.shape[-1]}")
        if self.centered:
            if self.window_size % 2 or self.target_index != self.window_size // 2:
                raise ValueError("Centered pose windows require target_index=window_size/2")
            if self.centered_non_overlapping:
                if self.stride != self.window_size:
                    raise ValueError(
                        "Centered non-overlap requires stride equal to window_size"
                    )
                # Use each full window exactly once and predict its center.
                # Boundary-clamped windows are deliberately excluded because
                # they would overlap the first/last regular window.
                radius = self.window_size // 2
                self.starts = np.arange(
                    radius,
                    common_length - radius + 1,
                    self.stride,
                    dtype=np.int64,
                )
            else:
                # The released loader returns one example per target frame. Near
                # either boundary it reuses the first/last full temporal window.
                self.starts = np.arange(0, common_length, self.stride, dtype=np.int64)
        elif self.centered_non_overlapping:
            raise ValueError("centered_non_overlapping requires centered=True")
        else:
            self.starts = np.arange(
                0,
                common_length - self.window_size + 1,
                self.stride,
                dtype=np.int64,
            )
        self.target_mean = np.nanmean(self.pose, axis=0).astype(np.float32)
        self.target_std = np.nanstd(self.pose, axis=0).astype(np.float32)
        self.target_std = np.maximum(self.target_std, 1e-6)
        self.pose = np.nan_to_num(self.pose, copy=False)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        position = int(self.starts[index])
        if self.centered:
            target_frame = position
            radius = self.window_size // 2
            start = max(0, target_frame - radius)
            stop = min(len(self.left), target_frame + radius)
            if start == 0:
                stop = self.window_size
            elif stop == len(self.left):
                start = len(self.left) - self.window_size
        else:
            start = position
            stop = start + self.window_size
            target_frame = start + self.target_index
        left_grid = self.left[start:stop]
        right_grid = self.right[start:stop]
        sensor_values = self.geometry.select_physical(left_grid, right_grid)
        if self.sensor_mean is not None and self.sensor_std is not None:
            sensor_values = (sensor_values - self.sensor_mean) / self.sensor_std
        sample = {
            "sensor": torch.from_numpy(sensor_values),
            "group_id": torch.tensor(
                stable_int64_id("sock-pose-split", Path(self.split_path).name), dtype=torch.long
            ),
            "sample_id": torch.tensor(
                stable_int64_id(
                    "sock-pose-window",
                    Path(self.split_path).name,
                    target_frame,
                    self.window_size,
                ),
                dtype=torch.long,
            ),
        }
        if self.include_pose:
            sample["relative_object_pose"] = torch.from_numpy(
                self.pose[target_frame].copy()
            ).unsqueeze(0)
        if self.include_grids:
            sample["left_grid"] = torch.from_numpy(left_grid.astype(np.float32, copy=False))
            sample["right_grid"] = torch.from_numpy(right_grid.astype(np.float32, copy=False))
        if self.include_graph:
            sample["graph"] = self.geometry.graph_dict()
        return sample


class SockCombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset)]
        if not self.datasets:
            raise ValueError("SockCombinedDataset needs at least one non-empty dataset")
        lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_sizes = np.cumsum(lengths).tolist()

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        dataset_id = bisect.bisect_right(self.cumulative_sizes, index)
        previous = 0 if dataset_id == 0 else self.cumulative_sizes[dataset_id - 1]
        return self.datasets[dataset_id][index - previous]


class SockTemperatureMixtureDataset(SockCombinedDataset):
    """Repeat smaller sources according to temperature-scaled source sizes.

    Every source is represented at least once per logical epoch. With
    ``temperature=1`` this is the natural size-proportional mixture; values
    below one give smaller sources more weight without discarding examples
    from the larger source.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        source_names: Sequence[str],
        temperature: float,
    ) -> None:
        if not 0.0 < temperature <= 1.0:
            raise ValueError("temperature must lie in (0, 1]")
        if len(datasets) != len(source_names):
            raise ValueError("datasets and source_names must have the same length")
        self.datasets = [dataset for dataset in datasets if len(dataset)]
        self.source_names = [
            name for dataset, name in zip(datasets, source_names) if len(dataset)
        ]
        if not self.datasets:
            raise ValueError("SockTemperatureMixtureDataset needs a non-empty source")

        lengths = np.asarray([len(dataset) for dataset in self.datasets], dtype=np.int64)
        weights = lengths.astype(np.float64) ** float(temperature)
        weights /= weights.sum()
        epoch_size = max(
            int(math.ceil(length / weight))
            for length, weight in zip(lengths, weights)
        )
        sampled_counts = np.maximum(lengths, np.ceil(epoch_size * weights).astype(np.int64))
        self.source_lengths = lengths.tolist()
        self.source_weights = weights.tolist()
        self.sampled_counts = sampled_counts.tolist()
        self.cumulative_sizes = np.cumsum(sampled_counts).tolist()

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        dataset_id = bisect.bisect_right(self.cumulative_sizes, index)
        previous = 0 if dataset_id == 0 else self.cumulative_sizes[dataset_id - 1]
        local_index = (index - previous) % len(self.datasets[dataset_id])
        return self.datasets[dataset_id][local_index]


def _estimate_input_stats(dataset: Dataset, max_samples: int) -> tuple[float, float]:
    sample_count = min(len(dataset), int(max_samples))
    if sample_count <= 0:
        raise ValueError("Cannot estimate normalization from an empty dataset")
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    total = 0.0
    total_sq = 0.0
    count = 0
    for index in indices:
        values = dataset[int(index)]["sensor"].double()
        total += values.sum().item()
        total_sq += values.square().sum().item()
        count += values.numel()
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))


def _attach_input_stats(dataset: Dataset, stats: tuple[float, float]) -> None:
    dataset.input_mean = np.asarray([stats[0]], dtype=np.float32)
    dataset.input_std = np.asarray([stats[1]], dtype=np.float32)


def _set_sensor_normalization(dataset: Dataset, stats: tuple[float, float]) -> None:
    dataset.sensor_mean = np.asarray([stats[0]], dtype=np.float32)
    dataset.sensor_std = np.asarray([max(stats[1], 1e-6)], dtype=np.float32)


def _geometry(root: str, sensor_map_path: Optional[str], edge_path: Optional[str]) -> SockGeometry:
    root_path = Path(root)
    return SockGeometry.from_csv(
        sensor_map_path or str(root_path / "sock_physical_sensor_map.csv"),
        edge_path or str(root_path / "sock_physical_graph_4neighbor_edges.csv"),
    )


def _action_recording_split(
    base: SockHDFWindowDataset,
    split_seed: int,
    val_recording_ratio: float,
    test_recording_ratio: float,
    max_train_per_class: int,
) -> tuple[list[int], list[int], list[int], dict[str, set[int]]]:
    """Reproduce the canonical action split, including its RNG consumption."""
    if not 0.0 < val_recording_ratio < 1.0 or not 0.0 < test_recording_ratio < 1.0:
        raise ValueError("Recording split ratios must lie within (0, 1)")
    if val_recording_ratio + test_recording_ratio >= 1.0:
        raise ValueError("Validation and test recording ratios must sum to less than one")
    rng = np.random.default_rng(split_seed)
    recordings_by_class: dict[int, list[int]] = {
        label: [] for label in range(len(SOCK_CLASSES))
    }
    for recording_id, recording in enumerate(base.recordings):
        recordings_by_class[recording.label].append(recording_id)
    entries_by_recording: dict[int, list[int]] = {
        recording_id: [] for recording_id in range(len(base.recordings))
    }
    for entry_id, (recording_id, _) in enumerate(base.entries):
        entries_by_recording[recording_id].append(entry_id)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    recording_splits = {"train": set(), "val": set(), "test": set()}
    for label in range(len(SOCK_CLASSES)):
        recording_ids = np.asarray(recordings_by_class[label], dtype=np.int64)
        rng.shuffle(recording_ids)
        if len(recording_ids) < 3:
            raise ValueError(
                f"Class {SOCK_CLASSES[label]} needs at least three recordings for leakage-free splits"
            )
        test_count = max(1, int(round(len(recording_ids) * test_recording_ratio)))
        val_count = max(1, int(round(len(recording_ids) * val_recording_ratio)))
        if test_count + val_count >= len(recording_ids):
            test_count, val_count = 1, 1
        test_recordings = recording_ids[:test_count]
        val_recordings = recording_ids[test_count : test_count + val_count]
        train_recordings = recording_ids[test_count + val_count :]
        recording_splits["test"].update(map(int, test_recordings))
        recording_splits["val"].update(map(int, val_recordings))
        recording_splits["train"].update(map(int, train_recordings))

        class_train = [
            entry
            for recording_id in train_recordings
            for entry in entries_by_recording[int(recording_id)]
        ]
        # This shuffle is intentionally part of the canonical split RNG stream.
        # Keeping it here preserves the recordings selected by the original code
        # for every class after the first one.
        rng.shuffle(class_train)
        train_indices.extend(class_train[:max_train_per_class])
        val_indices.extend(
            entry
            for recording_id in val_recordings
            for entry in entries_by_recording[int(recording_id)]
        )
        test_indices.extend(
            entry
            for recording_id in test_recordings
            for entry in entries_by_recording[int(recording_id)]
        )
    return train_indices, val_indices, test_indices, recording_splits


def _action_aaai_window_split(
    base: SockHDFWindowDataset,
    split_seed: int,
    val_per_class: int,
    test_per_class: int,
    train_per_class: int,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the released SensTextile action-classification split.

    The upstream loader preserves recording/time order, reserves the final
    ``val + test`` windows of every class, and samples the earlier train prefix
    *with replacement* to a fixed count.  Keeping that exact order is crucial:
    randomly scattering stride-2 windows across splits creates far more raw-
    frame overlap than the released protocol.
    """
    if min(val_per_class, test_per_class, train_per_class) <= 0:
        raise ValueError("AAAI per-class split sizes must be positive")

    # RandomState matches the legacy ``np.random.choice`` stream used by the
    # released code once its otherwise implicit seed is made reproducible.
    rng = np.random.RandomState(split_seed)
    entries_by_class: dict[int, list[int]] = {
        label: [] for label in range(len(SOCK_CLASSES))
    }
    for entry_id, (recording_id, _) in enumerate(base.entries):
        entries_by_class[base.recordings[recording_id].label].append(entry_id)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    held_out = val_per_class + test_per_class
    for label in range(len(SOCK_CLASSES)):
        class_indices = np.asarray(entries_by_class[label], dtype=np.int64)
        if len(class_indices) <= held_out:
            raise ValueError(
                f"Class {SOCK_CLASSES[label]} has {len(class_indices)} windows; "
                f"needs more than {held_out} for the AAAI split"
            )
        train_stop = len(class_indices) - held_out
        remaining = class_indices[:train_stop]
        sampled_train = rng.choice(remaining, size=train_per_class, replace=True)
        train_indices.extend(sampled_train.tolist())
        val_indices.extend(
            class_indices[train_stop : train_stop + val_per_class].tolist()
        )
        test_indices.extend(class_indices[-test_per_class:].tolist())

    return train_indices, val_indices, test_indices


def _action_official_source_entries(
    dataset: SockHDFWindowDataset,
    split_reference: SockHDFWindowDataset,
    split: str,
    val_per_class: int = 500,
    test_per_class: int = 1000,
) -> list[tuple[int, int]]:
    """Select SSL clips inside an official action split's ordered time range.

    The official supervised protocol is defined on 45-frame/stride-2 windows.
    SSL uses shorter clips, so copying supervised window IDs is impossible.
    Instead, retain every shorter clip whose aligned start lies within the same
    per-class ordered train/validation/test range. This keeps the downstream
    validation and test tails entirely out of action pretraining.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("classification_window_split must be train, val, or test")
    if dataset.classes != split_reference.classes:
        raise ValueError("Dataset and split reference must use the same classes")

    reference_by_class: dict[int, list[tuple[int, int]]] = {
        label: [] for label in range(len(SOCK_CLASSES))
    }
    for entry in split_reference.entries:
        reference_by_class[split_reference.recordings[entry[0]].label].append(entry)

    ranges: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    held_out = val_per_class + test_per_class
    for label, entries in reference_by_class.items():
        if len(entries) <= held_out:
            raise ValueError(
                f"Class {SOCK_CLASSES[label]} has {len(entries)} official windows; "
                f"needs more than {held_out}"
            )
        train_stop = len(entries) - held_out
        if split == "train":
            selected = entries[:train_stop]
        elif split == "val":
            selected = entries[train_stop : train_stop + val_per_class]
        else:
            selected = entries[-test_per_class:]
        ranges[label] = (selected[0], selected[-1])

    selected_entries = []
    for entry in dataset.entries:
        label = dataset.recordings[entry[0]].label
        first, last = ranges[label]
        if first <= entry <= last:
            selected_entries.append(entry)
    return selected_entries


def create_sock_pretrain_datasets(
    root: str,
    window_size: int = 5,
    stride: int = 5,
    include_classification: bool = True,
    classification_recording_split: str = "all",
    classification_window_split: Optional[str] = None,
    classification_split_seed: int = 42,
    classification_val_recording_ratio: float = 0.15,
    classification_test_recording_ratio: float = 0.15,
    pose_splits: Sequence[str] = ("train", "val", "test"),
    validation_samples: int = 512,
    normalization_samples: int = 4096,
    source_normalization: bool = False,
    sampling_temperature: Optional[float] = None,
    action_normalization_mean: Optional[float] = None,
    action_normalization_std: Optional[float] = None,
    pose_normalization_mean: Optional[float] = None,
    pose_normalization_std: Optional[float] = None,
    sensor_map_path: Optional[str] = None,
    edge_path: Optional[str] = None,
) -> tuple[Dataset, Dataset]:
    geometry = _geometry(root, sensor_map_path, edge_path)
    datasets: list[Dataset] = []
    source_names: list[str] = []
    if include_classification:
        if classification_recording_split not in {"all", "train", "val", "test"}:
            raise ValueError(
                "classification_recording_split must be all, train, val, or test"
            )
        if classification_window_split not in {None, "train", "val", "test"}:
            raise ValueError(
                "classification_window_split must be null, train, val, or test"
            )
        if classification_window_split is not None and classification_recording_split != "all":
            raise ValueError(
                "classification_window_split cannot be combined with a recording split"
            )
        classification_dataset = SockHDFWindowDataset(
            root=str(Path(root) / "data_classification" / "sock_classification"),
            geometry=geometry,
            window_size=window_size,
            stride=stride,
            # The release also contains one unlabeled `test` recording. Keep it
            # only in the transductive/full-data setting.
            classes=(
                (*SOCK_CLASSES, "test")
                if classification_recording_split == "all" and classification_window_split is None
                else SOCK_CLASSES
            ),
            include_labels=False,
            include_graph=True,
        )
        if classification_window_split is not None:
            split_reference = SockHDFWindowDataset(
                root=str(Path(root) / "data_classification" / "sock_classification"),
                geometry=geometry,
                window_size=45,
                stride=2,
                classes=SOCK_CLASSES,
                include_labels=False,
                include_graph=False,
            )
            classification_dataset.entries = _action_official_source_entries(
                classification_dataset,
                split_reference,
                classification_window_split,
            )
            classification_dataset.official_window_split = classification_window_split
        elif classification_recording_split != "all":
            split_reference = SockHDFWindowDataset(
                root=str(Path(root) / "data_classification" / "sock_classification"),
                geometry=geometry,
                window_size=45,
                stride=2,
                classes=SOCK_CLASSES,
                include_labels=False,
                include_graph=False,
            )
            _, _, _, recording_splits = _action_recording_split(
                split_reference,
                split_seed=classification_split_seed,
                val_recording_ratio=classification_val_recording_ratio,
                test_recording_ratio=classification_test_recording_ratio,
                max_train_per_class=4000,
            )
            selected_recordings = recording_splits[classification_recording_split]
            classification_dataset.entries = [
                entry
                for entry in classification_dataset.entries
                if entry[0] in selected_recordings
            ]
            classification_dataset.recording_ids = selected_recordings
        datasets.append(classification_dataset)
        source_names.append("action")
    pose_datasets: list[SockPoseDataset] = []
    for split in pose_splits:
        pose_dataset = SockPoseDataset(
                split_path=str(Path(root) / "tactile2pose" / "dataset" / f"{split}.p"),
                geometry=geometry,
                window_size=window_size,
                target_index=window_size - 1,
                stride=stride,
                include_pose=False,
                include_graph=True,
            )
        datasets.append(pose_dataset)
        pose_datasets.append(pose_dataset)
        source_names.append("pose")

    source_stats: dict[str, dict[str, float]] = {}
    if source_normalization:
        if include_classification:
            if (action_normalization_mean is None) != (action_normalization_std is None):
                raise ValueError("action normalization mean and std must be provided together")
            action_stats = (
                (float(action_normalization_mean), float(action_normalization_std))
                if action_normalization_mean is not None
                else _estimate_input_stats(classification_dataset, normalization_samples)
            )
            _set_sensor_normalization(classification_dataset, action_stats)
            source_stats["action"] = {"mean": action_stats[0], "std": action_stats[1]}
        if pose_datasets:
            pose_reference = next(
                (
                    dataset
                    for split, dataset in zip(pose_splits, pose_datasets)
                    if split == "train"
                ),
                pose_datasets[0],
            )
            if (pose_normalization_mean is None) != (pose_normalization_std is None):
                raise ValueError("pose normalization mean and std must be provided together")
            pose_stats = (
                (float(pose_normalization_mean), float(pose_normalization_std))
                if pose_normalization_mean is not None
                else _estimate_input_stats(pose_reference, normalization_samples)
            )
            for dataset in pose_datasets:
                _set_sensor_normalization(dataset, pose_stats)
            source_stats["pose"] = {"mean": pose_stats[0], "std": pose_stats[1]}

    if sampling_temperature is not None and len(datasets) > 1:
        train_dataset = SockTemperatureMixtureDataset(
            datasets,
            source_names=source_names,
            temperature=sampling_temperature,
        )
    else:
        train_dataset = SockCombinedDataset(datasets)
    val_indices = np.linspace(
        0,
        len(train_dataset) - 1,
        min(int(validation_samples), len(train_dataset)),
        dtype=np.int64,
    ).tolist()
    val_dataset = Subset(train_dataset, val_indices)
    stats = (0.0, 1.0) if source_normalization else _estimate_input_stats(
        train_dataset, normalization_samples
    )
    _attach_input_stats(train_dataset, stats)
    _attach_input_stats(val_dataset, stats)
    train_dataset.source_normalization_stats = source_stats
    val_dataset.source_normalization_stats = source_stats
    return train_dataset, val_dataset


def create_sock_action_datasets(
    root: str,
    window_size: int = 45,
    stride: int = 2,
    split_seed: int = 0,
    val_recording_ratio: float = 0.15,
    test_recording_ratio: float = 0.15,
    max_train_per_class: int = 4000,
    split_protocol: str = "recording_disjoint",
    aaai_val_per_class: int = 500,
    aaai_test_per_class: int = 1000,
    aaai_train_per_class: int = 4000,
    include_grids: bool = False,
    normalization_samples: int = 4096,
    normalize_inputs: bool = False,
    normalization_mean: Optional[float] = None,
    normalization_std: Optional[float] = None,
    sensor_map_path: Optional[str] = None,
    edge_path: Optional[str] = None,
) -> tuple[Dataset, Dataset, Dataset]:
    geometry = _geometry(root, sensor_map_path, edge_path)
    base = SockHDFWindowDataset(
        root=str(Path(root) / "data_classification" / "sock_classification"),
        geometry=geometry,
        window_size=window_size,
        stride=stride,
        include_labels=True,
        include_graph=False,
        include_grids=include_grids,
    )
    if split_protocol == "recording_disjoint":
        train_indices, val_indices, test_indices, recording_splits = _action_recording_split(
            base,
            split_seed=split_seed,
            val_recording_ratio=val_recording_ratio,
            test_recording_ratio=test_recording_ratio,
            max_train_per_class=max_train_per_class,
        )
    elif split_protocol == "aaai_window":
        train_indices, val_indices, test_indices = _action_aaai_window_split(
            base,
            split_seed=split_seed,
            val_per_class=aaai_val_per_class,
            test_per_class=aaai_test_per_class,
            train_per_class=aaai_train_per_class,
        )
        recording_splits = None
    else:
        raise ValueError(
            "split_protocol must be 'recording_disjoint' or 'aaai_window'"
        )
    train_dataset = Subset(base, train_indices)
    val_dataset = Subset(base, val_indices)
    test_dataset = Subset(base, test_indices)
    counts = np.bincount(
        [base.recordings[base.entries[index][0]].label for index in train_indices],
        minlength=len(SOCK_CLASSES),
    ).astype(np.float64)
    class_weights = (1.0 / counts)
    class_weights /= class_weights.sum()
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("normalization mean and std must be provided together")
    stats = (
        (float(normalization_mean), float(normalization_std))
        if normalization_mean is not None
        else _estimate_input_stats(train_dataset, normalization_samples)
    )
    if normalize_inputs:
        _set_sensor_normalization(base, stats)
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.classes = SOCK_CLASSES
        dataset.class_weights = class_weights.astype(np.float32)
        _attach_input_stats(dataset, (0.0, 1.0) if normalize_inputs else stats)
        dataset.source_normalization_stats = {
            "action": {"mean": stats[0], "std": stats[1]}
        }
        split_name = (
            "train"
            if dataset is train_dataset
            else "val"
            if dataset is val_dataset
            else "test"
        )
        dataset.split_protocol = split_protocol
        if recording_splits is not None:
            dataset.recording_ids = recording_splits[split_name]
    if recording_splits is not None:
        if not (
            train_dataset.recording_ids.isdisjoint(val_dataset.recording_ids)
            and train_dataset.recording_ids.isdisjoint(test_dataset.recording_ids)
            and val_dataset.recording_ids.isdisjoint(test_dataset.recording_ids)
        ):
            raise RuntimeError("Sock action recording split leakage detected")
    return train_dataset, val_dataset, test_dataset


def create_sock_pose_datasets(
    root: str,
    mode: str,
    stride: int = 1,
    centered_non_overlapping: bool = False,
    normalization_samples: int = 4096,
    normalize_inputs: bool = False,
    normalization_mean: Optional[float] = None,
    normalization_std: Optional[float] = None,
    sensor_map_path: Optional[str] = None,
    edge_path: Optional[str] = None,
) -> tuple[SockPoseDataset, SockPoseDataset, SockPoseDataset]:
    geometry = _geometry(root, sensor_map_path, edge_path)
    if mode == "jepa":
        window_size, target_index, include_grids, centered = 5, 4, False, False
    elif mode == "jepa_official":
        # Use exactly the same centered 60-frame inputs and target frame as the
        # released supervised pose baseline.
        window_size, target_index, include_grids, centered = 60, 30, False, True
    elif mode == "official_cnn":
        # Official --window=30 selects [t-30, t+30), so t is token 30.
        window_size, target_index, include_grids, centered = 60, 30, True, True
    else:
        raise ValueError("mode must be 'jepa', 'jepa_official', or 'official_cnn'")
    datasets = tuple(
        SockPoseDataset(
            split_path=str(Path(root) / "tactile2pose" / "dataset" / f"{split}.p"),
            geometry=geometry,
            window_size=window_size,
            target_index=target_index,
            stride=stride,
            include_pose=True,
            include_grids=include_grids,
            include_graph=False,
            centered=centered,
            centered_non_overlapping=centered_non_overlapping,
        )
        for split in ("train", "val", "test")
    )
    train = datasets[0]
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("normalization mean and std must be provided together")
    stats = (
        (float(normalization_mean), float(normalization_std))
        if normalization_mean is not None
        else _estimate_input_stats(train, normalization_samples)
    )
    if normalize_inputs:
        for dataset in datasets:
            _set_sensor_normalization(dataset, stats)
    for dataset in datasets:
        dataset.target_mean = train.target_mean
        dataset.target_std = train.target_std
        _attach_input_stats(dataset, (0.0, 1.0) if normalize_inputs else stats)
        dataset.source_normalization_stats = {
            "pose": {"mean": stats[0], "std": stats[1]}
        }
    return datasets

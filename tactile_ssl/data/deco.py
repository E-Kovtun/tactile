from __future__ import annotations

import bisect
import io
import json
import pickle
import re
import tarfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset

from tactile_ssl.data.deco_geometry import DecoGeometry, build_deco_geometry
from tactile_ssl.evaluation.ids import stable_int64_id


DECO_TASK4_VARIANTS = tuple(f"data-t4-{index}" for index in range(1, 7))
DECO_LEFT_SCALE = 3486.0
DECO_RIGHT_SCALE = 4050.0
_TACTILE_RE = re.compile(
    r"^(?P<prefix>episode_\d+)/tactiles/(?P<frame>\d+)_left_ee_tactile\.npy$"
)


@dataclass(frozen=True)
class DecoEpisode:
    archive: str
    variant: str
    episode_id: str
    prefix: str
    frame_count: int
    condition: int

    @property
    def name(self) -> str:
        return f"{self.variant}/{self.episode_id}"


def _archive_priority(path: Path) -> int:
    return 0 if path.suffix == ".tar" else 1


def discover_deco_episodes(root: str | Path) -> list[DecoEpisode]:
    """Inspect Task-4 archives without extracting or unpickling their metadata."""
    root = Path(root).expanduser().resolve()
    candidates: dict[tuple[str, str], Path] = {}
    for variant in DECO_TASK4_VARIANTS:
        variant_root = root / variant
        for pattern in ("episode_*.tar", "episode_*.tar.gz"):
            for path in variant_root.glob(pattern):
                if path.name.endswith((".partial", ".ready")):
                    continue
                episode_id = path.name.split(".tar", 1)[0]
                key = (variant, episode_id)
                previous = candidates.get(key)
                if previous is None or _archive_priority(path) < _archive_priority(previous):
                    candidates[key] = path

    episodes: list[DecoEpisode] = []
    for (variant, episode_id), path in sorted(candidates.items()):
        frames: list[int] = []
        prefix = episode_id
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                match = _TACTILE_RE.match(member.name)
                if match:
                    frames.append(int(match.group("frame")))
                    prefix = match.group("prefix")
        if not frames:
            raise ValueError(f"No left tactile frames found in {path}")
        expected = list(range(max(frames) + 1))
        if sorted(frames) != expected:
            raise ValueError(f"Non-contiguous tactile frames in {path}")
        condition = DECO_TASK4_VARIANTS.index(variant)
        episodes.append(
            DecoEpisode(
                archive=str(path.relative_to(root)),
                variant=variant,
                episode_id=episode_id,
                prefix=prefix,
                frame_count=len(frames),
                condition=condition,
            )
        )
    if not episodes:
        raise FileNotFoundError(f"No DECO Task-4 episode archives found below {root}")
    return episodes


def save_deco_manifest(
    root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(root).expanduser().resolve()
    output = Path(output_path) if output_path else root / "task4-manifest.json"
    payload = {
        "format_version": 1,
        "root": str(root),
        "episodes": [asdict(episode) for episode in discover_deco_episodes(root)],
    }
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def load_deco_manifest(
    root: str | Path,
    manifest_path: str | Path | None = None,
) -> list[DecoEpisode]:
    root = Path(root).expanduser().resolve()
    path = Path(manifest_path) if manifest_path else root / "task4-manifest.json"
    if not path.exists():
        return discover_deco_episodes(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported DECO manifest format in {path}")
    return [DecoEpisode(**item) for item in payload["episodes"]]


def split_deco_episodes(
    episodes: Sequence[DecoEpisode],
    *,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[DecoEpisode]]:
    """Split whole episodes, independently inside every Task-4 variant."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum below one")
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for variant in DECO_TASK4_VARIANTS:
        group = [episode for episode in episodes if episode.variant == variant]
        if not group:
            continue
        indices = rng.permutation(len(group))
        n_test = int(round(len(group) * test_ratio))
        n_val = int(round(len(group) * val_ratio))
        if len(group) >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, max(0, len(group) - 2))
        n_val = min(n_val, max(0, len(group) - n_test - 1))
        splits["test"].extend(group[index] for index in indices[:n_test])
        splits["val"].extend(group[index] for index in indices[n_test : n_test + n_val])
        splits["train"].extend(group[index] for index in indices[n_test + n_val :])
    return splits


class DecoArchiveStore:
    """Per-worker LRU cache for tar handles and trusted DECO dataframes."""

    def __init__(self, root: str | Path, max_open_archives: int = 8) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_open_archives = int(max_open_archives)
        self._archives: OrderedDict[str, tarfile.TarFile] = OrderedDict()
        self._tables: OrderedDict[str, object] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = OrderedDict()
        state["_tables"] = OrderedDict()
        return state

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()
        self._tables.clear()

    def _archive(self, episode: DecoEpisode) -> tarfile.TarFile:
        key = episode.archive
        archive = self._archives.pop(key, None)
        if archive is None:
            archive = tarfile.open(self.root / key, "r:*")
        self._archives[key] = archive
        while len(self._archives) > self.max_open_archives:
            _, old = self._archives.popitem(last=False)
            old.close()
        return archive

    def bytes(self, episode: DecoEpisode, relative_name: str) -> bytes:
        member_name = f"{episode.prefix}/{relative_name}"
        stream = self._archive(episode).extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(f"{member_name} not found in {episode.archive}")
        return stream.read()

    def tactile(self, episode: DecoEpisode, frame: int, hand: str) -> np.ndarray:
        payload = self.bytes(
            episode, f"tactiles/{frame:06d}_{hand}_ee_tactile.npy"
        )
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if array.shape != (1062,):
            raise ValueError(
                f"Expected 1062 tactile values in {episode.name} frame {frame}, "
                f"got {array.shape}"
            )
        return array.astype(np.float32, copy=False)

    def image(self, episode: DecoEpisode, frame: int, camera: int) -> Image.Image:
        payload = self.bytes(episode, f"colors/{frame:06d}_color_{camera}.jpg")
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def table(self, episode: DecoEpisode):
        key = episode.archive
        table = self._tables.pop(key, None)
        if table is None:
            # DECO-50 is a trusted project artifact; this intentionally mirrors
            # the public loader's pandas.read_pickle behavior.
            table = pickle.loads(self.bytes(episode, "data.pkl"))
        self._tables[key] = table
        while len(self._tables) > self.max_open_archives:
            self._tables.popitem(last=False)
        return table


def _normalize_tactile(
    left: np.ndarray,
    right: np.ndarray,
    mode: str,
    left_scale: float,
    right_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "deco_max":
        return left / left_scale, right / right_scale
    if mode == "none":
        return left, right
    raise ValueError(f"Unsupported DECO tactile normalization mode {mode!r}")


class DecoSSLWindowDataset(Dataset):
    """Three-frame Task-4 windows in the shared ``[T, N, C]`` contract."""

    def __init__(
        self,
        root: str | Path,
        episodes: Sequence[DecoEpisode],
        window_size: int = 3,
        stride: int = 1,
        normalization: str = "deco_max",
        left_scale: float = DECO_LEFT_SCALE,
        right_scale: float = DECO_RIGHT_SCALE,
        include_graph: bool = True,
        geometry: Optional[DecoGeometry] = None,
        max_open_archives: int = 8,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.root = str(Path(root).expanduser().resolve())
        self.episodes = list(episodes)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.normalization = str(normalization)
        self.left_scale = float(left_scale)
        self.right_scale = float(right_scale)
        self.include_graph = bool(include_graph)
        self.geometry = geometry or build_deco_geometry()
        self.store = DecoArchiveStore(self.root, max_open_archives)
        counts = [max(0, (episode.frame_count - self.window_size) // self.stride + 1) for episode in self.episodes]
        self.cumulative_sizes = np.cumsum(counts).tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["store"] = DecoArchiveStore(self.root, self.store.max_open_archives)
        return state

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _location(self, index: int) -> tuple[DecoEpisode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.cumulative_sizes, index)
        previous = 0 if episode_index == 0 else self.cumulative_sizes[episode_index - 1]
        return self.episodes[episode_index], (index - previous) * self.stride

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, start = self._location(index)
        left = np.stack(
            [self.store.tactile(episode, frame, "left") for frame in range(start, start + self.window_size)]
        )
        right = np.stack(
            [self.store.tactile(episode, frame, "right") for frame in range(start, start + self.window_size)]
        )
        left, right = _normalize_tactile(
            left, right, self.normalization, self.left_scale, self.right_scale
        )
        grouped = self.geometry.group_hands(left, right).astype(np.float32, copy=False)
        sample = {
            "sensor": torch.from_numpy(grouped),
            "group_id": torch.tensor(stable_int64_id("deco-episode", episode.name), dtype=torch.long),
            "sample_id": torch.tensor(
                stable_int64_id("deco-window", episode.name, start, self.window_size), dtype=torch.long
            ),
        }
        if self.include_graph:
            sample["graph"] = self.geometry.graph_dict()
        return sample


def _image_tensor(
    image: Image.Image, size: int = 256, normalization: str = "deco"
) -> torch.Tensor:
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    padded = ImageOps.pad(resized, (size, size), color=(128, 128, 128), centering=(0.5, 0.5))
    array = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.0
    statistics = {
        "deco": ([0.3574, 0.3694, 0.3745], [0.2222, 0.2093, 0.1849]),
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "unit": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    }
    if normalization not in statistics:
        raise ValueError(
            f"Unknown image normalization {normalization!r}; expected one of {tuple(statistics)}"
        )
    mean_values, std_values = statistics[normalization]
    mean = np.asarray(mean_values, dtype=np.float32)[:, None, None]
    std = np.asarray(std_values, dtype=np.float32)[:, None, None]
    return torch.from_numpy((array - mean) / std)


class DecoPolicyDataset(DecoSSLWindowDataset):
    """Task-4 policy samples with two cameras and configurable future actions."""

    def __init__(
        self,
        *args,
        action_chunk_size: int = 32,
        image_size: int = 256,
        image_normalization: str = "deco",
        observation_mean: Optional[Sequence[float]] = None,
        observation_std: Optional[Sequence[float]] = None,
        action_mean: Optional[Sequence[float]] = None,
        action_std: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_chunk_size = int(action_chunk_size)
        self.image_size = int(image_size)
        self.image_normalization = str(image_normalization)
        self.observation_mean = None if observation_mean is None else np.asarray(observation_mean, dtype=np.float32)
        self.observation_std = None if observation_std is None else np.maximum(np.asarray(observation_std, dtype=np.float32), 1e-8)
        self.action_mean = None if action_mean is None else np.asarray(action_mean, dtype=np.float32)
        self.action_std = None if action_std is None else np.maximum(np.asarray(action_std, dtype=np.float32), 1e-8)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        episode, start = self._location(index)
        current = start + self.window_size - 1
        table = self.store.table(episode)
        row = table.iloc[current]
        proprio = np.asarray(row["left_obs"] + row["right_obs"] + row["head_obs"], dtype=np.float32)
        stop = min(len(table), current + self.action_chunk_size)
        future = table.iloc[current:stop]
        actions = np.concatenate(
            [
                np.stack(future["left_action"].values),
                np.stack(future["right_action"].values),
                np.stack(future["head_action"].values),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        valid = len(actions)
        if valid < self.action_chunk_size:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.action_chunk_size - valid, axis=0)], axis=0
            )
        if self.observation_mean is not None:
            proprio = (proprio - self.observation_mean) / self.observation_std
        if self.action_mean is not None:
            actions = (actions - self.action_mean[None]) / self.action_std[None]
        images = torch.stack(
            [
                _image_tensor(
                    self.store.image(episode, current, camera),
                    self.image_size,
                    self.image_normalization,
                )
                for camera in (0, 1)
            ]
        )
        sample.update(
            {
                "images": images,
                "proprio": torch.from_numpy(proprio),
                "action": torch.from_numpy(actions),
                "action_valid_mask": torch.arange(self.action_chunk_size) < valid,
                "condition": torch.tensor(episode.condition, dtype=torch.long),
            }
        )
        return sample


def _create_deco_datasets(
    dataset_class,
    *,
    root: str,
    manifest_path: str | None = None,
    split_seed: int = 42,
    val_episode_ratio: float = 0.15,
    test_episode_ratio: float = 0.15,
    **dataset_kwargs,
):
    episodes = load_deco_manifest(root, manifest_path)
    splits = split_deco_episodes(
        episodes, seed=split_seed, val_ratio=val_episode_ratio, test_ratio=test_episode_ratio
    )
    datasets = {
        name: dataset_class(root=root, episodes=split, **dataset_kwargs)
        for name, split in splits.items()
    }
    datasets["train"].test_dataset = datasets["test"]
    return datasets["train"], datasets["val"]


def create_deco_ssl_datasets(**kwargs):
    return _create_deco_datasets(DecoSSLWindowDataset, **kwargs)


def create_deco_policy_datasets(**kwargs):
    policy_stats_path = kwargs.pop("policy_stats_path", None)
    validation_max_samples = kwargs.pop("validation_max_samples", None)
    root = kwargs.pop("root")
    manifest_path = kwargs.pop("manifest_path", None)
    split_seed = kwargs.pop("split_seed", 42)
    val_ratio = kwargs.pop("val_episode_ratio", 0.15)
    test_ratio = kwargs.pop("test_episode_ratio", 0.15)
    episodes = load_deco_manifest(root, manifest_path)
    splits = split_deco_episodes(
        episodes, seed=split_seed, val_ratio=val_ratio, test_ratio=test_ratio
    )
    stats = compute_deco_policy_stats(root, splits["train"], policy_stats_path)
    datasets = {
        name: DecoPolicyDataset(root=root, episodes=split, **stats, **kwargs)
        for name, split in splits.items()
    }
    datasets["train"].test_dataset = datasets["test"]
    validation = datasets["val"]
    if validation_max_samples is not None and len(validation) > validation_max_samples:
        validation_max_samples = int(validation_max_samples)
        if validation_max_samples <= 0:
            raise ValueError("validation_max_samples must be positive")
        sample_ids = torch.arange(validation_max_samples, dtype=torch.int64)
        indices = (
            (2 * sample_ids + 1) * len(validation) // (2 * validation_max_samples)
        ).tolist()
        validation = Subset(validation, indices)
    return datasets["train"], validation


def compute_deco_policy_stats(
    root: str | Path,
    episodes: Sequence[DecoEpisode],
    output_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """Compute observation/action z-score statistics from training episodes only."""
    path = Path(output_path) if output_path else None
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: payload[key] for key in ("observation_mean", "observation_std", "action_mean", "action_std")}

    store = DecoArchiveStore(root)
    obs_sum = np.zeros(28, dtype=np.float64)
    obs_square_sum = np.zeros(28, dtype=np.float64)
    action_sum = np.zeros(28, dtype=np.float64)
    action_square_sum = np.zeros(28, dtype=np.float64)
    count = 0
    try:
        for episode in episodes:
            table = store.table(episode)
            obs = np.concatenate(
                [
                    np.stack(table["left_obs"].values),
                    np.stack(table["right_obs"].values),
                    np.stack(table["head_obs"].values),
                ],
                axis=1,
            ).astype(np.float64, copy=False)
            actions = np.concatenate(
                [
                    np.stack(table["left_action"].values),
                    np.stack(table["right_action"].values),
                    np.stack(table["head_action"].values),
                ],
                axis=1,
            ).astype(np.float64, copy=False)
            obs_sum += obs.sum(axis=0)
            obs_square_sum += np.square(obs).sum(axis=0)
            action_sum += actions.sum(axis=0)
            action_square_sum += np.square(actions).sum(axis=0)
            count += len(table)
    finally:
        store.close()
    if count == 0:
        raise ValueError("Cannot compute DECO policy statistics from an empty training split")
    obs_mean = obs_sum / count
    action_mean = action_sum / count
    obs_std = np.sqrt(np.maximum(obs_square_sum / count - np.square(obs_mean), 1e-16))
    action_std = np.sqrt(np.maximum(action_square_sum / count - np.square(action_mean), 1e-16))
    result = {
        "observation_mean": obs_mean.astype(np.float32).tolist(),
        "observation_std": obs_std.astype(np.float32).tolist(),
        "action_mean": action_mean.astype(np.float32).tolist(),
        "action_std": action_std.astype(np.float32).tolist(),
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps({"format_version": 1, **result}, indent=2), encoding="utf-8")
        temporary.replace(path)
    return result

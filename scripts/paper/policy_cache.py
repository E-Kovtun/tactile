#!/usr/bin/env python3
"""Build resumable DECO Task-4 policy shards for trainable ResNet18 heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset

from tactile_ssl.data.deco import create_deco_policy_datasets
from tactile_ssl.data.deco_policy_cache import CACHE_FORMAT_VERSION, CACHE_TENSOR_KEYS


SPLITS = ("train", "val", "test")


def create_source_datasets(args: argparse.Namespace):
    train, val = create_deco_policy_datasets(
        root=str(args.source_root),
        manifest_path=str(args.source_manifest),
        policy_stats_path=str(args.policy_stats),
        window_size=3,
        stride=3,
        action_chunk_size=16,
        image_size=256,
        image_normalization="unit",
        split_seed=42,
        val_episode_ratio=0.15,
        test_episode_ratio=0.15,
        normalization="deco_max",
        left_scale=3486.0,
        right_scale=4050.0,
        include_graph=False,
    )
    return {"train": train, "val": val, "test": train.test_dataset}


def effective_length(dataset, max_samples: int | None) -> int:
    return len(dataset) if max_samples is None else min(len(dataset), max_samples)


def shard_plan(datasets, shard_size: int, max_samples: int | None) -> dict:
    splits = {}
    global_index = 0
    for split in SPLITS:
        length = effective_length(datasets[split], max_samples)
        shards = []
        for start in range(0, length, shard_size):
            stop = min(length, start + shard_size)
            shard_id = len(shards)
            shards.append(
                {
                    "id": shard_id,
                    "global_id": global_index,
                    "start": start,
                    "stop": stop,
                    "file": f"{split}/shard-{shard_id:05d}.pt",
                }
            )
            global_index += 1
        splits[split] = {"length": length, "shards": shards}
    return splits


def tensor_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(
    dataset,
    shard: dict,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    rank: int,
) -> None:
    output = output_root / shard["file"]
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = Subset(dataset, range(int(shard["start"]), int(shard["stop"])))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    accumulated = {key: [] for key in CACHE_TENSOR_KEYS}
    with torch.inference_mode():
        for batch in loader:
            images = batch.pop("images")
            # The source dataset returns [0,1] RGB tensors. Quantization is
            # lossless with respect to the resized uint8 PIL image and cuts the
            # cache footprint in half compared with BF16 image tensors.
            accumulated["images"].append(
                images.mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
            )
            for key in CACHE_TENSOR_KEYS:
                if key != "images":
                    accumulated[key].append(batch[key].cpu())
    payload = {key: torch.cat(parts, dim=0).contiguous() for key, parts in accumulated.items()}
    expected = int(shard["stop"]) - int(shard["start"])
    if any(tensor.shape[0] != expected for tensor in payload.values()):
        raise RuntimeError(f"Wrong tensor length while building {output}")
    temporary = output.with_suffix(output.suffix + f".partial-rank{rank}")
    torch.save(payload, temporary)
    temporary.replace(output)


def iter_assigned_shards(plan: dict, rank: int, world_size: int) -> Iterable[tuple[str, dict]]:
    for split in SPLITS:
        for shard in plan[split]["shards"]:
            if int(shard["global_id"]) % world_size == rank:
                yield split, shard


def build(args: argparse.Namespace) -> None:
    datasets = create_source_datasets(args)
    plan = shard_plan(datasets, args.shard_size, args.max_samples_per_split)
    assigned = list(iter_assigned_shards(plan, args.rank, args.world_size))
    for index, (split, shard) in enumerate(assigned, 1):
        print(
            f"rank={args.rank} shard={index}/{len(assigned)} split={split} "
            f"range=[{shard['start']},{shard['stop']})",
            flush=True,
        )
        write_shard(
            datasets[split], shard, args.output_root,
            args.batch_size, args.num_workers, args.rank,
        )


def finalize(args: argparse.Namespace) -> None:
    datasets = create_source_datasets(args)
    plan = shard_plan(datasets, args.shard_size, args.max_samples_per_split)
    missing = []
    for split in SPLITS:
        for shard in plan[split]["shards"]:
            path = args.output_root / shard["file"]
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            expected = int(shard["stop"]) - int(shard["start"])
            for key in CACHE_TENSOR_KEYS:
                if key not in payload or payload[key].shape[0] != expected:
                    raise RuntimeError(f"Invalid tensor {key} in {path}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cache shards; first: {missing[0]}")
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "source_root": str(args.source_root),
        "source_manifest": str(args.source_manifest),
        "policy_stats": str(args.policy_stats),
        "split_seed": 42,
        "window_size": 3,
        "stride": 3,
        "action_chunk_size": 16,
        "image_size": 256,
        "image_storage": "uint8-rgb-256",
        "model_image_normalization": "imagenet",
        "image_backbone": "resnet18-trained-from-scratch-in-policy",
        "shard_size": args.shard_size,
        "max_samples_per_split": args.max_samples_per_split,
        "splits": plan,
    }
    output = args.output_root / "manifest.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "finalize"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--policy-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int)
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.source_manifest = args.source_manifest.expanduser().resolve()
    args.policy_stats = args.policy_stats.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("rank must be in [0, world_size)")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "build":
        build(arguments)
    else:
        finalize(arguments)

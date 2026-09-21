#!/usr/bin/env python3
"""Cache frozen ImageNet ResNet18 GAP features for the DECO policy image cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from tactile_ssl.data.deco_policy_cache_final import VISION_CACHE_FORMAT_VERSION


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 2 or payload.get("image_storage") != "uint8-rgb-256":
        raise ValueError(f"Expected the DECO uint8 policy cache manifest, got {path}")
    return payload


def assigned_shards(manifest: dict, rank: int, world_size: int):
    for split in ("train", "val", "test"):
        for shard in manifest["splits"][split]["shards"]:
            if int(shard["global_id"]) % world_size == rank:
                yield split, shard


def make_backbone(device: torch.device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return torch.nn.Sequential(*list(model.children())[:-2]).eval().requires_grad_(False).to(device)


def encode_shard(
    source: Path,
    output: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    rank: int,
) -> None:
    if output.is_file():
        return
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    images = payload["images"]
    if images.dtype != torch.uint8 or images.shape[1:3] != (2, 3):
        raise ValueError(f"Unexpected images {tuple(images.shape)} {images.dtype} in {source}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    parts = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(device, non_blocking=True).float().div_(255.0)
            batch = (batch - mean) / std
            features = backbone(batch.flatten(0, 1)).mean(dim=(-2, -1))
            parts.append(features.reshape(batch.shape[0], 2, 512).half().cpu())
    result = {
        "image_features": torch.cat(parts, dim=0).contiguous(),
        "sample_id": payload["sample_id"].clone(),
        "group_id": payload["group_id"].clone(),
    }
    if result["image_features"].shape != (images.shape[0], 2, 512):
        raise RuntimeError(f"Wrong output shape for {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial-rank{rank}")
    torch.save(result, temporary)
    temporary.replace(output)


def build(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.source_manifest)
    device = torch.device(args.device)
    backbone = make_backbone(device)
    shards = list(assigned_shards(manifest, args.rank, args.world_size))
    for index, (split, shard) in enumerate(shards, 1):
        relative = Path(shard["file"])
        print(
            f"rank={args.rank} shard={index}/{len(shards)} split={split} file={relative}",
            flush=True,
        )
        encode_shard(
            args.source_root / relative,
            args.output_root / relative,
            backbone,
            device,
            args.batch_size,
            args.rank,
        )


def finalize(args: argparse.Namespace) -> None:
    source = load_manifest(args.source_manifest)
    missing = []
    for split in ("train", "val", "test"):
        for shard in source["splits"][split]["shards"]:
            path = args.output_root / shard["file"]
            if not path.is_file():
                missing.append(path)
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            expected = int(shard["stop"]) - int(shard["start"])
            if payload.get("image_features", torch.empty(0)).shape != (expected, 2, 512):
                raise RuntimeError(f"Invalid image_features in {path}")
            if payload["image_features"].dtype != torch.float16:
                raise RuntimeError(f"Expected FP16 image_features in {path}")
            for key in ("sample_id", "group_id"):
                if payload.get(key, torch.empty(0)).shape[0] != expected:
                    raise RuntimeError(f"Invalid {key} in {path}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} feature shards; first: {missing[0]}")
    manifest = {
        "format_version": VISION_CACHE_FORMAT_VERSION,
        "feature_storage": "resnet18-imagenet1k-v1-gap-fp16",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": ResNet18_Weights.IMAGENET1K_V1.url,
        "source_root": str(args.source_root),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "splits": source["splits"],
    }
    output = args.output_root / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "finalize"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.source_manifest = args.source_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("rank must be in [0, world_size)")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments) if arguments.mode == "build" else finalize(arguments)

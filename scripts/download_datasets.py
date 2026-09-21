#!/usr/bin/env python3
"""Download the paper datasets and arrange them for the default configs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile


DATASETS = ("xela", "socks", "deco")
SPARSH_REPO = "facebook/sparsh-skin-dataset"
DECO_REPO = "BAAI-Humanoid/DECO-50"
SOCKS_URL = "https://www.dropbox.com/sh/g70n60jfutzd0l5/AACnOgtLUG8tHbU8TLn5MBFba?dl=1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        dest="datasets",
        help="Dataset to download; repeat to select two. Defaults to all three.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Destination root (default: datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace conflicting files while arranging downloaded archives.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without changing files.",
    )
    return parser.parse_args(argv)


def selected_datasets(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DATASETS
    return tuple(dict.fromkeys(values))


def _snapshot_download(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; install the repository environment first."
        ) from error

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


def _merge_tree(source: Path, destination: Path, force: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Expected directory is missing from the release: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_file() and target.stat().st_size == path.stat().st_size:
                path.unlink()
                continue
            if not force:
                raise FileExistsError(
                    f"Conflicting file: {target}. Remove it or rerun with --force."
                )
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(path), str(target))


def download_xela(root: Path, force: bool) -> None:
    destination = root / "sparsh-skin"
    staging = root / ".downloads" / "sparsh-skin"
    print(f"Downloading Sparsh-skin to {destination}", flush=True)
    _snapshot_download(
        SPARSH_REPO,
        staging,
        ["pretraining/**", "downstream_tasks/**"],
    )
    _merge_tree(
        staging / "pretraining",
        destination / "xela" / "pretraining" / "extracted",
        force,
    )
    _merge_tree(staging / "downstream_tasks", destination / "downstream_tasks", force)
    shutil.rmtree(staging)
    required = [
        destination / "xela/pretraining/extracted/baseline/xela/data.pkl",
        destination / "xela/pretraining/extracted/urdf/ahrcpcpn.urdf",
        destination / "downstream_tasks/force_estimation",
        destination / "downstream_tasks/relative_pose_estimation",
    ]
    _require_paths("Sparsh-skin", required)


def _download_file(url: str, destination: Path) -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            "requests and tqdm are required; install the repository environment first."
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        length = int(response.headers.get("content-length", 0))
        total = offset + length if length else None
        with partial.open(mode) as output, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename}")
        bundle.extractall(destination)


def _find_dataset_dir(root: Path, name: str, required: Path) -> Path | None:
    for candidate in [root, *root.rglob(name)]:
        if candidate.is_dir() and candidate.name == name and (candidate / required).exists():
            return candidate
    return None


def arrange_socks_archive(archive: Path, destination: Path, force: bool) -> None:
    staging = archive.parent / "senstextile-extracted"
    if staging.exists():
        shutil.rmtree(staging)
    _extract_zip(archive, staging)

    classification = _find_dataset_dir(
        staging, "data_classification", Path("sock_classification")
    )
    if classification is None:
        nested = next(staging.rglob("data_classification.zip"), None)
        if nested is not None:
            nested_root = staging / "classification-unpacked"
            _extract_zip(nested, nested_root)
            classification = _find_dataset_dir(
                nested_root, "data_classification", Path("sock_classification")
            )
            if classification is None and (nested_root / "sock_classification").is_dir():
                classification = nested_root

    pose = _find_dataset_dir(staging, "tactile2pose", Path("dataset/train.p"))
    if pose is None:
        nested = next(staging.rglob("tactile2pose.zip"), None)
        if nested is not None:
            nested_root = staging / "pose-unpacked"
            _extract_zip(nested, nested_root)
            pose = _find_dataset_dir(nested_root, "tactile2pose", Path("dataset/train.p"))
            if pose is None and (nested_root / "dataset/train.p").is_file():
                pose = nested_root

    if classification is None or pose is None:
        raise RuntimeError(
            "The SensTextile archive layout was not recognized; expected "
            "data_classification/sock_classification and tactile2pose/dataset."
        )
    _merge_tree(classification, destination / "data_classification", force)
    _merge_tree(pose, destination / "tactile2pose", force)
    shutil.rmtree(staging)


def download_socks(root: Path, force: bool) -> None:
    destination = root / "socks"
    required = [
        destination / "data_classification/sock_classification",
        destination / "tactile2pose/dataset/train.p",
        destination / "tactile2pose/dataset/val.p",
        destination / "tactile2pose/dataset/test.p",
    ]
    if all(path.exists() for path in required) and not force:
        print(f"Tactile socks already ready at {destination}", flush=True)
        return
    archive = root / ".downloads" / "senstextile.zip"
    print(f"Downloading Tactile socks to {destination}", flush=True)
    _download_file(SOCKS_URL, archive)
    arrange_socks_archive(archive, destination, force)
    archive.unlink()
    _require_paths("Tactile socks", required)


def download_deco(root: Path, force: bool) -> None:
    del force  # Hugging Face resumes and verifies files in place.
    destination = root / "deco50"
    print(f"Downloading DECO-50 task4 to {destination}", flush=True)
    _snapshot_download(DECO_REPO, destination, ["task4/**"])
    required = [destination / "task4" / f"data-t4-{index}" for index in range(1, 7)]
    _require_paths("DECO-50", required)
    for folder in required:
        if not any(folder.glob("episode_*.tar*")):
            raise RuntimeError(f"DECO-50 directory contains no episode archives: {folder}")


def _require_paths(dataset: str, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{dataset} download is incomplete; missing: {', '.join(missing)}")
    print(f"{dataset} is ready", flush=True)


def _print_plan(root: Path, datasets: tuple[str, ...]) -> None:
    plans = {
        "xela": f"{SPARSH_REPO} -> {root / 'sparsh-skin'}",
        "socks": f"SensTextile Dropbox archive -> {root / 'socks'}",
        "deco": f"{DECO_REPO} task4 -> {root / 'deco50/task4'}",
    }
    for dataset in datasets:
        print(plans[dataset])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = selected_datasets(args.datasets)
    root = args.root.expanduser().resolve()
    if args.dry_run:
        _print_plan(root, datasets)
        return
    root.mkdir(parents=True, exist_ok=True)
    actions = {
        "xela": download_xela,
        "socks": download_socks,
        "deco": download_deco,
    }
    for dataset in datasets:
        actions[dataset](root, args.force)
    print(f"Selected datasets are ready under {root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

import io
from pathlib import Path
import zipfile
import pytest
import sys
from types import SimpleNamespace

from scripts import download_datasets as downloader
from scripts.download_datasets import arrange_socks_archive, parse_args, selected_datasets


def test_dataset_selection_defaults_to_all_and_removes_duplicates():
    assert selected_datasets(None) == ("xela", "socks", "deco")
    assert selected_datasets(["socks", "deco", "socks"]) == ("socks", "deco")
    args = parse_args(["--dataset", "xela", "--dataset", "deco"])
    assert selected_datasets(args.datasets) == ("xela", "deco")


def test_socks_archive_is_arranged_for_default_paths(tmp_path: Path):
    classification = io.BytesIO()
    with zipfile.ZipFile(classification, "w") as nested:
        nested.writestr(
            "data_classification/sock_classification/walk/walk_1/sample.hdf5", b"data"
        )

    archive = tmp_path / "senstextile.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("/", b"")
        bundle.writestr("data_classification.zip", classification.getvalue())
        for split in ("train", "val", "test"):
            bundle.writestr(f"tactile2pose/dataset/{split}.p", b"data")

    destination = tmp_path / "datasets" / "socks"
    arrange_socks_archive(archive, destination, force=False)

    assert (
        destination
        / "data_classification/sock_classification/walk/walk_1/sample.hdf5"
    ).read_bytes() == b"data"
    assert (destination / "tactile2pose/dataset/train.p").read_bytes() == b"data"


@pytest.mark.parametrize("name", ["../escape", "/escape", "folder/../../escape", "/"])
def test_zip_rejects_unsafe_paths_before_extracting(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe.txt", b"safe")
        bundle.writestr(name, b"not an empty root marker")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="Unsafe path"):
        downloader._extract_zip(archive, destination)
    assert not (destination / "safe.txt").exists()


def test_socks_reuses_archive_after_extraction_failure(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    archive = root / ".downloads/senstextile.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("/", b"")
        bundle.writestr("data_classification/sock_classification/walk/sample.hdf5", b"data")
        for split in ("train", "val", "test"):
            bundle.writestr(f"tactile2pose/dataset/{split}.p", b"data")

    def unexpected_download(*args):
        pytest.fail("Existing archive must not be downloaded again")

    monkeypatch.setattr(downloader, "_download_file", unexpected_download)
    downloader.download_socks(root, force=False)
    assert (root / "socks/tactile2pose/dataset/train.p").is_file()
    assert not archive.exists()


def test_xela_release_is_moved_to_the_configured_layout(tmp_path: Path, monkeypatch):
    def fake_download(repo_id, local_dir, allow_patterns):
        assert repo_id == downloader.SPARSH_REPO
        assert allow_patterns == ["pretraining/**", "downstream_tasks/**"]
        for relative in (
            "pretraining/baseline/xela/data.pkl",
            "pretraining/urdf/ahrcpcpn.urdf",
            "downstream_tasks/force_estimation/sample.pkl",
            "downstream_tasks/relative_pose_estimation/sample.pkl",
        ):
            path = local_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")

    monkeypatch.setattr(downloader, "_snapshot_download", fake_download)
    root = tmp_path / "datasets"
    downloader.download_xela(root, force=False)

    destination = root / "sparsh-skin"
    assert (destination / "xela/pretraining/extracted/baseline/xela/data.pkl").is_file()
    assert (destination / "downstream_tasks/force_estimation/sample.pkl").is_file()


def test_deco_download_is_limited_to_task4(tmp_path: Path, monkeypatch):
    def fake_download(repo_id, local_dir, allow_patterns):
        assert repo_id == downloader.DECO_REPO
        assert allow_patterns == ["task4/**"]
        for index in range(1, 7):
            path = local_dir / "task4" / f"data-t4-{index}" / "episode_0000.tar.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")

    monkeypatch.setattr(downloader, "_snapshot_download", fake_download)
    root = tmp_path / "datasets"
    downloader.download_deco(root, force=False)

    assert (root / "deco50/task4/data-t4-6/episode_0000.tar.gz").is_file()


def test_snapshot_download_passes_dataset_scope_and_patterns(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: calls.append(kwargs)
    ))
    for repo, patterns in ((downloader.SPARSH_REPO, ["pretraining/**", "downstream_tasks/**"]),
                           (downloader.DECO_REPO, ["task4/**"])):
        downloader._snapshot_download(repo, tmp_path, patterns)
        assert calls[-1] == dict(repo_id=repo, repo_type="dataset", local_dir=tmp_path,
                                allow_patterns=patterns)


@pytest.mark.parametrize("dataset", ["xela", "deco"])
def test_cli_dry_run_does_not_download_or_create_root(tmp_path, monkeypatch, dataset):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not download")
    monkeypatch.setattr(downloader, "_snapshot_download", forbidden)
    root = tmp_path / "absent"
    downloader.main(["--dataset", dataset, "--root", str(root), "--dry-run"])
    assert not root.exists()


def test_xela_can_repeat_arrangement(tmp_path, monkeypatch):
    # Exercise the full branch twice; the mock replaces network transfer only.
    test_xela_release_is_moved_to_the_configured_layout(tmp_path, monkeypatch)
    downloader.download_xela(tmp_path / "datasets", force=False)
    assert not (tmp_path / "datasets/.downloads/sparsh-skin").exists()


def test_deco_can_repeat_download(tmp_path, monkeypatch):
    test_deco_download_is_limited_to_task4(tmp_path, monkeypatch)
    downloader.download_deco(tmp_path / "datasets", force=False)


@pytest.mark.parametrize("dataset", ["xela", "deco"])
def test_incomplete_release_fails(tmp_path, monkeypatch, dataset):
    def empty_download(repo_id, local_dir, allow_patterns):
        local_dir.mkdir(parents=True)
    monkeypatch.setattr(downloader, "_snapshot_download", empty_download)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        getattr(downloader, f"download_{dataset}")(tmp_path, force=False)


def test_deco_requires_archives_not_just_folders(tmp_path, monkeypatch):
    def empty_folders(repo_id, local_dir, allow_patterns):
        for index in range(1, 7):
            (local_dir / "task4" / f"data-t4-{index}").mkdir(parents=True)
    monkeypatch.setattr(downloader, "_snapshot_download", empty_folders)
    with pytest.raises(RuntimeError, match="no episode archives"):
        downloader.download_deco(tmp_path, force=False)


def test_merge_preserves_conflicting_file_without_force(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "data.pkl").write_bytes(b"new data")
    (destination / "data.pkl").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        downloader._merge_tree(source, destination, force=False)
    assert (source / "data.pkl").read_bytes() == b"new data"
    assert (destination / "data.pkl").read_bytes() == b"old"

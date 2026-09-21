import io
from pathlib import Path
import zipfile

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

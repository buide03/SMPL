from io import BytesIO
from tarfile import TarInfo, open as open_tar

from colab import hmr2_assets


def test_prepare_checkpoint_extracts_inference_assets_and_reuses_them(tmp_path, monkeypatch):
    selected = {hmr2_assets.CHECKPOINT: b"checkpoint", "data/smpl_mean_params.npz": b"parameters"}
    archive_path = tmp_path / "hmr2_data.tar.gz.part"
    with open_tar(archive_path, "w") as archive:
        for name, data in {**selected, "training/train_data.npz": b"unused"}.items():
            entry = TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, BytesIO(data))
    monkeypatch.setattr(hmr2_assets, "ARCHIVE_SIZE", archive_path.stat().st_size)
    monkeypatch.setattr(hmr2_assets, "MEMBER_SIZES", {name: len(data) for name, data in selected.items()})

    checkpoint = hmr2_assets.prepare_checkpoint(tmp_path)
    assert checkpoint.read_bytes() == selected[hmr2_assets.CHECKPOINT]
    assert (tmp_path / "data" / "smpl_mean_params.npz").read_bytes() == b"parameters"
    assert not (tmp_path / "training").exists()
    assert not archive_path.exists()
    assert hmr2_assets.prepare_checkpoint(tmp_path) == checkpoint

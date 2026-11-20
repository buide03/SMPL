"""Stage the upstream HMR2.0b checkpoint and neutral model for a Colab run."""

from hashlib import file_digest
from pathlib import Path
import shutil
import subprocess
from tarfile import open as open_tar
from tempfile import NamedTemporaryFile


UPSTREAM = "https://github.com/shubham-goel/4D-Humans.git"
REVISION = "efe18deff163b29dff87ddbd575fa29b716a356c"
CHUMPY_REVISION = "580566eafc9ac68b2614b64d6f7aaa84eebb70da"
CHECKPOINT_ARCHIVE = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
ARCHIVE_SIZE = 2_710_589_440
CHECKPOINT = "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
MEMBER_SIZES = {
    CHECKPOINT: 2_709_521_501,
    "logs/train/multiruns/hmr2/0/model_config.yaml": 3_083,
    "data/SMPL_to_J19.pkl": 1_047_441,
    "data/smpl_mean_params.npz": 1_310,
}
SMPL_SOURCE_NAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepare_checkpoint(cache: Path) -> Path:
    """Download or resume the official archive and unpack only inference assets."""
    if all((cache / name).is_file() and (cache / name).stat().st_size == size
           for name, size in MEMBER_SIZES.items()):
        return cache / CHECKPOINT

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "hmr2_data.tar.gz.part"
    if not archive.exists() or archive.stat().st_size < ARCHIVE_SIZE:
        print("Downloading the official 4D-Humans HMR2.0b archive (2.71 GB).", flush=True)
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3",
            "--continue-at", "-", "--output", str(archive), CHECKPOINT_ARCHIVE,
        ], check=True)
    if archive.stat().st_size != ARCHIVE_SIZE:
        raise ValueError(f"The official archive did not match the expected release size: {archive}")

    with open_tar(archive, mode="r:*") as source:
        for name, size in MEMBER_SIZES.items():
            entry = source.getmember(name)
            if not entry.isfile() or entry.size != size:
                raise ValueError(f"Unexpected member in the official archive: {name}")
            destination = cache / name
            if destination.is_file() and destination.stat().st_size == size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                member = source.extractfile(entry)
                if member is None:
                    raise ValueError(f"Could not open the official archive member: {name}")
                with member, staged.open("wb") as output:
                    shutil.copyfileobj(member, output, length=1024 * 1024)
                if staged.stat().st_size != size:
                    raise ValueError(f"Incomplete official archive member: {name}")
                staged.replace(destination)
            finally:
                staged.unlink(missing_ok=True)
    archive.unlink()
    return cache / CHECKPOINT

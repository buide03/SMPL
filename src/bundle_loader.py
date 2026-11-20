"""Read reconstruction archives in memory without extracting or loading pickle."""
from dataclasses import dataclass
from io import BytesIO
import json
import re
import stat
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile
from zlib import error as ZlibError

import numpy as np
from PIL import Image

from src.validation import SHAPES, validate_person

MIB = 1024 ** 2
MAX_UPLOAD = 64 * MIB
MAX_PIXELS = 12_000_000
SYNTHETIC_LABEL = "SYNTHETIC TEST DATA — NOT HMR2 INFERENCE"


class BundleError(ValueError):
    """A reconstruction package could not be validated."""


@dataclass(frozen=True)
class Bundle:
    manifest: dict
    images: dict[str, Image.Image]
    people: dict[str, dict[str, np.ndarray]]


def inspect_archive(archive: ZipFile, limit: int, entry_limit: int) -> set[str]:
    """Apply path and resource limits before opening any member."""
    members = archive.infolist()
    if len(members) > entry_limit or sum(item.file_size for item in members) > limit:
        raise BundleError("Archive exceeds the entry count or uncompressed size limit")
    names = set()
    normalized = set()
    for item in members:
        name = item.orig_filename
        parts = name.rstrip("/").split("/")
        if (len(name) > 240 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or
                                    p in (".", "..") or p.endswith(".") for p in parts)
                or stat.S_ISLNK(item.external_attr >> 16) or name.lower() in normalized):
            raise BundleError(f"Unsafe or duplicate ZIP path: {name!r}")
        if (item.flag_bits & 1 or item.compress_type not in (ZIP_DEFLATED, ZIP_STORED)
                or item.file_size > 2000 * max(1, item.compress_size)):
            raise BundleError(f"Unsupported or suspicious ZIP member: {name!r}")
        normalized.add(name.lower())
        if not item.is_dir():
            names.add(name)
    return names


def read_image(archive: ZipFile, name: str, expected: tuple[int, int] | None) -> Image.Image:
    with Image.open(BytesIO(archive.read(name)), formats=["PNG"]) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, "is_animated", False):
            raise BundleError(f"{name}: image exceeds limits or is animated")
        if expected and image.size != expected:
            raise BundleError(f"{name}: dimensions differ from manifest")
        image.verify()
    with Image.open(BytesIO(archive.read(name)), formats=["PNG"]) as image:
        return image.convert("RGB")


def read_npz(payload: bytes) -> dict[str, np.ndarray]:
    """Bound nested ZIP expansion and check NPY headers before NumPy allocates."""
    with ZipFile(BytesIO(payload)) as archive:
        names = inspect_archive(archive, 2 * MIB, len(SHAPES))
        if names != {f"{name}.npy" for name in SHAPES}:
            raise BundleError("NPZ must contain exactly the required arrays")
        for name, max_shape in SHAPES.items():
            member = f"{name}.npy"
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise BundleError(f"{name}: unsupported NPY header version")
                shape, _, dtype = readers[version](stream)
                kinds = "iu" if name == "faces" else "f"
                shape_ok = shape == max_shape
                if name in ("vertices", "faces"):
                    shape_ok = len(shape) == 2 and 0 < shape[0] <= max_shape[0] and shape[1] == 3
                if not shape_ok or dtype.kind not in kinds or dtype.itemsize > 8:
                    raise BundleError(f"{name}: unsupported NPY shape or dtype")
                if archive.getinfo(member).file_size - stream.tell() != int(np.prod(shape)) * dtype.itemsize:
                    raise BundleError(f"{name}: wrong NPY data length")
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in SHAPES}


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise BundleError("Expected schema_version 1")
    for key in ("scene_id", "frame_id", "camera_id", "model_name", "checkpoint_identifier",
                "smpl_model_version", "coordinate_system", "data_kind"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip() or len(manifest[key]) > 200:
            raise BundleError(f"Manifest requires a nonempty string for {key}")
    kind = manifest["data_kind"]
    coordinate = manifest["coordinate_system"]
    if kind == "synthetic" and coordinate == "synthetic":
        if manifest["smpl_model_version"] != "none":
            raise BundleError("Synthetic meshes must declare smpl_model_version 'none'")
    elif not (kind == "real_inference" and coordinate in ("smpl_local", "hmr2_crop_camera")):
        raise BundleError("Unsupported data_kind or coordinate_system combination")
    for name in ("image_width", "image_height"):
        if type(manifest.get(name)) is not int or not 0 < manifest[name] <= 8192:
            raise BundleError(f"Invalid {name}")
    records = manifest.get("people")
    if not isinstance(records, list) or len(records) > 64:
        raise BundleError("Manifest requires a people list with up to 64 entries")
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("person_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", record["person_id"])
                or not isinstance(record.get("npz_path"), str)
                or not re.fullmatch(r"people/[A-Za-z0-9_-]+\.npz", record["npz_path"])):
            raise BundleError("Invalid person ID or NPZ path")
    for name in ("person_id", "npz_path"):
        if len({record[name] for record in records}) != len(records):
            raise BundleError(f"Duplicate {name} in manifest")


def load_bundle(payload: bytes) -> Bundle:
    """Load one scene/frame/camera and validate all its person records."""
    if len(payload) > MAX_UPLOAD:
        raise BundleError("Upload exceeds 64 MiB")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            names = inspect_archive(archive, 96 * MIB, 72)
            for name in ("manifest.json", "source.png", "overlay.png"):
                if name not in names:
                    raise BundleError(f"Missing required file: {name}")
            if archive.getinfo("manifest.json").file_size > 64 * 1024:
                raise BundleError("Manifest exceeds 64 KiB")
            manifest = json.loads(archive.read("manifest.json"))
            validate_manifest(manifest)
            size = manifest["image_width"], manifest["image_height"]
            images = {name: read_image(archive, name, None if name == "side_view.png" else size)
                      for name in ("source.png", "overlay.png", "side_view.png") if name in names}
            people = {}
            for record in manifest["people"]:
                person_id, path = record["person_id"], record["npz_path"]
                if path not in names or archive.getinfo(path).file_size > 2 * MIB:
                    raise BundleError(f"{person_id}: missing or oversized NPZ {path}")
                try:
                    arrays = read_npz(archive.read(path))
                    validate_person(arrays, *size, manifest["data_kind"] == "synthetic")
                except ValueError as exc:
                    raise BundleError(f"{person_id}: {exc}") from exc
                people[person_id] = arrays
            return Bundle(manifest, images, people)
    except BundleError:
        raise
    except (BadZipFile, OSError, ValueError, EOFError, OverflowError, RecursionError, SyntaxError,
            ZlibError, Image.DecompressionBombError) as exc:
        raise BundleError(f"Invalid reconstruction archive: {exc}") from exc

"""Read reconstruction archives in memory without extracting or loading pickle."""

from dataclasses import dataclass
from io import BytesIO
import json
from math import prod
import re
import stat
from typing import TypedDict, cast
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile
from zlib import error as ZlibError

import numpy as np
from PIL import Image

from src.validation import SHAPES, PersonArrays, validate_array_spec, validate_person

MIB = 1024 ** 2
MAX_UPLOAD = 64 * MIB
MAX_PIXELS = 12_000_000
SYNTHETIC_LABEL = "SYNTHETIC TEST DATA — NOT HMR2 INFERENCE"
WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
# ZIP general-purpose flags for encryption, patched data, and strong encryption.
UNSUPPORTED_ZIP_FLAGS = (1 << 0) | (1 << 5) | (1 << 6)

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class PersonRecord(TypedDict):
    person_id: str
    npz_path: str


class Manifest(TypedDict):
    schema_version: int
    scene_id: str
    frame_id: str
    camera_id: str
    model_name: str
    checkpoint_identifier: str
    smpl_model_version: str
    coordinate_system: str
    data_kind: str
    image_width: int
    image_height: int
    people: list[PersonRecord]


class BundleError(ValueError):
    """A reconstruction package could not be validated."""


@dataclass(frozen=True)
class Bundle:
    manifest: Manifest
    images: dict[str, Image.Image]
    people: dict[str, PersonArrays]


def inspect_archive(archive: ZipFile, limit: int, entry_limit: int) -> set[str]:
    """Apply path and resource limits before opening any member."""
    members = archive.infolist()
    if len(members) > entry_limit or sum(item.file_size for item in members) > limit:
        raise BundleError("Archive exceeds the entry count or uncompressed size limit")
    names: set[str] = set()
    normalized: set[str] = set()
    for item in members:
        name = item.orig_filename
        path = name.removesuffix("/")
        parts = path.split("/")
        mode = stat.S_IFMT(item.external_attr >> 16)
        if (len(name) > 240 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                                   or part.endswith(".") or WINDOWS_DEVICE.match(part) for part in parts)
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (mode == stat.S_IFDIR and not item.is_dir())
                or path.lower() in normalized):
            raise BundleError(f"Unsafe or duplicate ZIP path: {name!r}")
        if (item.flag_bits & UNSUPPORTED_ZIP_FLAGS
                or item.compress_type not in (ZIP_DEFLATED, ZIP_STORED)
                or item.file_size > 2000 * max(1, item.compress_size)):
            raise BundleError(f"Unsupported or suspicious ZIP member: {name!r}")
        normalized.add(path.lower())
        if not item.is_dir():
            names.add(name)
    files = {name.lower() for name in names}
    for name in normalized:
        parent = name.rpartition("/")[0]
        while parent:
            if parent in files:
                raise BundleError(f"Unsafe ZIP path with a file as its parent: {name!r}")
            parent = parent.rpartition("/")[0]
    return names


def read_image(archive: ZipFile, name: str, expected: tuple[int, int] | None) -> Image.Image:
    payload = archive.read(name)
    with Image.open(BytesIO(payload), formats=["PNG"]) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, "is_animated", False):
            raise BundleError(f"{name}: image exceeds limits or is animated")
        if expected and image.size != expected:
            raise BundleError(f"{name}: dimensions differ from manifest")
        image.verify()
    with Image.open(BytesIO(payload), formats=["PNG"]) as image:
        return image.convert("RGB")


def read_npz(payload: bytes, *, synthetic: bool = False) -> PersonArrays:
    """Bound nested ZIP expansion and check NPY headers before NumPy allocates."""
    with ZipFile(BytesIO(payload)) as archive:
        names = inspect_archive(archive, 2 * MIB, len(SHAPES))
        if names != {f"{name}.npy" for name in SHAPES}:
            raise BundleError("NPZ must contain exactly the required arrays")
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        for name in SHAPES:
            member = f"{name}.npy"
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in readers:
                    raise BundleError(f"{name}: unsupported NPY header version")
                shape, _, dtype = readers[version](stream)
                validate_array_spec(name, shape, dtype, synthetic)
                if archive.getinfo(member).file_size - stream.tell() != prod(shape) * dtype.itemsize:
                    raise BundleError(f"{name}: wrong NPY data length")
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        return cast(PersonArrays, {name: arrays[name] for name in SHAPES})


def unique_json_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    obj: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in obj:
            raise BundleError(f"Duplicate manifest key: {key!r}")
        obj[key] = value
    return obj


def validate_manifest(manifest: object) -> Manifest:
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
    if manifest["image_width"] * manifest["image_height"] > MAX_PIXELS:
        raise BundleError("Source dimensions exceed the pixel limit")
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
    return cast(Manifest, manifest)


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
            raw_manifest = archive.read("manifest.json").decode("utf-8-sig")
            data = json.loads(raw_manifest, object_pairs_hook=unique_json_keys)
            # Validate extra metadata as well as the contracted fields.
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            manifest = validate_manifest(data)
            size = manifest["image_width"], manifest["image_height"]
            images = {name: read_image(archive, name, None if name == "side_view.png" else size)
                      for name in ("source.png", "overlay.png", "side_view.png") if name in names}
            people: dict[str, PersonArrays] = {}
            synthetic = manifest["data_kind"] == "synthetic"
            for record in manifest["people"]:
                person_id, path = record["person_id"], record["npz_path"]
                if path not in names or archive.getinfo(path).file_size > 2 * MIB:
                    raise BundleError(f"{person_id}: missing or oversized NPZ {path}")
                try:
                    arrays = read_npz(archive.read(path), synthetic=synthetic)
                    validate_person(arrays, *size, synthetic)
                except (ValueError, BadZipFile, EOFError, ZlibError, NotImplementedError) as exc:
                    raise BundleError(f"{person_id}: {exc}") from exc
                people[person_id] = arrays
            return Bundle(manifest, images, people)
    except BundleError:
        raise
    except (BadZipFile, OSError, ValueError, EOFError, OverflowError, RecursionError, SyntaxError,
            NotImplementedError,
            ZlibError, Image.DecompressionBombError) as exc:
        raise BundleError(f"Invalid reconstruction archive: {exc}") from exc

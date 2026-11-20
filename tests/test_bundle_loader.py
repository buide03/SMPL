from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from src.bundle_loader import BundleError, load_bundle
from src.synthetic_fixture import fixture_bytes


def repack(changes: dict[str, bytes | None]) -> bytes:
    with ZipFile(BytesIO(fixture_bytes())) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries.update(changes)
    out = BytesIO()
    with ZipFile(out, "w") as archive:
        for name, value in entries.items():
            if value is not None:
                archive.writestr(name, value)
    return out.getvalue()


def test_load_two_people():
    bundle = load_bundle(fixture_bytes())
    assert bundle.manifest["data_kind"] == "synthetic"
    assert bundle.images["source.png"].size == (640, 360)
    assert list(bundle.people) == ["person_000", "person_001"]
    assert bundle.people["person_001"]["body_pose"].shape == (23, 3, 3)


@pytest.mark.parametrize(("changes", "message"), [
    ({"manifest.json": None}, "Missing required file"),
    ({"../escape.png": b"x"}, "Unsafe"),
    ({"C:/escape.png": b"x"}, "Unsafe"),
    ({"SOURCE.PNG": b"x"}, "duplicate"),
    ({"people/person_000.npz": None}, "missing or oversized NPZ"),
    ({"people/person_000.npz": b"invalid"}, "Invalid reconstruction archive"),
    ({"source.png": b"invalid"}, "Invalid reconstruction archive"),
])
def test_invalid_archive(changes, message):
    with pytest.raises(BundleError, match=message):
        load_bundle(repack(changes))


def test_npy_object_rejected():
    arrays = load_bundle(fixture_bytes()).people["person_000"]
    arrays["betas"] = np.full(10, object(), dtype=object)
    payload = BytesIO()
    np.savez(payload, **arrays)
    with pytest.raises(BundleError, match="betas.*dtype"):
        load_bundle(repack({"people/person_000.npz": payload.getvalue()}))


def test_metadata_size_mismatch():
    bundle = load_bundle(fixture_bytes())
    bundle.manifest["image_width"] = 1
    with pytest.raises(BundleError, match="dimensions differ"):
        load_bundle(repack({"manifest.json": json.dumps(bundle.manifest).encode()}))


def test_optional_side_view():
    assert "side_view.png" not in load_bundle(repack({"side_view.png": None})).images


def test_real_schema_and_coordinate_frame():
    bundle = load_bundle(fixture_bytes())
    manifest = bundle.manifest
    manifest.update(data_kind="real_inference", coordinate_system="smpl_local",
                    smpl_model_version="test-only", people=manifest["people"][:1])
    arrays = bundle.people["person_000"]
    arrays["vertices"] = np.resize(arrays["vertices"], (6890, 3))
    arrays["faces"] = np.resize(arrays["faces"], (13776, 3))
    payload = BytesIO()
    np.savez_compressed(payload, **arrays)
    changes = {"manifest.json": json.dumps(manifest).encode(),
               "people/person_000.npz": payload.getvalue()}
    assert load_bundle(repack(changes)).people["person_000"]["vertices"].shape == (6890, 3)
    manifest["coordinate_system"] = "world"
    changes["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(BundleError, match="coordinate_system"):
        load_bundle(repack(changes))

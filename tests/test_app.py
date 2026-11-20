from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app import draw_detections
from src.bundle_loader import load_bundle
from src.synthetic_fixture import fixture_bytes

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_viewer_modes_and_person_selection():
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert "SYNTHETIC TEST DATA" in app.warning[0].value
    assert len(app.get("image")) == 3
    assert len(app.get("plotly_chart")) == 1
    assert len(app.dataframe[0].value) == 10
    app.selectbox(key="pose-joint").select("Left knee").run()
    assert app.dataframe[1].value.shape == (3, 3)
    for mode, count in (("Original RGB", 0), ("3D Human Mesh", 1),
                        ("Mesh Wireframe", 1), ("All Detected People", 2)):
        app.radio[0].set_value(mode).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == count
    next(box for box in app.selectbox if box.label == "Person").select("person_001").run()
    assert not app.exception
    assert any("person_001" in caption.value for caption in app.caption)


def test_zip_upload_and_validation_error():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.get("file_uploader")[0].upload("fixture.zip", fixture_bytes()).run()
    assert not app.exception
    assert app.success
    app.get("file_uploader")[0].upload("invalid.zip", b"invalid").run()
    assert not app.exception
    assert "Invalid reconstruction archive" in app.error[0].value
    assert not app.get("plotly_chart")


def test_subpixel_box_at_image_boundary():
    bundle = load_bundle(fixture_bytes())
    bundle.people["person_000"]["bbox_xyxy"] = np.array([639.5, 359.5, 640., 360.])
    labeled = draw_detections(bundle, "person_000")
    assert labeled.getpixel((639, 359)) == (120, 249, 215)


def test_smpl_editor_keeps_synthetic_fixture_in_the_viewer():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.radio[0].set_value("SMPL Explorer").run()
    assert not app.exception
    assert "Import an HMR2 reconstruction" in app.info[0].value
    assert not app.slider
    app.radio[0].set_value("3D Human Mesh").run()
    assert not app.exception and len(app.get("plotly_chart")) == 1


@pytest.mark.parametrize("dtype", [">f2", ">f4", ">f8"])
def test_uploaded_big_endian_pose_is_displayed(dtype):
    fixture = fixture_bytes()
    arrays = load_bundle(fixture).people["person_000"]
    root = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype)
    arrays["global_orient"] = root[None]
    arrays["body_pose"] = np.repeat(root.T[None], 23, axis=0).astype(dtype)
    person_data = BytesIO()
    np.savez(person_data, **arrays)
    archive = BytesIO()
    with ZipFile(BytesIO(fixture)) as source, ZipFile(archive, "w") as output:
        for name in source.namelist():
            payload = person_data.getvalue() if name == "people/person_000.npz" else source.read(name)
            output.writestr(name, payload)

    app = AppTest.from_file(str(APP_PATH)).run()
    app.get("file_uploader")[0].upload("big-endian.zip", archive.getvalue()).run()
    assert not app.exception
    assert app.success
    np.testing.assert_array_equal(app.dataframe[1].value, root)
    app.selectbox(key="pose-joint").select("Left knee").run()
    assert not app.exception
    np.testing.assert_array_equal(app.dataframe[1].value, root.T)

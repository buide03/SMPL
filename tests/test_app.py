from pathlib import Path

import numpy as np
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

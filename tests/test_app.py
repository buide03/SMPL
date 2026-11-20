from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_viewer_modes_and_person_selection():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert "SYNTHETIC TEST DATA" in app.warning[0].value
    assert len(app.get("image")) == 3
    for mode, count in (("Original RGB", 0), ("3D Human Mesh", 1),
                        ("Mesh Wireframe", 1), ("All Detected People", 2)):
        app.radio[0].set_value(mode).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == count
    app.selectbox[-1].select("person_001").run()
    assert not app.exception
    assert any("person_001" in heading.value for heading in app.subheader)


def test_zip_upload_and_validation_error():
    from src.synthetic_fixture import fixture_bytes

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.get("file_uploader")[0].upload("fixture.zip", fixture_bytes()).run()
    assert not app.exception
    assert app.success
    app.get("file_uploader")[0].upload("invalid.zip", b"invalid").run()
    assert not app.exception
    assert "Invalid reconstruction archive" in app.error[0].value

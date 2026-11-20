from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from colab.export_results import predictions_from_batch, write_reconstruction_zip
from colab.run_hmr2 import select_people
from src.bundle_loader import BundleError, load_bundle


def predictions():
    identities = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 24, 3, 3))
    vertices = np.arange(2 * 6890 * 3, dtype=np.float32).reshape(2, 6890, 3)
    faces = np.arange(13776 * 3, dtype=np.int32).reshape(13776, 3) % 6890
    return predictions_from_batch(
        {"betas": np.zeros((2, 10), dtype=np.float32), "global_orient": identities[:, :1],
         "body_pose": identities[:, 1:]},
        vertices, faces,
        np.array([[1, 0, 0], [2, 0.5, 0.75]], dtype=np.float32),
        np.array([[0, 0, 10], [0.5, 0.75, 5]], dtype=np.float32),
        np.array([[3, 4, 12], [6, 8, 7]], dtype=np.float32),
        np.array([[0, 0, 4, 10], [5, 2, 10, 10]], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
    )


def write_example(path: Path, people, *, overlay_size=(10, 10)) -> Path:
    # Numerical fixtures exercise the real-inference schema in a temporary directory.
    return write_reconstruction_zip(
        path, metadata={
            "scene_id": "006", "frame_id": "03", "camera_id": "front_camera",
            "model_name": "schema test", "checkpoint_identifier": "test-only",
            "smpl_model_version": "test-only",
        },
        source=Image.new("RGB", (10, 10), "gray"),
        overlay=Image.new("RGB", overlay_size, "green"),
        side_view=Image.new("RGB", (20, 10), "white"),
        predictions=people,
    )


def test_export_round_trips_native_arrays_and_camera(tmp_path):
    people = predictions()
    path = write_example(tmp_path / "result.zip", people)
    bundle = load_bundle(path.read_bytes())
    assert bundle.manifest["frame_id"] == "03"
    assert bundle.manifest["coordinate_system"] == "smpl_local"
    assert list(bundle.people) == ["person_000", "person_001"]
    assert bundle.images["overlay.png"].getpixel((0, 0)) == (0, 128, 0)
    assert bundle.images["side_view.png"].size == (20, 10)
    for person_id, expected, record in zip(bundle.people, people, bundle.manifest["people"], strict=True):
        for name, values in bundle.people[person_id].items():
            np.testing.assert_array_equal(values, expected.arrays[name])
        assert record["full_frame_cam_t"] == expected.full_frame_translation.tolist()
        assert record["detector_score"] == expected.detector_score


def test_invalid_export_keeps_directory_clean(tmp_path):
    with pytest.raises(BundleError, match="dimensions differ"):
        write_example(tmp_path / "result.zip", predictions(), overlay_size=(2, 2))
    assert list(tmp_path.iterdir()) == []


def test_existing_result_is_preserved(tmp_path):
    path = tmp_path / "result.zip"
    path.write_bytes(b"old output")
    with pytest.raises(FileExistsError):
        write_example(path, predictions())
    assert path.read_bytes() == b"old output"


def test_person_filter_clips_and_keeps_scores_with_their_boxes():
    boxes, scores = select_people(
        np.array([[-4, 0, 12, 30], [5, 4, 11, 9], [15, 4, 19, 12], [2, 3, np.nan, 8],
                  [60, 50, 75, 65], [4, 2, 8, 6]], dtype=np.float32),
        np.array([1, 1, 8, 1, 1, 1]),
        np.array([0.55, 0.95, 0.99, 0.99, 0.90, 0.49], dtype=np.float32),
        person_label=1, threshold=0.5, size=(20, 16),
    )
    np.testing.assert_array_equal(boxes, [[5, 4, 11, 9], [0, 0, 12, 16]])
    np.testing.assert_allclose(scores, [0.95, 0.55])

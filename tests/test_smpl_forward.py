import os
from pathlib import Path

import numpy as np
import pytest

from src.mesh_viewer import comparison_figure
from src.pose_utils import JOINTS, neutral_rotations, rotations_to_vectors, vector_to_rotation
from src.smpl_forward import (
    SMPLModelError, imported_local_vertices, load_neutral_smpl, smpl_forward,
)


def test_rotation_vector_round_trip_at_half_turn():
    pytest.importorskip("scipy")
    rotation = vector_to_rotation(np.array([0.0, 180.0, 0.0]))
    recovered = rotations_to_vectors(rotation[None])[0]
    np.testing.assert_allclose(vector_to_rotation(recovered), rotation, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_crop_camera_reference_is_rebased_once_without_modification():
    vertices = np.array([[0.0, 0.0, 3.25]])
    arrays = {"vertices": vertices, "pred_cam_t": np.array([0.0, 0.0, 3.0])}
    np.testing.assert_array_equal(imported_local_vertices(arrays, "smpl_local"), vertices)
    np.testing.assert_array_equal(imported_local_vertices(arrays, "hmr2_crop_camera"), [[0, 0, 0.25]])
    np.testing.assert_array_equal(vertices, [[0.0, 0.0, 3.25]])


def test_missing_model_has_readable_error(tmp_path):
    with pytest.raises(SMPLModelError, match="authorized local neutral"):
        load_neutral_smpl(tmp_path / "SMPL_NEUTRAL.pkl")


def test_comparison_plot_keeps_native_vertices_and_shared_axis_scales():
    reference = np.array([[0., -0.3, 0.2], [1., 0.6, 0.1], [0., 0.0, 0.0]])
    canonical = reference * [1, -1, -1]
    posed = reference * 1.5
    faces = np.array([[0, 1, 2]])
    figure = comparison_figure(canonical, posed, reference, faces, show_original=True)
    assert [trace.scene for trace in figure.data] == ["scene", "scene2", "scene2"]
    for trace, points in zip(figure.data, (canonical, posed, reference), strict=True):
        np.testing.assert_array_equal(np.column_stack((trace.x, trace.y, trace.z)), points)
    assert figure.layout.scene.yaxis.range == tuple(-y for y in reversed(figure.layout.scene2.yaxis.range))


def test_licensed_smpl_forward_on_cpu_changes_shape_and_articulates_knee():
    path = os.environ.get("SMPL_MODEL_PATH")
    if not path:
        pytest.skip("Set SMPL_MODEL_PATH to run the licensed CPU integration test.")
    torch = pytest.importorskip("torch")
    pytest.importorskip("smplx")
    model = load_neutral_smpl(Path(path))
    assert model.layer.v_template.device == torch.device("cpu")
    assert model.faces.shape == (13776, 3)
    beta = np.zeros(10)
    rest = neutral_rotations()
    baseline = smpl_forward(beta, rest[1:], rest[:1], model=model)
    assert baseline.shape == (6890, 3)
    beta[0] = 2
    changed_shape = smpl_forward(beta, rest[1:], rest[:1], model=model)
    assert np.max(np.linalg.norm(changed_shape - baseline, axis=1)) > 0.005
    edited = rest.copy()
    knee = JOINTS.index("Left knee")
    edited[knee] = vector_to_rotation(np.array([50.0, 0.0, 0.0]))
    changed_pose = smpl_forward(beta, edited[1:], edited[:1], model=model)
    leg = [knee, JOINTS.index("Left ankle"), JOINTS.index("Left foot")]
    lower_leg = model.layer.lbs_weights[:, leg].sum(dim=1).numpy() > 0.5
    assert np.max(np.linalg.norm((changed_pose - changed_shape)[lower_leg], axis=1)) > 0.03
    assert np.max(np.linalg.norm(changed_pose - changed_shape, axis=1)) > 0.03
    np.testing.assert_array_equal(smpl_forward(beta, rest[1:], rest[:1], model=model), changed_shape)

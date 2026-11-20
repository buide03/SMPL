import numpy as np
import pytest

from src.bundle_loader import load_bundle
from src.synthetic_fixture import fixture_bytes
from src.validation import validate_person


@pytest.mark.parametrize(("name", "value", "message"), [
    ("betas", np.full(10, np.nan), "non-finite"),
    ("pred_cam", np.array([np.inf, 0., 0.]), "non-finite"),
    ("faces", np.array([[0, 1, 9]]), "outside vertices"),
    ("faces", np.array([[0, 1, 1]]), "repeats"),
    ("faces", np.array([[0., 1., 2.]]), "dtype"),
    ("global_orient", np.diag([-1., 1., 1.])[None], "proper rotation"),
    ("body_pose", np.zeros((23, 3, 3)), "proper rotation"),
    ("bbox_xyxy", np.array([-1., 0., 25., 25.]), "inside"),
])
def test_validation(name, value, message):
    arrays = load_bundle(fixture_bytes()).people["person_000"]
    arrays[name] = value
    with pytest.raises(ValueError, match=message):
        validate_person(arrays, 640, 360, True)

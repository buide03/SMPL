"""Validate the numeric contract before data reaches the viewer."""
import numpy as np

ROTATION_TOLERANCE = 1e-4
SHAPES = {
    "betas": (10,), "global_orient": (1, 3, 3), "body_pose": (23, 3, 3),
    "vertices": (6890, 3), "faces": (13776, 3), "pred_cam": (3,),
    "pred_cam_t": (3,), "bbox_xyxy": (4,),
}


def validate_person(arrays: dict[str, np.ndarray], width: int, height: int,
                    synthetic: bool) -> None:
    """Validate dimensions, finite values, triangle indices and SO(3)."""
    for name, expected in SHAPES.items():
        values = arrays[name]
        if synthetic and name in ("vertices", "faces"):
            shape_ok = values.ndim == 2 and values.shape[1] == 3 and 0 < len(values) <= expected[0]
        else:
            shape_ok = values.shape == expected
        if not shape_ok:
            raise ValueError(f"{name}: invalid shape {values.shape}, expected {expected}")
        kinds = "iu" if name == "faces" else "f"
        if values.dtype.kind not in kinds or values.dtype.itemsize > 8 or not np.isfinite(values).all():
            raise ValueError(f"{name}: invalid dtype or non-finite values")
    faces = arrays["faces"]
    if faces.min() < 0 or faces.max() >= len(arrays["vertices"]):
        raise ValueError("faces: index outside vertices")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) |
              (faces[:, 0] == faces[:, 2])):
        raise ValueError("faces: triangle repeats a vertex index")
    for name in ("global_orient", "body_pose"):
        rotations = arrays[name].astype(np.float64)
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3),
                            rtol=0, atol=ROTATION_TOLERANCE)
                or not np.allclose(np.linalg.det(rotations), 1, rtol=0, atol=ROTATION_TOLERANCE)):
            raise ValueError(f"{name}: expected proper rotation matrices")
    x1, y1, x2, y2 = arrays["bbox_xyxy"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox_xyxy: expected a nonempty box inside the source image")

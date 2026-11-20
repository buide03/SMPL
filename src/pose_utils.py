"""SMPL joint order and axis-angle controls for the CPU editor."""

import numpy as np


JOINTS = (
    "Pelvis", "Left hip", "Right hip", "Spine 1", "Left knee", "Right knee", "Spine 2",
    "Left ankle", "Right ankle", "Spine 3", "Left foot", "Right foot", "Neck",
    "Left collar", "Right collar", "Head", "Left shoulder", "Right shoulder",
    "Left elbow", "Right elbow", "Left wrist", "Right wrist", "Left hand", "Right hand",
)


def neutral_rotations() -> np.ndarray:
    """Return the identity rotation for the pelvis and each of the 23 body joints."""
    return np.repeat(np.eye(3, dtype=np.float32)[None], len(JOINTS), axis=0)


def rotations_to_vectors(rotations: np.ndarray) -> np.ndarray:
    """Return principal rotation vectors in degrees for the UI."""
    from scipy.spatial.transform import Rotation

    vectors = Rotation.from_matrix(np.array(rotations, dtype=np.float64)).as_rotvec(degrees=True)
    return np.clip(vectors, -180, 180)


def vector_to_rotation(vector: np.ndarray) -> np.ndarray:
    """Convert one axis-angle vector in degrees to a rotation matrix."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(vector, degrees=True).as_matrix()

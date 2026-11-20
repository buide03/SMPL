"""CPU-only SMPL Forward and consistency checks against an imported mesh."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
import pickle
import re
from typing import TYPE_CHECKING

import numpy as np

from src.validation import PersonArrays, SHAPES, validate_array_spec
from src.pose_utils import JOINTS


if TYPE_CHECKING:
    from smplx import SMPLLayer


SMPL_MATCH_TOLERANCE_M = 1e-4
SMPLIFY_NEUTRAL = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


class SMPLModelError(ValueError):
    """The local SMPL model is unavailable or inconsistent with an imported mesh."""


@dataclass(frozen=True)
class NeutralSMPL:
    layer: SMPLLayer
    faces: np.ndarray
    sha256: str


@dataclass(frozen=True)
class MeshAgreement:
    rms_m: float
    max_m: float


@dataclass(frozen=True)
class SMPLGeometry:
    vertices: np.ndarray
    joints: np.ndarray


def load_neutral_smpl(path: Path) -> NeutralSMPL:
    """Open a trusted, user-supplied neutral asset through the official SMPL layer."""
    if not path.is_file() or path.suffix.lower() != ".pkl":
        raise SMPLModelError("Choose an authorized local neutral SMPL .pkl asset.")
    try:
        import torch
        from smplx import SMPLLayer
    except ImportError as exc:
        raise SMPLModelError("Install the optional CPU SMPL dependencies in requirements-smpl.txt.") from exc

    try:
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        layer = SMPLLayer(model_path=str(path), gender="neutral", num_betas=10)
        layer = layer.eval().to(device="cpu", dtype=torch.float32)
    except ImportError as exc:
        raise SMPLModelError("Install the optional dependencies for this SMPL asset, including chumpy.") from exc
    except (OSError, ValueError, RuntimeError, EOFError, TypeError, AttributeError,
            KeyError, pickle.UnpicklingError) as exc:
        raise SMPLModelError(f"Could not load the local neutral SMPL model. {exc}") from exc
    faces = np.array(layer.faces, copy=True)
    if tuple(layer.v_template.shape) != SHAPES["vertices"] or faces.shape != SHAPES["faces"]:
        raise SMPLModelError("The local model must have standard SMPL topology (6,890 vertices and 13,776 faces).")
    faces.setflags(write=False)
    return NeutralSMPL(layer, faces, digest)


def smpl_forward(
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    *,
    model: NeutralSMPL,
) -> np.ndarray:
    """Generate posed local XYZ in meters from beta coefficients and rotation matrices."""
    return smpl_geometry(betas, body_pose[None], global_orient[None], model=model).vertices[0]


def smpl_geometry(
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    *,
    model: NeutralSMPL,
) -> SMPLGeometry:
    """Evaluate a batch of poses with one shape, returning vertices and 24 kinematic joints per pose."""
    import torch

    count = len(body_pose)
    if not count or count != len(global_orient):
        raise ValueError("SMPL geometry requires the same nonzero number of body and root rotations.")
    inputs = {"betas": betas, "body_pose": body_pose, "global_orient": global_orient}
    for name, values in inputs.items():
        shape = values.shape if name == "betas" else values.shape[1:]
        validate_array_spec(name, shape, values.dtype, synthetic=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite values")
    tensors = {name: torch.tensor(np.asarray(value, dtype=np.float32), device="cpu")
               for name, value in inputs.items()}
    tensors["betas"] = tensors["betas"].expand(count, -1)
    with torch.inference_mode():
        result = model.layer(**tensors, pose2rot=False, return_verts=True)
        vertices = result.vertices.numpy().copy()
        # SMPLLayer appends extra surface landmarks after the kinematic joints.
        joints = result.joints[:, :len(JOINTS)].numpy().copy()
    if (vertices.shape != (count, *SHAPES["vertices"]) or joints.shape != (count, len(JOINTS), 3)
            or not np.isfinite(vertices).all() or not np.isfinite(joints).all()):
        raise SMPLModelError("SMPL Forward returned invalid geometry.")
    return SMPLGeometry(vertices, joints)


def imported_local_vertices(arrays: PersonArrays, coordinate_system: str) -> np.ndarray:
    """Express an imported prediction in the SMPL vertex frame used by Forward."""
    vertices = arrays["vertices"].astype(np.float64)
    if coordinate_system == "smpl_local":
        return vertices
    if coordinate_system == "hmr2_crop_camera":
        return vertices - arrays["pred_cam_t"]
    raise SMPLModelError("This vertex coordinate system cannot be compared to SMPL Forward.")


def check_imported_prediction(
    model: NeutralSMPL,
    arrays: PersonArrays,
    local_vertices: np.ndarray,
    smpl_model_version: str,
) -> MeshAgreement:
    """Check model identity, ordered faces, and the unedited vertex reconstruction."""
    digest = re.search(r"\bsha256:([0-9a-f]{64})\b", smpl_model_version, re.IGNORECASE)
    if digest and digest.group(1).lower() != model.sha256:
        raise SMPLModelError("The local SMPL file does not match the SHA-256 recorded by the exporter.")
    if not np.array_equal(model.faces, arrays["faces"]):
        raise SMPLModelError("The imported face order differs from the local SMPL topology.")
    baseline = smpl_forward(arrays["betas"], arrays["body_pose"], arrays["global_orient"], model=model)
    distances = np.linalg.norm(baseline - local_vertices, axis=1)
    error = MeshAgreement(float(np.sqrt(np.mean(distances ** 2))), float(np.max(distances)))
    if error.max_m > SMPL_MATCH_TOLERANCE_M:
        raise SMPLModelError(f"SMPL Forward differs from the imported mesh by up to {error.max_m * 1000:.3f} mm. "
                             "Check the neutral model and vertex coordinate system.")
    return error

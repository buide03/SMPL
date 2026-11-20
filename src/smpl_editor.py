"""Streamlit controls for editing a validated SMPL prediction."""

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
import os
from pathlib import Path

import numpy as np
import streamlit as st

from src.bundle_loader import Bundle
from src.mesh_viewer import comparison_figure, linked_comparison_html
from src.pose_utils import JOINTS, neutral_rotations, rotations_to_vectors, vector_to_rotation
from src.smpl_forward import (
    MeshAgreement, NeutralSMPL, SMPL_MATCH_TOLERANCE_M, SMPLModelError, SMPLIFY_NEUTRAL,
    check_imported_prediction, imported_local_vertices, load_neutral_smpl, smpl_forward,
)
from src.validation import PersonArrays


@dataclass
class Edits:
    betas: np.ndarray
    rotations: np.ndarray
    vectors: np.ndarray
    template_up: bool


@dataclass
class EditorSession:
    original_vertices: np.ndarray
    agreement: MeshAgreement
    edits: Edits
    widget_revision: int = 0


def initial_edits(arrays: PersonArrays, neutral: bool = False) -> Edits:
    """Use independent arrays so an edit never writes into the imported prediction."""
    if neutral:
        rotations = neutral_rotations()
        return Edits(np.zeros(10), rotations, np.zeros((len(rotations), 3)), template_up=True)
    rotations = np.concatenate((arrays["global_orient"], arrays["body_pose"])).astype(np.float64)
    return Edits(arrays["betas"].astype(np.float64), rotations, rotations_to_vectors(rotations),
                 template_up=False)


@st.cache_resource(max_entries=2, show_spinner=False)
def cached_model(path: Path, file_revision: tuple[int, int]) -> NeutralSMPL:
    """Cache an evaluated CPU model by path, modification time, and size."""
    return load_neutral_smpl(path)


def configured_model() -> NeutralSMPL:
    configured = os.environ.get("SMPL_MODEL_PATH")
    models = Path(__file__).resolve().parents[1] / "models"
    candidates = [Path(configured).expanduser()] if configured else [
        models / "SMPL_NEUTRAL.pkl", models / SMPLIFY_NEUTRAL,
    ]
    try:
        for path in candidates:
            if path.is_file():
                info = path.stat()
                return cached_model(path.resolve(), (info.st_mtime_ns, info.st_size))
    except OSError as exc:
        raise SMPLModelError(f"Could not read the configured model. {exc}") from exc
    raise SMPLModelError(
        "Set SMPL_MODEL_PATH to an authorized neutral SMPL .pkl and restart Streamlit. "
        "See models/README.md for Windows setup."
    )


def reset_edits(session: EditorSession, arrays: PersonArrays, neutral: bool) -> None:
    session.edits = initial_edits(arrays, neutral=neutral)
    # Remount sliders with the new values in the browser after a reset.
    session.widget_revision += 1


def update_beta(edits: Edits, index: int, key: str) -> None:
    edits.betas[index] = st.session_state[key]


def update_joint(edits: Edits, joint: int, axis: int, key: str) -> None:
    edits.vectors[joint, axis] = st.session_state[key]
    edits.rotations[joint] = vector_to_rotation(edits.vectors[joint])


def editor_parameters(edits: Edits, arrays: PersonArrays, scope: str, prefix: str) -> None:
    shape, pose = st.tabs(("Shape β", "Pose θ"))
    with shape:
        st.caption("HMR2 values appear beside each coefficient.")
        columns = st.columns(2)
        for index, beta in enumerate(edits.betas):
            original = float(arrays["betas"][index])
            limit = max(5, ceil(abs(original)))
            key = f"{prefix}beta-{index}"
            columns[index % 2].slider(
                f"β{index} · {original:+.4f}", -float(limit), float(limit), float(beta),
                step=0.01, key=key, on_change=update_beta, args=(edits, index, key),
            )
    with pose:
        joint = st.selectbox("SMPL joint", JOINTS, key=f"smpl-joint-{scope}")
        index = JOINTS.index(joint)
        st.caption("Pelvis is the root rotation." if index == 0 else
                   "This rotation is local to the parent joint.")
        st.caption("Axis-angle components in degrees. The vector's length gives the rotation angle.")
        for axis, label in enumerate(("X", "Y", "Z")):
            key = f"{prefix}joint-{index}-{axis}"
            st.slider(f"{label} component", -180.0, 180.0, float(edits.vectors[index, axis]), step=1.0,
                      key=key, on_change=update_joint, args=(edits, index, axis, key))


def smpl_editor(bundle: Bundle, person_id: str | None, package_id: str) -> None:
    """Render linked canonical and posed views when the local prediction can be reproduced."""
    st.subheader("SMPL Explorer")
    if bundle.manifest["data_kind"] == "synthetic" or person_id is None:
        st.info("Import an HMR2 reconstruction and select a person to enable the editor.")
        return
    arrays = bundle.people[person_id]
    try:
        with st.spinner("Checking the local SMPL model against the imported prediction"):
            model = configured_model()
            scope = sha256(f"{package_id}:{person_id}:{model.sha256}".encode()).hexdigest()
            sessions = st.session_state.setdefault("smpl_editor_sessions", {})
            if scope not in sessions:
                reference = imported_local_vertices(arrays, bundle.manifest["coordinate_system"])
                agreement = check_imported_prediction(
                    model, arrays, reference, bundle.manifest["smpl_model_version"],
                )
                sessions[scope] = EditorSession(reference, agreement, initial_edits(arrays))
            session = sessions[scope]
    except (ImportError, RuntimeError, ValueError) as exc:
        st.warning(f"SMPL editing unavailable. {exc}")
        return

    prefix = f"smpl-input-{scope}-{session.widget_revision}-"
    st.caption(f"{person_id} · CPU SMPL Forward · vertices in local SMPL coordinates")
    st.caption(f"Prediction match · RMS {session.agreement.rms_m * 1000:.6f} mm · "
               f"maximum {session.agreement.max_m * 1000:.6f} mm · "
               f"tolerance {SMPL_MATCH_TOLERANCE_M * 1000:.2f} mm")
    prediction, neutral, comparison = st.columns((1, 1, 2))
    prediction.button("Reset to Prediction", key=f"restore-{scope}", on_click=reset_edits,
                      args=(session, arrays, False))
    neutral.button("Reset to Neutral", key=f"neutral-{scope}", on_click=reset_edits,
                   args=(session, arrays, True))
    show_original = comparison.checkbox("Show original HMR2 mesh", key=f"original-{scope}")

    controls, preview = st.columns((1, 2.3))
    with controls:
        editor_parameters(session.edits, arrays, scope, prefix)
    with preview:
        edits = session.edits
        rest = neutral_rotations()
        try:
            canonical = smpl_forward(edits.betas, rest[1:], rest[:1], model=model)
            posed = smpl_forward(edits.betas, edits.rotations[1:], edits.rotations[:1], model=model)
        except (RuntimeError, ValueError) as exc:
            st.warning(f"Could not regenerate the mesh. {exc}")
            return
        figure = comparison_figure(canonical, posed, session.original_vertices, model.faces,
                                   show_original=show_original, posed_y_up=edits.template_up)
        st.iframe(linked_comparison_html(figure, scope), height=433,
                  alt="Linked canonical and posed SMPL mesh viewports")
        st.caption("The canonical mesh uses the current β at rest. The posed mesh uses β and θ. "
                   "Camera parameters control placement in the source image.")

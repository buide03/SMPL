"""Inspect packaged SMPL predictions in a local Streamlit workspace."""

from hashlib import sha256
from math import ceil, floor
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from src.bundle_loader import Bundle, BundleError, MAX_UPLOAD, SYNTHETIC_LABEL, load_bundle
from src.mesh_geometry import create_mesh, export_mesh, mesh_statistics
from src.mesh_viewer import PLOTLY_CONFIG, mesh_figure
from src.pose_utils import JOINTS
from src.smpl_editor import smpl_editor
from src.smpl_lesson import LESSON_MODE, smpl_lesson
from src.synthetic_fixture import fixture_bytes
from src.validation import PersonArrays

MODES = (
    "Original RGB",
    "RGB + Reconstruction Overlay",
    "3D Human Mesh",
    "Mesh Wireframe",
    "All Detected People",
    "SMPL Explorer",
    LESSON_MODE,
)


@st.cache_data(max_entries=1, show_spinner=False)
def demo_package() -> bytes:
    return fixture_bytes()


@st.cache_data(max_entries=1, show_spinner=False)
def local_example(path: Path, revision: tuple[int, int]) -> bytes:
    with path.open("rb") as archive:
        return archive.read(MAX_UPLOAD + 1)


def load_for_session(payload: bytes) -> tuple[str, Bundle]:
    """Keep only the current validated package in this browser session."""
    package_id = sha256(payload).hexdigest()
    if st.session_state.get("package_id") != package_id:
        st.session_state.pop("package_id", None)
        st.session_state.pop("bundle", None)
        st.session_state.pop("smpl_editor_sessions", None)
        st.session_state.pop("smpl_lesson_baseline", None)
        st.session_state["bundle"] = load_bundle(payload)
        st.session_state["package_id"] = package_id
    return package_id, st.session_state["bundle"]


def draw_detections(bundle: Bundle, selected: str | None) -> Image.Image:
    """Label boxes in source pixel space and draw the active detection last."""
    image = bundle.images["source.png"].copy()
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 450)
    font_size = max(14, image.width // 85)
    for person_id in sorted(bundle.people, key=lambda value: value == selected):
        x1, y1, x2, y2 = bundle.people[person_id]["bbox_xyxy"]
        box = (floor(x1), floor(y1), ceil(x2) - 1, ceil(y2) - 1)
        color = "#78f9d7" if person_id == selected else "#eed58b"
        draw.rectangle(box, outline=color, width=line_width)
        label_width = draw.textlength(person_id, font_size=font_size) + 10
        label_x = max(0, min(x1, image.width - label_width))
        label_y = max(0, y1 - font_size - 6)
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + font_size + 6),
                       fill=color)
        draw.text((label_x + 5, label_y + 3), person_id, anchor="lt",
                  fill="#0b2521", font_size=font_size)
    return image


def source_panel(bundle: Bundle, selected: str | None, *, boxed: bool = True) -> None:
    with st.container(border=True):
        st.subheader("Original RGB Image")
        preview = st.container()
        show_boxes = st.checkbox("Show detection boxes", value=boxed, disabled=not bundle.people)
        image = draw_detections(bundle, selected) if show_boxes else bundle.images["source.png"]
        preview.image(image, width="stretch")


def overlay_panel(bundle: Bundle, synthetic: bool) -> None:
    with st.container(border=True):
        st.subheader("Synthetic Overlay" if synthetic else "HMR2 Reconstruction")
        st.image(bundle.images["overlay.png"], width="stretch")
        if "side_view.png" in bundle.images:
            with st.expander("Side view"):
                st.image(bundle.images["side_view.png"], width="stretch")


def geometry_panel(arrays: PersonArrays, person_id: str, package_id: str) -> None:
    with st.expander("Open3D geometry and export"):
        if not st.checkbox("Run geometry analysis", key=f"geometry-{package_id}-{person_id}"):
            return
        try:
            mesh = create_mesh(arrays["vertices"], arrays["faces"])
            st.json(mesh_statistics(mesh))
            for column, extension in zip(st.columns(2), ("obj", "ply")):
                column.download_button(
                    f"Download {extension.upper()}", export_mesh(mesh, extension),
                    f"{person_id}.{extension}", "application/octet-stream", on_click="ignore",
                )
        except (ImportError, OSError, RuntimeError) as exc:
            st.warning(f"Open3D geometry unavailable. {exc}")


def parameters_panel(person_id: str, arrays: PersonArrays, package_id: str, synthetic: bool) -> None:
    with st.container(border=True):
        st.subheader("SMPL Parameters")
        st.caption(f"{person_id} · {'Fixture values' if synthetic else 'Imported values'}")
        shape, pose, camera = st.tabs(("Shape β", "Pose θ", "Camera & mesh"))
        with shape:
            st.caption("Each β is a coefficient of the learned shape basis.")
            st.dataframe(
                pd.DataFrame({"Coefficient": [f"β{i}" for i in range(10)],
                              "Value": arrays["betas"].astype(float, copy=False)}),
                hide_index=True, height=290, width="stretch",
            )
        with pose:
            st.caption("Pelvis gives the global orientation. The 23 body rotations are parent-relative.")
            joint = st.selectbox("Rotation matrix", JOINTS, key="pose-joint")
            index = JOINTS.index(joint)
            rotation = arrays["body_pose"][index - 1] if index else arrays["global_orient"][0]
            # Arrow requires native byte order. A float64 view or copy preserves every input value.
            st.dataframe(rotation.astype(float, copy=False), hide_index=True, width="stretch")
        with camera:
            st.json({
                "pred_cam [s, tx, ty]": arrays["pred_cam"].tolist(),
                "pred_cam_t [tx, ty, tz]": arrays["pred_cam_t"].tolist(),
                "bbox_xyxy [px]": arrays["bbox_xyxy"].tolist(),
                "vertices": len(arrays["vertices"]),
                "faces": len(arrays["faces"]),
            })
        geometry_panel(arrays, person_id, package_id)


def mesh_panel(bundle: Bundle, person_id: str | None, package_id: str, mode: str) -> None:
    all_people = mode == "All Detected People"
    synthetic = bundle.manifest["data_kind"] == "synthetic"
    with st.container(border=True):
        st.subheader("Detected People" if all_people else
                     "3D Fixture Mesh" if synthetic else "3D Human Mesh")
        if person_id is None:
            st.info("This package contains no detections.")
            return
        st.caption(f"Vertex frame · {bundle.manifest['coordinate_system']}")
        axes = st.checkbox("Show coordinate axes", value=True)
        wireframe = (st.checkbox("Wireframe", key="all-wireframe") if all_people
                     else mode == "Mesh Wireframe")
        if all_people:
            st.caption("Each detection uses a separate viewport in its stored coordinate frame.")
        ids = list(bundle.people) if all_people else [person_id]
        page_size = 4
        if len(ids) > page_size:
            pages = (len(ids) + page_size - 1) // page_size
            page = st.selectbox("Mesh page", range(1, pages + 1), key=f"pages-{package_id}")
            ids = ids[(page - 1) * page_size:page * page_size]
        for index in range(0, len(ids), 2):
            row = ids[index:index + 2]
            columns = st.columns(2) if len(row) == 2 else [st.container()]
            for column, current in zip(columns, row):
                column.plotly_chart(
                    mesh_figure(current, bundle.people[current], wireframe, axes, package_id),
                    key=f"mesh-{package_id}-{current}", width="stretch", theme=None,
                    config=PLOTLY_CONFIG,
                )


def main() -> None:
    st.set_page_config(page_title="SMPL Reconstruction Lab", page_icon="🧊", layout="wide")
    st.title("SMPL Reconstruction Lab")
    st.caption("Inspect RGB observations, SMPL parameters, and mesh geometry.")

    with st.sidebar:
        st.header("Workspace")
        upload = st.file_uploader("Reconstruction ZIP", type=["zip"],
                                  help="One scene, frame, and camera. Up to 64 MiB.")
        example_path = os.environ.get("SMPL_DEMO_ZIP")
        demo = st.checkbox("Dùng ví dụ PandaSet" if example_path else "Use synthetic fixture",
                           value=True, disabled=upload is not None)
        if upload is None and not demo:
            st.info("Upload a reconstruction ZIP to begin.")
            return
        local_demo = upload is None and bool(example_path)
        try:
            with st.spinner("Validating reconstruction archive"):
                if upload is not None:
                    payload = upload.getvalue()
                elif example_path:
                    path = Path(example_path).expanduser()
                    info = path.stat()
                    payload = local_example(path, (info.st_mtime_ns, info.st_size))
                else:
                    payload = demo_package()
                package_id, bundle = load_for_session(payload)
        except (BundleError, OSError) as exc:
            st.error(str(exc))
            return
        metadata = bundle.manifest
        st.divider()
        st.subheader("Observation")
        st.selectbox("Scene", [metadata["scene_id"]])
        st.selectbox("Frame", [metadata["frame_id"]])
        st.selectbox("Camera", [metadata["camera_id"]])
        people = list(bundle.people)
        preferred = os.environ.get("SMPL_DEMO_PERSON") if local_demo else None
        initial_person = people.index(preferred) if preferred in people else 0
        person_id = st.selectbox("Person", people, index=initial_person, placeholder="No detections",
                                 key=f"person-{package_id}")
        st.success(f"Archive validated · {len(bundle.people)} detections")
        with st.expander("Reconstruction metadata"):
            st.json(metadata, expanded=1)

    synthetic = metadata["data_kind"] == "synthetic"
    if synthetic:
        st.warning(SYNTHETIC_LABEL)

    mode = st.radio("Visualization", MODES, index=MODES.index(LESSON_MODE) if local_demo else 1, horizontal=True)
    if mode == LESSON_MODE:
        smpl_lesson(bundle, person_id, package_id)
        return
    if not synthetic:
        st.text(f"Imported inference · {metadata['model_name']} · {metadata['checkpoint_identifier']}")
    if mode == "SMPL Explorer":
        smpl_editor(bundle, person_id, package_id)
        return
    if mode == "Original RGB":
        source_panel(bundle, person_id, boxed=False)
    elif mode == "RGB + Reconstruction Overlay":
        left, right = st.columns(2)
        with left:
            source_panel(bundle, person_id)
        with right:
            overlay_panel(bundle, synthetic)
    if mode == "All Detected People":
        mesh_panel(bundle, person_id, package_id, mode)
    if mode in ("RGB + Reconstruction Overlay", "3D Human Mesh", "Mesh Wireframe"):
        if person_id is None:
            mesh_panel(bundle, person_id, package_id, mode)
        else:
            mesh_column, parameters_column = st.columns([3, 2])
            with mesh_column:
                mesh_panel(bundle, person_id, package_id, mode)
            with parameters_column:
                parameters_panel(person_id, bundle.people[person_id], package_id, synthetic)
    elif person_id is not None:
        parameters_panel(person_id, bundle.people[person_id], package_id, synthetic)


if __name__ == "__main__":
    main()

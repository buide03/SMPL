"""Windows CPU viewer for validated SMPL reconstruction packages."""
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

from src.bundle_loader import Bundle, BundleError, SYNTHETIC_LABEL, load_bundle
from src.mesh_geometry import create_mesh, export_mesh, mesh_statistics
from src.mesh_viewer import mesh_figure
from src.synthetic_fixture import fixture_bytes


def geometry_panel(arrays: dict[str, np.ndarray], person_id: str) -> None:
    with st.expander("Open3D geometry and export"):
        if not st.checkbox("Run geometry analysis", key=f"geometry-{person_id}"):
            return
        try:
            mesh = create_mesh(arrays["vertices"], arrays["faces"])
            st.json(mesh_statistics(mesh))
            for extension in ("obj", "ply"):
                st.download_button(extension.upper(), export_mesh(mesh, extension),
                                   f"{person_id}.{extension}", "application/octet-stream")
        except (ImportError, OSError, RuntimeError) as exc:
            st.warning(f"Open3D geometry unavailable: {exc}")


def draw_detections(bundle: Bundle) -> Image.Image:
    image = bundle.images["source.png"].copy()
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 450)
    for person_id, arrays in bundle.people.items():
        x1, y1, x2, y2 = arrays["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline="#00e5ff", width=line_width)
        draw.text((x1 + 3, y1 + 3), person_id, fill="white", stroke_width=2, stroke_fill="black",
                  font_size=max(13, image.width // 65))
    return image


def show_parameters(person_id: str, arrays: dict[str, np.ndarray]) -> None:
    st.subheader(f"SMPL Parameters · {person_id}")
    st.caption("β coefficients are coordinates in the learned SMPL shape basis.")
    st.table([{"Coefficient": f"β{i}", "Value": float(beta)}
              for i, beta in enumerate(arrays["betas"])])
    st.json({
        "global_orient": arrays["global_orient"].tolist(),
        "body_pose (23 joint rotation matrices)": arrays["body_pose"].tolist(),
        "pred_cam": arrays["pred_cam"].tolist(),
        "pred_cam_t": arrays["pred_cam_t"].tolist(),
        "bbox_xyxy": arrays["bbox_xyxy"].tolist(),
        "vertex_count": len(arrays["vertices"]), "face_count": len(arrays["faces"]),
    }, expanded=1)
    geometry_panel(arrays, person_id)


@st.cache_data(max_entries=1, show_spinner=False)
def parse(payload: bytes) -> Bundle:
    return load_bundle(payload)


def main() -> None:
    st.set_page_config(page_title="SMPL Reconstruction Lab", layout="wide")
    st.title("SMPL Reconstruction Lab")
    with st.sidebar:
        upload = st.file_uploader("Reconstruction ZIP", type=["zip"])
        demo = st.checkbox("Use synthetic fixture", value=True, disabled=upload is not None)
        if upload is None and not demo:
            st.info("Upload a reconstruction ZIP to begin.")
            return
        try:
            with st.spinner("Validating reconstruction archive"):
                bundle = parse(upload.getvalue() if upload is not None else fixture_bytes())
        except BundleError as exc:
            st.error(str(exc))
            return
        metadata = bundle.manifest
        for key, label in (("scene_id", "Scene"), ("frame_id", "Frame"), ("camera_id", "Camera")):
            st.selectbox(label, [metadata[key]])
        person_id = st.selectbox("Person", list(bundle.people), placeholder="No detections")
        st.success(f"Validation passed · {len(bundle.people)} people")
        st.subheader("Reconstruction metadata")
        st.json({k: v for k, v in metadata.items() if k != "people"}, expanded=False)

    if metadata["data_kind"] == "synthetic":
        st.warning(SYNTHETIC_LABEL)
    else:
        st.caption(f"Imported inference · {metadata['model_name']} · {metadata['checkpoint_identifier']}")

    mode = st.radio("Visualization", ["Original RGB", "RGB + Reconstruction Overlay", "3D Human Mesh",
                                      "Mesh Wireframe", "All Detected People"],
                    index=1, horizontal=True)
    if mode in {"Original RGB", "RGB + Reconstruction Overlay"}:
        left, right = st.columns(2) if mode.startswith("RGB +") else (st.container(), None)
        with left:
            st.subheader("Original RGB Image")
            boxes = st.checkbox("Show detection boxes", value=True)
            st.image(draw_detections(bundle) if boxes else bundle.images["source.png"], width="stretch")
        if right:
            with right:
                st.subheader("HMR2 Reconstruction" if metadata["data_kind"] == "real_inference"
                             else "Synthetic Overlay")
                st.image(bundle.images["overlay.png"], width="stretch")
                if "side_view.png" in bundle.images:
                    with st.expander("Side view"):
                        st.image(bundle.images["side_view.png"], width="stretch")
    elif person_id is not None:
        st.subheader("3D Human Mesh" if metadata["data_kind"] == "real_inference" else "3D Fixture Mesh")
        axes = st.checkbox("Show coordinate axes", value=True)
        all_people = mode == "All Detected People"
        wireframe = st.checkbox("Wireframe") if all_people else mode == "Mesh Wireframe"
        st.caption(f"Vertex frame · {metadata['coordinate_system']}")
        if all_people:
            st.caption("Each detection has an independent viewport in its stored coordinate frame.")
        ids = list(bundle.people) if all_people else [person_id]
        page_size = 6
        if len(ids) > page_size:
            pages = (len(ids) + page_size - 1) // page_size
            page = st.selectbox("Mesh page", range(1, pages + 1))
            ids = ids[(page - 1) * page_size:page * page_size]
        for index in range(0, len(ids), 2):
            row = ids[index:index + 2]
            for column, current in zip(st.columns(len(row)), row):
                view_key = "/".join(str(metadata[k]) for k in ("scene_id", "frame_id", "camera_id"))
                column.plotly_chart(mesh_figure(current, bundle.people[current], wireframe, axes, view_key),
                                    key=f"mesh-{current}", width="stretch",
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "displaylogo": False})
    else:
        st.info("This package contains no detections.")
    if person_id is not None:
        show_parameters(person_id, bundle.people[person_id])


if __name__ == "__main__":
    main()

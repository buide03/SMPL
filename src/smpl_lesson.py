"""Interactive demonstrations built from one real HMR2 prediction."""

from hashlib import sha256

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.bundle_loader import Bundle
from src.mesh_viewer import comparison_figure, linked_comparison_html
from src.pose_utils import JOINTS, neutral_rotations, rotations_to_vectors, vector_to_rotation
from src.smpl_editor import configured_model
from src.smpl_forward import (
    NeutralSMPL, SMPLGeometry, check_imported_prediction, imported_local_vertices, smpl_geometry,
)
from src.validation import PersonArrays


LESSON_MODE = "Minh họa SMPL"
JOINT_NAMES = (
    "Xương chậu", "Hông trái", "Hông phải", "Cột sống 1", "Gối trái", "Gối phải", "Cột sống 2",
    "Cổ chân trái", "Cổ chân phải", "Cột sống 3", "Bàn chân trái", "Bàn chân phải", "Cổ",
    "Xương đòn trái", "Xương đòn phải", "Đầu", "Vai trái", "Vai phải", "Khuỷu tay trái",
    "Khuỷu tay phải", "Cổ tay trái", "Cổ tay phải", "Bàn tay trái", "Bàn tay phải",
)
JOINT_VIEW, SHAPE_VIEW, MOTION_VIEW = "Vị trí khớp", "Hình thể β", "Chuyển động θ"
HIGHLIGHT = "#c25212"


def descendant_joints(parents: np.ndarray, joint: int) -> set[int]:
    """SMPL lists the ancestors of each joint earlier in the hierarchy."""
    downstream = {joint}
    for index, parent in enumerate(parents):
        if parent in downstream:
            downstream.add(index)
    return downstream - {joint}


def rotation_sequence(rotations: np.ndarray, joint: int, axis: int, angle: float) -> np.ndarray:
    """Rotate about a joint-local axis on top of the imported pose, keeping other local rotations fixed."""
    progress = np.zeros((15, 3))
    progress[:, axis] = np.linspace(0, angle, len(progress))
    sequence = np.repeat(rotations[None], len(progress), axis=0)
    sequence[:, joint] = rotations[joint] @ vector_to_rotation(progress)
    return sequence


def skeleton_xyz(points: np.ndarray, parents: np.ndarray) -> np.ndarray:
    bones = np.full((len(parents) - 1, 3, 3), np.nan)
    bones[:, 0] = points[parents[1:]]
    bones[:, 1] = points[1:]
    return bones.reshape(-1, 3).T


def joint_traces(points: np.ndarray, parents: np.ndarray, selected: int,
                 downstream: set[int]) -> list[go.Scatter3d]:
    x, y, z = skeleton_xyz(points, parents)
    skeleton = go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                            line={"color": "#244b4a", "width": 4})
    x, y, z = points.T
    markers = go.Scatter3d(
        x=x, y=y, z=z, mode="markers+text", hovertext=JOINT_NAMES,
        hovertemplate="%{hovertext}<br>X %{x:.3f} · Y %{y:.3f} · Z %{z:.3f} m<extra></extra>",
        text=[name if index == selected else "" for index, name in enumerate(JOINT_NAMES)],
        textposition="top center", textfont={"color": "#8c3000", "size": 13},
        marker={"color": [HIGHLIGHT if j == selected else "#0c8970" if j in downstream else "#244b4a"
                          for j in range(len(points))],
                "size": [8 if j == selected else 5 for j in range(len(points))],
                "line": {"color": "white", "width": 1}},
    )
    return [skeleton, markers]


def show_geometry(figure: go.Figure, scope: str) -> None:
    figure.update_layout(height=445, showlegend=False)
    for scene in (figure.layout.scene, figure.layout.scene2):
        for axis in (scene.xaxis, scene.yaxis, scene.zaxis):
            axis.visible = False
    st.iframe(linked_comparison_html(figure, scope), height=453, alt="Hai mô hình 3D minh họa SMPL")


def add_motion(figure: go.Figure, geometry: SMPLGeometry, parents: np.ndarray,
               joint: int, downstream: set[int], path_joint: int | None) -> None:
    """Keep connectivity, cameras, and the left reference fixed during the playback."""
    figure.add_traces(joint_traces(geometry.joints[-1], parents, joint, downstream), rows=[1, 1], cols=[2, 2])
    right_bones, right_markers = len(figure.data) - 2, len(figure.data) - 1
    if path_joint is not None:
        x, y, z = geometry.joints[:, path_joint].T
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                     line={"color": "#0c8970", "width": 5, "dash": "dash"}), row=1, col=2)

    frames = []
    for index, (vertices, joints) in enumerate(zip(geometry.vertices, geometry.joints, strict=True)):
        x, y, z = vertices.T
        bx, by, bz = skeleton_xyz(joints, parents)
        jx, jy, jz = joints.T
        frames.append(go.Frame(name=str(index), traces=[1, right_bones, right_markers], data=[
            go.Mesh3d(x=x, y=y, z=z),
            go.Scatter3d(x=bx, y=by, z=bz),
            go.Scatter3d(x=jx, y=jy, z=jz),
        ]))
    figure.frames = frames
    for axis, dim in zip("xyz", range(3), strict=True):
        current = figure.layout.scene2[f"{axis}axis"].range
        data = geometry.vertices[:, :, dim]
        margin = (current[1] - current[0]) * 0.025
        shared = (min(current[0], float(data.min()) - margin), max(current[1], float(data.max()) + margin))
        for scene in (figure.layout.scene, figure.layout.scene2):
            scene[f"{axis}axis"].range = shared
    figure.update_layout(margin_b=72, updatemenus=[{
        "type": "buttons", "x": 0.53, "y": 0, "xanchor": "left", "yanchor": "top",
        "showactive": False, "buttons": [{"label": "▶ Phát chuyển động", "method": "animate",
            "args": [[frame.name for frame in frames], {"mode": "immediate", "fromcurrent": False,
                     "frame": {"duration": 95, "redraw": True}, "transition": {"duration": 0}}]}],
    }])


def select_joint(parents: np.ndarray, scope: str) -> int:
    joint = st.selectbox("Khớp SMPL", range(len(JOINTS)), index=JOINTS.index("Left knee"),
                         format_func=JOINT_NAMES.__getitem__, key=f"lesson-joint-{scope}")
    parent = parents[joint]
    relation = f"Khớp cha {JOINT_NAMES[parent]}" if parent >= 0 else "Khớp gốc"
    st.caption(f"Khớp {joint:02d} · {JOINTS[joint]} · {relation}")
    return joint


def joint_comparison(original: SMPLGeometry, faces: np.ndarray, parents: np.ndarray, scope: str) -> go.Figure:
    joint = select_joint(parents, scope)
    st.caption("Di chuột lên một điểm để đọc tên và tọa độ SMPL của khớp.")
    titles = ("Tư thế nghỉ", "Pose HMR2")
    for label, point in zip(titles, original.joints[:, joint], strict=True):
        values = " · ".join(f"{axis} {value:+.3f}" for axis, value in zip("XYZ", point))
        st.markdown(f"**{label}**  \n{values} m")

    left, right = original.vertices
    figure = comparison_figure(left, right, right, faces, show_original=False, titles=titles)
    for column, joints in enumerate(original.joints, start=1):
        figure.add_traces(joint_traces(joints, parents, joint, set()), rows=[1, 1], cols=[column, column])
    return figure


def shape_comparison(arrays: PersonArrays, rest: np.ndarray, base: np.ndarray,
                     model: NeutralSMPL, scope: str) -> go.Figure:
    st.caption("Hai mô hình giữ cùng tư thế nghỉ để nhìn rõ thay đổi hình thể.")
    coefficient = st.selectbox("Hệ số hình thể", range(10), format_func=lambda n: f"β{n}",
                               key=f"lesson-beta-{scope}")
    delta = st.slider("Mức thay đổi Δβ", -3.0, 3.0, 2.0, 0.1, key=f"lesson-delta-{scope}")
    betas = arrays["betas"].astype(np.float64)
    value = betas[coefficient]
    betas[coefficient] += delta
    st.markdown(f"**β{coefficient}** · {value:+.2f} → {betas[coefficient]:+.2f}")

    changed = smpl_geometry(betas, rest[None, 1:], rest[None, :1], model=model).vertices[0]
    distances = np.linalg.norm(changed - base, axis=1) * 100
    st.metric("Độ dịch chuyển đỉnh lớn nhất", f"{distances.max():.2f} cm")
    st.caption("Mỗi β là một thành phần hình thể học từ dữ liệu và có thể làm thay đổi nhiều vùng.")

    figure = comparison_figure(base, changed, base, model.faces, show_original=False,
                               posed_y_up=True, titles=("Hình thể HMR2", "Sau khi chỉnh β"))
    figure.data[1].update(color=None, intensity=distances, intensitymode="vertex", colorscale="Viridis",
                          cmin=0, cmax=max(0.01, distances.max()), showscale=True,
                          colorbar={"title": "cm", "thickness": 10, "len": 0.65}, hoverinfo="skip")
    return figure


def motion_comparison(arrays: PersonArrays, pose: np.ndarray, original: SMPLGeometry,
                      model: NeutralSMPL, parents: np.ndarray, scope: str) -> go.Figure:
    joint = select_joint(parents, scope)
    axis = st.radio("Trục cục bộ", ("X", "Y", "Z"), horizontal=True, key=f"lesson-axis-{scope}")
    angle = st.slider("Góc xoay thêm (°)", -75, 75, 45, key=f"lesson-angle-{scope}")
    st.caption("β giữ nguyên. Khớp xoay thêm quanh trục cục bộ đã chọn.")
    st.caption("Các chấm xanh ngọc là khớp con của khớp màu cam. Đây là nội suy từ một pose HMR2.")

    poses = rotation_sequence(pose, joint, "XYZ".index(axis), angle)
    with st.expander("Tham số xoay θ"):
        st.caption("Thành phần axis-angle của khớp, tính theo độ.")
        vectors = rotations_to_vectors(poses[[0, -1], joint])
        for label, vector in zip(("HMR2", "Sau khi xoay"), vectors, strict=True):
            values = " · ".join(f"{axis} {value:+.1f}°" for axis, value in zip("XYZ", vector))
            st.markdown(f"**{label}**  \n{values}")
    motion = smpl_geometry(arrays["betas"], poses[:, 1:], poses[:, :1], model=model)
    left, right = original.vertices[1], motion.vertices[-1]
    figure = comparison_figure(left, right, left, model.faces, show_original=False,
                               canonical_y_up=False, titles=("Pose HMR2", "Sau khi xoay khớp"))
    downstream = descendant_joints(parents, joint)
    shifts = np.linalg.norm(motion.joints[-1] - motion.joints[0], axis=1)
    path_joint = max(sorted(downstream), key=shifts.__getitem__, default=None)
    if path_joint is not None and shifts[path_joint] < 1e-5:
        path_joint = None
    if path_joint is not None:
        st.caption(f"Đường nét đứt theo {JOINT_NAMES[path_joint].lower()} trong lúc xoay.")

    add_motion(figure, motion, parents, joint, downstream, path_joint)
    figure.add_traces(joint_traces(original.joints[1], parents, joint, set()), rows=[1, 1], cols=[1, 1])
    return figure


def smpl_lesson(bundle: Bundle, person_id: str | None, package_id: str) -> None:
    with st.container(border=True):
        st.subheader(LESSON_MODE)
        if bundle.manifest["data_kind"] != "real_inference" or person_id is None:
            st.info("Tải ZIP kết quả HMR2 và chọn một người để mở thẻ minh họa.")
            return
        arrays = bundle.people[person_id]
        metadata = bundle.manifest
        try:
            model = configured_model()
            scope = sha256(f"{package_id}:{person_id}:{model.sha256}:lesson".encode()).hexdigest()
            baseline = st.session_state.get("smpl_lesson_baseline")
            if baseline is None or baseline[0] != scope:
                with st.spinner("Đang kiểm tra mẫu HMR2 và dựng mô hình 3D"):
                    reference = imported_local_vertices(arrays, metadata["coordinate_system"])
                    check_imported_prediction(model, arrays, reference, metadata["smpl_model_version"])
                    pose = np.concatenate((arrays["global_orient"], arrays["body_pose"])).astype(np.float64)
                    rest = neutral_rotations()
                    poses = np.stack((rest, pose))
                    geometry = smpl_geometry(arrays["betas"], poses[:, 1:], poses[:, :1], model=model)
                    baseline = (scope, pose, rest, geometry)
                    st.session_state["smpl_lesson_baseline"] = baseline
        except (ImportError, RuntimeError, ValueError) as exc:
            st.warning(f"Chưa mở được minh họa SMPL. {exc}")
            return

        _, pose, rest, original = baseline
        parents = model.layer.parents.numpy()
        with st.expander(f"{person_id} · Cảnh {metadata['scene_id']} · Khung {metadata['frame_id']} · "
                         f"{metadata['camera_id']}"):
            st.image(bundle.images["source.png"].crop(tuple(arrays["bbox_xyxy"])),
                     caption="Vùng phát hiện trong ảnh nguồn", width=140)
        view = st.segmented_control("Nội dung minh họa", (JOINT_VIEW, SHAPE_VIEW, MOTION_VIEW),
                                    default=JOINT_VIEW, key=f"lesson-view-{scope}") or JOINT_VIEW
        controls, preview = st.columns((1, 3), gap="medium")
        try:
            with controls:
                if view == SHAPE_VIEW:
                    figure = shape_comparison(arrays, rest, original.vertices[0], model, scope)
                elif view == JOINT_VIEW:
                    figure = joint_comparison(original, model.faces, parents, scope)
                else:
                    figure = motion_comparison(arrays, pose, original, model, parents, scope)
            with preview:
                if view != SHAPE_VIEW:
                    for trace in figure.data[:2]:
                        trace.update(opacity=0.30, hoverinfo="skip")
                show_geometry(figure, f"{scope}-{(JOINT_VIEW, SHAPE_VIEW, MOTION_VIEW).index(view)}")
        except (RuntimeError, ValueError) as exc:
            st.warning(f"Chưa dựng được hình minh họa. {exc}")

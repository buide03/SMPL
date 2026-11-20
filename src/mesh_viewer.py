"""Plotly rendering in the imported vertex frame."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.validation import PersonArrays


PLOTLY_CONFIG = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False}
POSED_CAMERA = {"eye": {"x": 1.3, "y": -0.5, "z": -1.8}, "up": {"x": 0, "y": -1, "z": 0}}
CANONICAL_CAMERA = {"eye": {"x": 1.3, "y": 0.5, "z": 1.8}, "up": {"x": 0, "y": 1, "z": 0}}


def solid_mesh(vertices: np.ndarray, faces: np.ndarray, *, color: str = "#249c80",
               **options) -> go.Mesh3d:
    x, y, z = np.asarray(vertices, dtype=np.float64).T
    i, j, k = np.asarray(faces, dtype=np.int32).T
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color,
                     lighting={"ambient": 0.45, "diffuse": 0.75}, **options)


def scene_axes(visible: bool) -> dict:
    return {f"{axis}axis": {
        "visible": visible, "title": axis.upper(), "showbackground": False,
        "gridcolor": "#c6d8d4", "zerolinecolor": "#93ada7",
    } for axis in "xyz"}


def mesh_figure(person_id: str, arrays: PersonArrays, wireframe: bool,
                axes: bool, view_key: str) -> go.Figure:
    """Render stored vertices with common scale on all three axes."""
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int32)
    if wireframe:
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        segments = np.full((len(edges), 3, 3), np.nan)
        segments[:, :2] = vertices[edges]
        x, y, z = segments.reshape(-1, 3).T
        trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", connectgaps=False,
                             line={"color": "#096750", "width": 2}, hoverinfo="skip")
    else:
        trace = solid_mesh(vertices, faces)
    figure = go.Figure(trace)
    figure.update_layout(
        title={"text": person_id, "x": 0.05, "font": {"size": 14}},
        height=390,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        paper_bgcolor="#f2f7f6",
        font={"color": "#25413d"},
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": POSED_CAMERA,
            **scene_axes(axes),
        },
        uirevision=view_key,
    )
    return figure


def comparison_figure(
    canonical: np.ndarray,
    posed: np.ndarray,
    imported: np.ndarray,
    faces: np.ndarray,
    *,
    show_original: bool,
    posed_y_up: bool = False,
) -> go.Figure:
    """Use shared axis scales while keeping the canonical and posed vertices in native frames."""
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                           subplot_titles=("Canonical Mesh", "Posed Mesh"), horizontal_spacing=0.05)
    figure.add_trace(solid_mesh(canonical, faces, showlegend=False), row=1, col=1)
    figure.add_trace(solid_mesh(posed, faces, name="Current", showlegend=True), row=1, col=2)
    if show_original:
        figure.add_trace(solid_mesh(imported, faces, color="#6f7598", name="HMR2 prediction",
                                    showlegend=True, opacity=0.28), row=1, col=2)
    # Align the two range calculations in the posed display convention. Vertices are unchanged.
    camera_frames_differ = not posed_y_up
    display_canonical = canonical * [1, -1, -1] if camera_frames_differ else canonical
    displayed = (display_canonical, posed, imported) if show_original else (display_canonical, posed)
    ranges = np.concatenate(displayed)
    lower, upper = ranges.min(axis=0), ranges.max(axis=0)
    margin = max(float(np.max(upper - lower)) * 0.06, 0.02)
    display_ranges = np.column_stack((lower - margin, upper + margin))
    posed_camera = CANONICAL_CAMERA if posed_y_up else POSED_CAMERA
    for scene, camera, canonical_view in ((figure.layout.scene, CANONICAL_CAMERA, True),
                                          (figure.layout.scene2, posed_camera, False)):
        scene.update(aspectmode="data", dragmode="orbit", camera=camera, **scene_axes(True))
        for index, axis in enumerate("xyz"):
            axis_range = display_ranges[index]
            if camera_frames_differ and canonical_view and axis in "yz":
                axis_range = -axis_range[::-1]
            scene[f"{axis}axis"].range = axis_range.tolist()
    figure.update_layout(
        height=425, margin={"l": 4, "r": 4, "t": 54, "b": 30},
        meta={"camera_frames_differ": camera_frames_differ},
        paper_bgcolor="#f2f7f6", font={"color": "#25413d"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 0, "yanchor": "top"},
    )
    return figure


def linked_comparison_html(figure: go.Figure, view_key: str) -> str:
    """Embed the packaged Plotly runtime and synchronize the two camera controls in the browser."""
    script = Path(__file__).with_name("linked_mesh_camera.js").read_text(encoding="utf-8")
    sign = "-1" if figure.layout.meta["camera_frames_differ"] else "1"
    script = script.replace("{linked_yz_sign}", sign)
    chart = figure.to_html(full_html=False, include_plotlyjs=True, config=PLOTLY_CONFIG,
                           div_id=f"smpl-editor-{view_key}", post_script=script)
    return "<style>body { margin: 0; background: #f2f7f6; }</style>" + chart

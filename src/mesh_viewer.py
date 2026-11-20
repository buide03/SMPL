"""Plotly rendering in the imported vertex frame."""
import numpy as np
import plotly.graph_objects as go

from src.validation import PersonArrays


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
        x, y, z = vertices.T
        i, j, k = faces.T
        trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#249c80",
                          lighting={"ambient": 0.45, "diffuse": 0.75})
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
            "camera": {"eye": {"x": 1.3, "y": -0.5, "z": -1.8},
                       "up": {"x": 0, "y": -1, "z": 0}},
            **{f"{axis}axis": {
                "visible": axes, "title": axis.upper(), "showbackground": False,
                "gridcolor": "#c6d8d4", "zerolinecolor": "#93ada7",
            } for axis in "xyz"},
        },
        uirevision=view_key,
    )
    return figure

"""Plotly rendering in the imported vertex frame."""
import numpy as np
import plotly.graph_objects as go


def mesh_figure(person_id: str, arrays: dict[str, np.ndarray], wireframe: bool,
                axes: bool, view_key: str) -> go.Figure:
    vertices, faces = arrays["vertices"], arrays["faces"]
    if wireframe:
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        segments = np.full((len(edges), 3, 3), np.nan)
        segments[:, :2] = vertices[edges.astype(np.intp)]
        x, y, z = segments.reshape(-1, 3).T
        trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", connectgaps=False,
                             line={"color": "#137b91", "width": 2}, hoverinfo="skip")
    else:
        x, y, z = vertices.T
        i, j, k = faces.astype(np.int32).T
        trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#169ab1",
                          lighting={"ambient": .45, "diffuse": .75})
    figure = go.Figure(trace)
    figure.update_layout(
        title=person_id, height=430, margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={"aspectmode": "data", "dragmode": "orbit",
               "camera": {"eye": {"x": 1.3, "y": -.5, "z": -1.8},
                          "up": {"x": 0, "y": -1, "z": 0}},
               **{f"{axis}axis": {"visible": axes, "title": axis.upper()} for axis in "xyz"}},
        uirevision=view_key,
    )
    return figure

"""Optional Open3D geometry inspection and OBJ/PLY export."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from open3d.geometry import TriangleMesh


def create_mesh(vertices: np.ndarray, faces: np.ndarray) -> TriangleMesh:
    """Build an Open3D TriangleMesh and compute vertex normals."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    return mesh.compute_vertex_normals()


def mesh_statistics(mesh: TriangleMesh) -> dict[str, int | float | bool | list[float]]:
    """Summarize geometry and report individual topology checks."""
    bounds = mesh.get_axis_aligned_bounding_box()
    triangles = np.asarray(mesh.triangles)
    points = np.asarray(mesh.vertices)[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    return {
        "vertices": len(mesh.vertices), "faces": len(mesh.triangles),
        "surface_area": mesh.get_surface_area(),
        "bounds_min": bounds.get_min_bound().tolist(),
        "bounds_max": bounds.get_max_bound().tolist(),
        "bounds_extent": bounds.get_extent().tolist(),
        "zero_area_faces": int(np.count_nonzero(np.all(cross == 0, axis=1))),
        "duplicate_faces": len(triangles) - len(np.unique(np.sort(triangles, axis=1), axis=0)),
        "edge_manifold": mesh.is_edge_manifold(),
        "vertex_manifold": mesh.is_vertex_manifold(),
        "self_intersecting": mesh.is_self_intersecting(),
        "watertight": mesh.is_watertight(),
        "orientable": mesh.is_orientable(),
    }


def export_mesh(mesh: TriangleMesh, extension: str) -> bytes:
    """Serialize to OBJ or PLY using a private temporary file."""
    import open3d as o3d

    if extension not in {"obj", "ply"}:
        raise ValueError("Mesh export must be OBJ or PLY")
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"mesh.{extension}"
        if not o3d.io.write_triangle_mesh(str(path), mesh, write_triangle_uvs=False):
            raise OSError(f"Open3D could not export {extension.upper()}")
        return path.read_bytes()

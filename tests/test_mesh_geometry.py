import numpy as np
import pytest

from src.bundle_loader import load_bundle
from src.mesh_geometry import create_mesh, export_mesh, mesh_statistics
from src.synthetic_fixture import fixture_bytes


def test_tetrahedron_geometry_and_roundtrip(tmp_path):
    open3d = pytest.importorskip("open3d")
    arrays = load_bundle(fixture_bytes()).people["person_000"]
    mesh = create_mesh(arrays["vertices"], arrays["faces"])
    np.testing.assert_allclose(np.linalg.norm(mesh.vertex_normals, axis=1), 1)
    stats = mesh_statistics(mesh)
    assert stats["watertight"] and stats["orientable"]
    assert not stats["self_intersecting"]
    assert stats["zero_area_faces"] == 0
    assert stats["surface_area"] > 0
    assert stats["bounds_extent"] == [2, 2, 1]
    for extension in ("obj", "ply"):
        path = tmp_path / f"mesh.{extension}"
        path.write_bytes(export_mesh(mesh, extension))
        loaded = open3d.io.read_triangle_mesh(str(path))
        np.testing.assert_allclose(loaded.vertices, mesh.vertices)
        np.testing.assert_array_equal(loaded.triangles, mesh.triangles)

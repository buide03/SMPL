"""Generate a reproducible visual test package with two tetrahedra."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from src.bundle_loader import SYNTHETIC_LABEL


def draw_fixture_image(name: str, size: tuple[int, int],
                       meshes: list[tuple[np.ndarray, np.ndarray]]) -> Image.Image:
    """Draw a synthetic source and mesh projections with an embedded provenance label."""
    width, height = size
    image = Image.new("RGB", size, "#132e2a")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 24):
        draw.line((x, 48, x, height - 48), fill="#234742")
    for y in range(48, height - 47, 24):
        draw.line((0, y, width, y), fill="#234742")
    draw.line((0, 286, width, 286), fill="#41665e", width=2)
    prefix, suffix = SYNTHETIC_LABEL.split(" — ")
    label_color = "#d2fff1"
    draw.text((20, 14), prefix, fill=label_color, font_size=20)
    dash_x = 27 + draw.textlength(prefix, font_size=20)
    draw.line((dash_x, 28, dash_x + 16, 28), fill=label_color, width=2)
    draw.text((dash_x + 23, 14), suffix, fill=label_color, font_size=20)
    labels = {
        "source.png": "REFERENCE DRAWING",
        "overlay.png": "MESH PROJECTION · FRONT",
        "side_view.png": "MESH PROJECTION · SIDE",
    }
    draw.text((20, height - 32), labels[name], fill="#b0d1c7", font_size=16)
    for center, (vertices, faces) in zip((171, 470), meshes):
        horizontal, depth = (2, 0) if name == "side_view.png" else (0, 2)
        points = vertices[:, [horizontal, 1]] * 70 + [center, 180]
        if name == "source.png":
            draw.polygon([tuple(point) for point in points[:3]], fill="#294d46", outline="#74978c",
                         width=2)
            draw.ellipse((center - 3, 177, center + 3, 183), fill="#aac8bf")
            continue
        order = np.argsort(vertices[faces, depth].mean(axis=1))
        fills = ("#22685a", "#419583", "#54bda6", "#7ae2c8")
        for rank, index in enumerate(order):
            draw.polygon([tuple(point) for point in points[faces[index]]], fill=fills[rank],
                         outline="#c8faec", width=2)
    return image


def fixture_bytes() -> bytes:
    """Build an archive with placeholder parameters and simple closed meshes."""
    width, height = 640, 360
    result = BytesIO()

    def write(archive: ZipFile, name: str, payload: bytes) -> None:
        archive.writestr(ZipInfo(name), payload, compress_type=ZIP_DEFLATED)

    manifest = {
        "schema_version": 1, "scene_id": "fixture", "frame_id": "000",
        "camera_id": "drawing", "model_name": "synthetic tetrahedra",
        "checkpoint_identifier": "none", "image_width": width, "image_height": height,
        "coordinate_system": "synthetic", "smpl_model_version": "none",
        "data_kind": "synthetic", "people": [],
    }
    base_vertices = np.array([[-1., 1., 0.], [1., 1., 0.], [0., -1., 0.], [0., .5, 1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    meshes = []
    with ZipFile(result, "w") as archive:
        for index, (center, scale) in enumerate(((171, 1.), (470, 1.4))):
            person_id = f"person_{index:03d}"
            path = f"people/{person_id}.npz"
            manifest["people"].append({"person_id": person_id, "npz_path": path})
            values = BytesIO()
            vertices = base_vertices * scale
            meshes.append((vertices, faces))
            radius = 70 * scale + 12
            np.savez(values, betas=np.zeros(10), global_orient=np.eye(3)[None],
                     body_pose=np.repeat(np.eye(3)[None], 23, axis=0),
                     vertices=vertices, faces=faces,
                     pred_cam=np.array([1., 0., 0.]), pred_cam_t=np.zeros(3),
                     bbox_xyxy=np.array([center - radius, 180 - radius,
                                         center + radius, 180 + radius]))
            write(archive, path, values.getvalue())
        for name in ("source.png", "overlay.png", "side_view.png"):
            encoded = BytesIO()
            draw_fixture_image(name, (width, height), meshes).save(encoded, format="PNG")
            write(archive, name, encoded.getvalue())
        write(archive, "manifest.json", json.dumps(manifest, indent=2).encode())
    return result.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/fixtures/synthetic.zip"))
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(fixture_bytes())
    print(output)

"""Generate a reproducible visual test package with two tetrahedra."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from src.bundle_loader import SYNTHETIC_LABEL


def fixture_bytes() -> bytes:
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
    with ZipFile(result, "w") as archive:
        for index, (left, size) in enumerate(((100, 1.), (365, 1.4))):
            person_id = f"person_{index:03d}"
            path = f"people/{person_id}.npz"
            manifest["people"].append({"person_id": person_id, "npz_path": path})
            values = BytesIO()
            np.savez(values, betas=np.zeros(10), global_orient=np.eye(3)[None],
                     body_pose=np.repeat(np.eye(3)[None], 23, axis=0),
                     vertices=size * np.array([[-1., 1., 0.], [1., 1., 0.],
                                               [0., -1., 0.], [0., .5, 1.]]),
                     faces=np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]),
                     pred_cam=np.array([1., 0., 0.]), pred_cam_t=np.zeros(3),
                     bbox_xyxy=np.array([left, 95., left + 150, 290.]))
            write(archive, path, values.getvalue())
        for name in ("source.png", "overlay.png", "side_view.png"):
            image = Image.new("RGB", (width, height), "#112737")
            draw = ImageDraw.Draw(image)
            draw.text((25, 25), SYNTHETIC_LABEL, fill="white", font_size=21)
            if name == "overlay.png":
                for left in (100, 365):
                    draw.polygon([(left, 280), (left + 75, 100), (left + 150, 280)], fill="#21a7b8")
            if name == "side_view.png":
                draw.polygon([(165, 280), (165, 100), (275, 250)], fill="#21a7b8")
            draw.text((25, 320), name, fill="white", font_size=18)
            encoded = BytesIO()
            image.save(encoded, format="PNG")
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

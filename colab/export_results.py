"""Convert HMR2 predictions to the Phase 1 ZIP contract."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.bundle_loader import JsonValue, load_bundle
from src.validation import PersonArrays


@dataclass(frozen=True)
class Prediction:
    arrays: PersonArrays
    detector_score: float
    full_frame_translation: np.ndarray


def predictions_from_batch(
    smpl: Mapping[str, np.ndarray],
    vertices: np.ndarray,
    faces: np.ndarray,
    crop_camera: np.ndarray,
    crop_translation: np.ndarray,
    full_frame_translation: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> list[Prediction]:
    """Keep the native SMPL and crop-camera outputs associated with each detection."""
    return [
        Prediction(
            arrays={
                "betas": beta,
                "global_orient": root,
                "body_pose": pose,
                "vertices": points,
                "faces": faces,
                "pred_cam": camera,
                "pred_cam_t": crop_t,
                "bbox_xyxy": box,
            },
            detector_score=float(score),
            full_frame_translation=full_t,
        )
        for beta, root, pose, points, camera, crop_t, full_t, box, score in zip(
            smpl["betas"], smpl["global_orient"], smpl["body_pose"], vertices,
            crop_camera, crop_translation, full_frame_translation, boxes, scores,
            strict=True,
        )
    ]


def write_reconstruction_zip(
    destination: Path,
    *,
    metadata: Mapping[str, JsonValue],
    source: Image.Image,
    overlay: Image.Image,
    side_view: Image.Image,
    predictions: Sequence[Prediction],
) -> Path:
    """Validate the complete package with the viewer's loader before publishing it."""
    if destination.exists():
        raise FileExistsError(f"Choose a new ZIP path: {destination}")
    if not 1 <= len(predictions) <= 64:
        raise ValueError("A Phase 2 ZIP requires between 1 and 64 predictions.")
    people: list[JsonValue] = []
    manifest: dict[str, JsonValue] = {
        **metadata,
        "schema_version": 1,
        "image_width": source.width,
        "image_height": source.height,
        "coordinate_system": "smpl_local",
        "data_kind": "real_inference",
        "people": people,
    }
    # PNG and NPZ payloads fit within the viewer's limits at PandaSet resolution.
    # Stored members also avoid rejected compression ratios for low-entropy arrays.
    with NamedTemporaryFile(dir=destination.parent, prefix=".hmr2-", suffix=".tmp",
                            delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with ZipFile(staged, "w") as archive:
            for index, prediction in enumerate(predictions):
                person_id = f"person_{index:03d}"
                path = f"people/{person_id}.npz"
                people.append({
                    "person_id": person_id,
                    "npz_path": path,
                    "detector_score": prediction.detector_score,
                    "full_frame_cam_t": prediction.full_frame_translation.tolist(),
                })
                buffer = BytesIO()
                np.savez(buffer, **prediction.arrays)
                archive.writestr(path, buffer.getvalue())
            for path, image in (("source.png", source), ("overlay.png", overlay),
                                ("side_view.png", side_view)):
                image_buffer = BytesIO()
                image.convert("RGB").save(image_buffer, format="PNG")
                archive.writestr(path, image_buffer.getvalue())
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                                          allow_nan=False, indent=2))
        load_bundle(staged.read_bytes())
        if destination.exists():
            raise FileExistsError(f"Choose a new ZIP path: {destination}")
        staged.rename(destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination

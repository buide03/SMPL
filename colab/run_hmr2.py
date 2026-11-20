"""Run pretrained 4D-Humans on one PandaSet image in Colab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
import os
from pathlib import Path
import platform
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from PIL import Image

from colab.export_results import Prediction, predictions_from_batch, write_reconstruction_zip
from colab.hmr2_assets import (
    CHECKPOINT_ARCHIVE, CHUMPY_REVISION, REVISION, SMPL_SOURCE_NAME, prepare_checkpoint,
    sha256_file,
)


DETECTOR_NAME = "torchvision Faster R-CNN ResNet-50 FPN v2 COCO_V1"

if TYPE_CHECKING:
    from hmr2.utils.renderer import Renderer


def select_people(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    person_label: int,
    threshold: float,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Filter the detector's person class and clip boxes to source-image bounds."""
    selected = ((labels == person_label) & np.isfinite(scores) & (scores >= threshold)
                & np.isfinite(boxes).all(axis=1))
    width, height = size
    clipped = np.clip(boxes[selected], [0, 0, 0, 0], [width, height, width, height])
    scores = scores[selected]
    usable = (clipped[:, 0] < clipped[:, 2]) & (clipped[:, 1] < clipped[:, 3])
    clipped, scores = clipped[usable], scores[usable]
    order = np.argsort(-scores, kind="stable")
    return clipped[order].astype(np.float32), scores[order].astype(np.float32)


def render_images(
    renderer: Renderer,
    source: Image.Image,
    predictions: Sequence[Prediction],
    focal_length: float,
    crop_size: int,
) -> tuple[Image.Image, Image.Image]:
    """Render a full-frame overlay and a tile of independent 90-degree side views."""
    import torch

    mesh_color = (0.40, 0.78, 0.66)
    rgba = renderer.render_rgba_multiple(
        [person.arrays["vertices"] for person in predictions],
        cam_t=[person.full_frame_translation for person in predictions],
        render_res=source.size,
        focal_length=focal_length,
        mesh_base_color=mesh_color,
    )
    foreground = rgba[:, :, 3:]
    if not np.isfinite(rgba).all() or not np.any(foreground > 0):
        raise RuntimeError("The full-frame render contains no visible mesh.")
    composite = np.asarray(source, dtype=np.float32) * (1 - foreground)
    composite += rgba[:, :, :3] * foreground * 255
    overlay = Image.fromarray(np.clip(np.rint(composite), 0, 255).astype(np.uint8))

    columns = min(4, len(predictions))
    rows = (len(predictions) + columns - 1) // columns
    side_view = Image.new("RGB", (columns * crop_size, rows * crop_size), "white")
    blank = torch.zeros((3, crop_size, crop_size), dtype=torch.float32)
    for index, person in enumerate(predictions):
        rgba = renderer(
            person.arrays["vertices"], person.arrays["pred_cam_t"].copy(), blank,
            side_view=True, rot_angle=90, mesh_base_color=mesh_color, return_rgba=True,
        )
        alpha = rgba[:, :, 3:]
        if not np.isfinite(rgba).all() or not np.any(alpha > 0):
            raise RuntimeError(f"The side view for person_{index:03d} contains no visible mesh.")
        pixels = np.rint((rgba[:, :, :3] * alpha + 1 - alpha) * 255)
        side_view.paste(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)),
                        ((index % columns) * crop_size, (index // columns) * crop_size))
    return overlay, side_view


def run(args: argparse.Namespace) -> Path:
    import torch
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Google Colab.")
    if args.smpl_model.name != SMPL_SOURCE_NAME or not args.smpl_model.is_file():
        raise FileNotFoundError(f"Supply your authorized {SMPL_SOURCE_NAME} file.")
    if not args.source.is_file():
        raise FileNotFoundError(f"PandaSet input image was not found: {args.source}")
    if not 0.0 < args.threshold <= 1.0 or not 1 <= args.min_people <= 64:
        raise ValueError("Use a score threshold in (0, 1] and a minimum of 1–64 people.")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("Use a batch size between 1 and 8.")

    # Set the platform before HMR2 imports pyrender and before resolving its cache.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import convert_pkl, load_hmr2
    from hmr2.utils import recursive_to
    from hmr2.utils.renderer import Renderer, cam_crop_to_full

    device = torch.device("cuda")
    print(f"CUDA GPU: {torch.cuda.get_device_name(device)}", flush=True)
    with Image.open(args.source) as image:
        source = image.convert("RGB")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
    with torch.inference_mode():
        result = detector([weights.transforms()(source).to(device)])[0]
    boxes, scores = select_people(
        result["boxes"].cpu().numpy(), result["labels"].cpu().numpy(),
        result["scores"].cpu().numpy(), person_label=weights.meta["categories"].index("person"),
        threshold=args.threshold, size=source.size,
    )
    del detector, result
    torch.cuda.empty_cache()
    print(f"People detected above {args.threshold:.2f}: {len(boxes)}", flush=True)
    if len(boxes) < args.min_people:
        raise RuntimeError(f"At least {args.min_people} people are required for this run. "
                           "Inspect the frame and choose a different frame or score threshold.")
    if len(boxes) > 64:
        raise RuntimeError("The viewer accepts at most 64 detections per frame. Adjust the threshold.")

    cache = Path(CACHE_DIR_4DHUMANS)
    checkpoint = prepare_checkpoint(cache)
    model_digest = sha256_file(args.smpl_model)
    staged_model = cache / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    staged_model.parent.mkdir(parents=True, exist_ok=True)
    # The user-supplied, authorized SMPLify asset is a legacy Python 2 pickle.
    # Use the upstream converter only on this explicitly selected local model.
    convert_pkl(str(args.smpl_model), str(staged_model))
    # Only the official archive checkpoint needs Lightning's legacy pickle loader.
    with patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}):
        model, cfg = load_hmr2(str(checkpoint))
    model.eval().to(device)
    faces = np.asarray(model.smpl.faces, dtype=np.int32)
    print("Official HMR2.0b checkpoint loaded. Reconstructing people.", flush=True)

    bgr = np.ascontiguousarray(np.asarray(source)[:, :, ::-1])
    dataset = ViTDetDataset(cfg, bgr, boxes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    focal_length = float(cfg.EXTRA.FOCAL_LENGTH) / cfg.MODEL.IMAGE_SIZE * max(source.size)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            indices = batch["personid"].numpy()
            batch = recursive_to(batch, device)
            output = model(batch)
            full_frame_t = cam_crop_to_full(
                output["pred_cam"], batch["box_center"].float(), batch["box_size"].float(),
                batch["img_size"].float(), focal_length,
            )
            translations = full_frame_t.cpu().numpy()
            if not np.isfinite(translations).all() or np.any(translations[:, 2] <= 0):
                raise RuntimeError("HMR2 returned an invalid full-frame camera translation.")
            predictions.extend(predictions_from_batch(
                {name: value.cpu().numpy() for name, value in output["pred_smpl_params"].items()},
                output["pred_vertices"].cpu().numpy(), faces,
                output["pred_cam"].cpu().numpy(), output["pred_cam_t"].cpu().numpy(),
                translations, boxes[indices], scores[indices],
            ))
    renderer = Renderer(cfg, faces=faces)
    overlay, side_view = render_images(renderer, source, predictions, focal_length, cfg.MODEL.IMAGE_SIZE)

    metadata = {
        "scene_id": args.scene_id,
        "frame_id": args.frame_id,
        "camera_id": args.camera_id,
        "model_name": "4D-Humans HMR2.0b",
        "checkpoint_identifier": f"{checkpoint.name} sha256:{sha256_file(checkpoint)}",
        "checkpoint_archive": CHECKPOINT_ARCHIVE,
        "hmr2_revision": REVISION,
        "chumpy_revision": CHUMPY_REVISION,
        "smpl_model_version": f"SMPL neutral v1.0.0 sha256:{model_digest}",
        "source": {
            "dataset": "PandaSet",
            "attribution": "PandaSet by Scale AI and Hesai Photonics Technology",
            "image_name": args.source.name,
            "sha256": sha256_file(args.source),
        },
        "detector": {"name": DETECTOR_NAME, "weights_url": weights.url, "score_threshold": args.threshold},
        "full_frame_focal_length_px": focal_length,
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            **{package: version(package) for package in (
                "numpy", "pytorch-lightning", "smplx", "chumpy", "pyrender",
                "PyOpenGL", "trimesh", "timm", "scikit-image", "opencv-python", "pillow",
            )},
        },
    }
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / "reconstruction.zip"
    write_reconstruction_zip(destination, metadata=metadata, source=source, overlay=overlay,
                             side_view=side_view, predictions=predictions)
    print(f"Validated ZIP: {destination} ({len(predictions)} people)", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-people", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

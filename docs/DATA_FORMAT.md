# Reconstruction ZIP v1

One ZIP represents one scene, one frame, and one RGB camera. Place these files at the root.

| Path | Content |
| --- | --- |
| manifest.json | UTF-8 JSON metadata with unique keys and finite numeric values |
| source.png | Source RGB image |
| overlay.png | Full-frame reconstruction overlay with the dimensions of source.png |
| side_view.png | Optional side-view render with independent dimensions |
| people/person_000.npz | One NumPy archive per detection |

| Manifest field | Contract |
| --- | --- |
| schema_version | Integer 1 |
| scene_id | Nonempty source scene identifier |
| frame_id | Nonempty frame identifier, preserving leading zeros |
| camera_id | Nonempty source camera identifier |
| model_name | Inference model name |
| checkpoint_identifier | Reproducible checkpoint reference, ideally a digest |
| image_width, image_height | Positive integer source image dimensions in pixels, up to 8192 per axis and 12 million total pixels |
| coordinate_system | `smpl_local`, `hmr2_crop_camera` or `synthetic`, per the [coordinate contract](COORDINATE_SYSTEMS.md) |
| smpl_model_version | SMPL asset version or immutable identifier. Use `none` for synthetic geometry |
| data_kind | `real_inference` or `synthetic`. The exporter supplies this provenance declaration |
| people | Zero to 64 records containing unique `person_id` and `npz_path` strings |

Person IDs use ASCII letters, digits, underscores, and hyphens (1–64 chars). NPZ paths use those characters for the basename under `people/`. Additional metadata can record exporter or dependency versions. The manifest declares provenance and coordinate conventions. Archive validation checks the data contract. Confirming inference provenance requires access to the export process.

The Phase 2 exporter saves the checkpoint archive URL, 4D-Humans and chumpy revisions, runtime dependency versions, detector identity and threshold, source image name and SHA-256, and the model-implied full-image focal length in pixels. Each person record includes the detector score and `full_frame_cam_t` in meters, which is used to place that person's mesh in the composite. The side-view image follows the order of the person records from left to right and top to bottom. The record order uses descending detector score.

| Array | Shape | Meaning |
| --- | --- | --- |
| betas | (10,) | Dimensionless SMPL shape coefficients |
| global_orient | (1, 3, 3) | Dimensionless root rotation matrix |
| body_pose | (23, 3, 3) | Dimensionless parent-relative rotation matrices in SMPL joint order |
| vertices | (6890, 3) | XYZ positions in meters for SMPL predictions |
| faces | (13776, 3) | Zero-based triangle vertex indices |
| pred_cam | (3,) | HMR2 crop weak-perspective parameters in [scale, tx, ty] order. Scale is dimensionless and tx/ty carry through to crop translation in model units |
| pred_cam_t | (3,) | HMR2 perspective crop translation [tx, ty, tz] in model units (meters) |
| bbox_xyxy | (4,) | Source-image box [left, top, right, bottom] in pixels, with top-left origin. Clip to image bounds before export |

All numeric arrays use finite floating-point values with element widths up to 64 bits, except faces which use signed or unsigned integers up to 64 bits. Rotations must satisfy RᵀR ≈ I and det(R) ≈ +1 with absolute tolerance 1e-4 and zero relative tolerance. Faces must have distinct in-range indices. The loader checks dimensions and indices. Optional Open3D analysis reports topology properties such as manifoldness and watertightness.

Synthetic data uses arbitrary geometry units, with 1–6890 vertices and 1–13776 faces. Rotation and camera values in the fixture are placeholders.

ZIP limits are 64 MiB upload, 96 MiB declared outer expansion, 72 outer entries, 64 KiB manifest, 12 million pixels per PNG, and 2 MiB per NPZ for both the archived file and nested expansion. Supported compression methods are stored and deflated. The loader rejects unsafe paths, case-insensitive collisions, Windows device names, links, encrypted entries, and expansion ratios above 2000:1. It checks NPY headers (v1 or v2) for the expected dtype, shape, and data length before loading the required arrays with pickle disabled.

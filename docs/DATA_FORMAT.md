# Reconstruction ZIP v1

One ZIP represents one scene, one frame, and one RGB camera. Place these files at the root.

| Path | Content |
| --- | --- |
| manifest.json | UTF-8 JSON metadata |
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
| image_width, image_height | Positive integer source image dimensions in pixels |
| coordinate_system | `smpl_local`, `hmr2_crop_camera` or `synthetic`, per the [coordinate contract](COORDINATE_SYSTEMS.md) |
| smpl_model_version | SMPL asset version or immutable identifier. Use `none` for synthetic geometry |
| data_kind | `real_inference` or `synthetic`. This provenance field is supplied by the exporter |
| people | Zero to 64 records containing unique `person_id` and `npz_path` strings |

Person IDs use ASCII letters, digits, underscores, and hyphens (1–64 chars). NPZ paths use those characters for the basename under `people/`.

| Array | Shape | Meaning |
| --- | --- | --- |
| betas | (10,) | Dimensionless SMPL shape coefficients |
| global_orient | (1, 3, 3) | Dimensionless root rotation matrix |
| body_pose | (23, 3, 3) | Dimensionless parent-relative rotation matrices in SMPL joint order |
| vertices | (6890, 3) | XYZ positions in meters for SMPL predictions |
| faces | (13776, 3) | Zero-based triangle vertex indices |
| pred_cam | (3,) | HMR2 crop weak-perspective parameters in [scale, tx, ty] order, in the model's native convention |
| pred_cam_t | (3,) | HMR2 perspective crop translation [tx, ty, tz] in model units (meters) |
| bbox_xyxy | (4,) | Source-image box [left, top, right, bottom] in pixels, with top-left origin. Clip to image bounds before export |

All numeric arrays use finite floating-point values with element widths up to 64 bits, except faces which use signed or unsigned integers up to 64 bits. Rotations must satisfy RᵀR ≈ I and det(R) ≈ +1 with absolute tolerance 1e-4 and zero relative tolerance. Faces must have distinct in-range indices. The loader checks dimensions and indices. Optional Open3D analysis reports topology properties such as manifoldness and watertightness.

Synthetic data uses arbitrary geometry units, with 1–6890 vertices and 1–13776 faces. Rotation and camera values in the fixture are placeholders.

ZIP limits are 64 MiB upload, 96 MiB declared outer expansion, 72 outer entries, 64 KiB manifest, 12 million pixels per PNG, and 2 MiB per NPZ for both the stored file and nested expansion. Only stored and deflated entries are supported. The loader rejects unsafe paths, case-insensitive collisions, encryption, symlinks and expansion ratios above 2000:1. It checks nested NPY headers (v1 or v2) and loads with pickle disabled.

# Coordinate systems

- `smpl_local` stores HMR2 `pred_vertices` with the predicted root orientation and body pose applied. Crop camera translation is stored separately as `pred_cam_t`.
- `hmr2_crop_camera` stores vertices with `pred_cam_t` already added once. Each detection is in its own crop camera frame.
- `synthetic` stores fixture geometry in arbitrary units.

Plotly and Open3D use stored XYZ values directly. The viewer places each detection in its own viewport. HMR2 projects with X right, Y down, and +Z into the scene in its perspective camera convention. The initial Plotly camera uses −Y up while keeping the input axes intact. Read [HMR2 forward](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/hmr2.py) and the [official rendering demo](https://github.com/shubham-goel/4D-Humans/blob/main/demo.py).

The official demo uses crop center, crop size, image size and focal length to compute a separate full-frame translation for compositing. Metric placement into PandaSet world coordinates requires verified camera calibration and a justified depth or alignment estimate.

# SMPL Reconstruction Lab

The Windows CPU viewer inspects one scene, frame, and camera per reconstruction ZIP. Streamlit presents source RGB, detection boxes, overlay, side view, and per-person SMPL parameters. Plotly renders the mesh in its stored vertex frame. Optional Open3D utilities inspect geometry and export OBJ or PLY. Phase 2 adds a Colab GPU workflow that runs pretrained HMR2 on a PandaSet image and exports a ZIP for this viewer.

## Run on Windows

Install 64-bit Python 3.12 for the viewer and optional Open3D. For the viewer alone, Python 3.13 x64 also works.

```powershell
git clone https://github.com/buide03/SMPL.git
Set-Location SMPL
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The built-in fixture displays two synthetic tetrahedra and placeholder parameters. Upload [`data/fixtures/synthetic.zip`](data/fixtures/synthetic.zip) to exercise the importer. This command regenerates the fixture.

```powershell
.\.venv\Scripts\python.exe -m src.synthetic_fixture
```

For an exported prediction, choose **Reconstruction ZIP** and follow the [ZIP contract](docs/DATA_FORMAT.md). Pick a detection in the sidebar. The default view presents the source, overlay, selected mesh and parameter panels. In **Pose θ**, choose the pelvis or one of the 23 named body joints to read its matrix. **All Detected People** shows separate viewports, with four per page for larger packages. Drag to orbit, scroll over the graph to zoom, and use the mode bar to pan or reset.

## Run HMR2 in Colab

1. Open the [Phase 2 notebook in Colab](https://colab.research.google.com/github/buide03/SMPL/blob/main/colab/hmr2_inference.ipynb). In **Runtime → Change runtime type**, select a GPU and runtime version **2026.07 (Python 3.12)**. The notebook reuses Colab's matched CUDA builds of PyTorch and TorchVision. [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)
2. Run the first cell and authorize the Drive mount. The default paths select `006_front_camera_03.jpg` with IDs `006`, `03`, and `front_camera`. Set the paths and IDs in that cell when using another image. Supply the SMPLify neutral model you obtained through the official download process as `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. A single RGB image is sufficient. [4D-Humans setup](https://github.com/shubham-goel/4D-Humans#installation-and-setup)
3. Run the setup, inference, and preview cells in order. The first run downloads the official 2.71 GB HMR2.0b archive. The default score threshold is `0.50` and the run requires at least two detections. The notebook displays the overlay and side views, then syncs a validated `reconstruction.zip` to a unique folder under `reconstruction-results` in Drive.
4. Download `reconstruction.zip` and import it in Streamlit. Check the selected scene, frame, and camera, then inspect the overlay and meshes for at least two detections.

The detector is TorchVision Faster R-CNN ResNet-50 FPN v2 with COCO weights. Crop preprocessing, HMR2 forward, camera conversion, and rendering use the official 4D-Humans code. Colab records the checkpoint and asset SHA-256 digests, the upstream revision, the detector weights, and installed inference versions in the ZIP manifest. [TorchVision detector](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html) · [4D-Humans demo](https://github.com/shubham-goel/4D-Humans/blob/main/demo.py)

## Geometry and tests

Install the geometry requirements to enable Open3D. Run pytest with the same interpreter that launches Streamlit.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-geometry.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

In the app, expand **Open3D geometry and export** and enable analysis for the selected person. The geometry test skips when Open3D is absent. With Open3D installed it checks the fixture's normals, bounds, and topology and reads back both export formats.

## Troubleshooting

- If `py -3.12` fails, check `py -0p` and repair the Python registration or run `-m venv .venv` with the absolute path to a working Python 3.12 executable.
- If Open3D has no compatible wheel, use Python 3.12 x64 for geometry. RGB, parameters and Plotly remain available with the base installation.
- If ZIP validation fails, check the sidebar message and the ZIP contract. Place `manifest.json` directly at the archive root.
- For a port conflict, append `--server.port 8502` to the Streamlit command.
- If a 3D viewport is blank, enable browser hardware acceleration or use another WebGL-capable browser.
- If Colab reports that CUDA is unavailable, select a GPU runtime and rerun from the first cell.
- If the detector returns fewer than two people, check the source frame and try a prepared alternate. You can adjust `--threshold` in Colab's inference cell.
- If the upstream archive download stops, rerun the inference cell in the same Colab runtime to resume it. If an unexpected archive size is reported, clear the incomplete checkpoint archive in the Colab cache before retrying.

[Architecture](docs/ARCHITECTURE.md) · [SMPL concepts](docs/SMPL_CONCEPTS.md) · [Coordinate systems](docs/COORDINATE_SYSTEMS.md)

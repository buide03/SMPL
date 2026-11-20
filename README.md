# SMPL Reconstruction Lab

Phase 1 is a Windows CPU viewer for a single scene, frame, and camera per reconstruction ZIP. Inspect source RGB, detection boxes, an imported overlay and optional side view, multiple meshes, and SMPL parameters. Open3D adds geometry inspection and OBJ/PLY export.

## Run on Windows

Install 64-bit Python 3.12 for the viewer and optional Open3D. Python 3.13 can run the viewer without Open3D.

```powershell
git clone https://github.com/buide03/SMPL.git
Set-Location SMPL
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The built-in fixture displays two synthetic tetrahedra and placeholder parameters. To exercise the ZIP uploader, generate that package and upload the resulting file. An upload takes precedence over the built-in fixture.

```powershell
.\.venv\Scripts\python.exe -m src.synthetic_fixture
```

For a prediction from an external exporter, choose **Reconstruction ZIP** and follow the [ZIP contract](docs/DATA_FORMAT.md). **All Detected People** shows independent interactive viewports. Use Plotly's mode bar to switch between orbit and pan or reset the camera. Scroll to zoom.

## Geometry and tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-geometry.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

In the app, expand **Open3D geometry and export** and enable analysis for the selected person. The geometry test skips when Open3D is absent.

## Troubleshooting

- If `py -3.12` fails, check `py -0p` and install or repair 64-bit Python 3.12.
- If Open3D has no compatible wheel, use Python 3.12 x64 for geometry. RGB, parameters and Plotly remain available with the base installation.
- If ZIP validation fails, check the sidebar message and the ZIP contract. Place `manifest.json` directly at the archive root.
- For a port conflict, append `--server.port 8502` to the Streamlit command.
- If a 3D viewport is blank, enable browser hardware acceleration or use another WebGL-capable browser.

[Architecture](docs/ARCHITECTURE.md) · [SMPL concepts](docs/SMPL_CONCEPTS.md) · [Coordinate systems](docs/COORDINATE_SYSTEMS.md)

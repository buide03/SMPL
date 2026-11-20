# SMPL Reconstruction Lab

Phase 1 is a local Windows CPU application for inspecting one scene, frame, and camera per reconstruction ZIP. Streamlit presents the source RGB, detection boxes, imported overlay, optional side view, and per-person SMPL parameters. Plotly renders the mesh in its stored vertex frame. Optional Open3D utilities inspect geometry and export OBJ or PLY. The [architecture](docs/ARCHITECTURE.md) describes how a later GPU exporter can feed the same viewer.

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

For an external prediction, choose **Reconstruction ZIP** and follow the [ZIP contract](docs/DATA_FORMAT.md). Pick a detection in the sidebar. The default view presents the source, overlay, selected mesh and parameter panels. In **Pose θ**, choose the pelvis or one of the 23 named body joints to read its matrix. **All Detected People** shows separate viewports, with four per page for larger packages. Drag to orbit, scroll over the graph to zoom, and use the mode bar to pan or reset.

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

The fixture verifies local behavior with synthetic geometry. Testing on a real PandaSet frame requires a Phase 2 HMR2 exporter and an authorized SMPL neutral model.

[Architecture](docs/ARCHITECTURE.md) · [SMPL concepts](docs/SMPL_CONCEPTS.md) · [Coordinate systems](docs/COORDINATE_SYSTEMS.md)

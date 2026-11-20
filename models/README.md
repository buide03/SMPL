# Local SMPL neutral model

Obtain the neutral model through the official SMPLify download and use it under its license. The Phase 2 notebook uses `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from the SMPLify code and model package. [SMPLify download](https://smplify.is.tue.mpg.de/download.php) · [4D-Humans setup](https://github.com/shubham-goel/4D-Humans#installation-and-setup)

Set `SMPL_MODEL_PATH` to the authorized file before launching Streamlit. You can also place it in this directory as `SMPL_NEUTRAL.pkl` or under its original name. For a Phase 2 bundle with a recorded SHA-256, the local file must match the asset supplied to that run.

```powershell
$env:SMPL_MODEL_PATH = 'D:\SMPLify-Neutral\basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The pickle reader is used only for the trusted local model you configure. Keep model files out of Git, reconstruction ZIPs, and shared folders. The Phase 1 viewer continues to open exported results when the optional SMPL model and dependencies are unavailable.

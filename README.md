# Mango Ripeness Study

A small academic project website for exploring the existing mango image-classification model. The application uses Python, Streamlit, TensorFlow, and the saved model/scaler files included here.

## Run on Windows

### One-click launch

1. Install Python 3.11 and the **Python Launcher for Windows**.
2. Double-click `run_app.bat` in this folder.
3. Wait for Streamlit to print a local address, usually `http://localhost:8501`.
4. Keep the console window open while using the website. Press **Ctrl+C** in that window to stop it.

The launcher creates a local `.venv`, installs the versions specified in `requirements.txt`, and starts the app.

### PowerShell

Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`. If `.venv` already exists, skip the first command.

## Workflow

The website preserves the existing path:

**Home/About → Start Detection → upload a mango image → Predict → classification and score → Home**

The detection page accepts JPG, JPEG, and PNG images. It displays the class label, the model's class-aligned score, the raw model output, and the existing `0.5` decision threshold. Light and dark themes are available on both pages.

## Existing model behavior and limitations

- `best_mango_model_95acc.h5` and `mango_scaler.pkl` are the existing trained assets; neither has been changed.
- The app rebuilds the original three-input architecture and loads the H5 weights. Direct loading of the legacy H5 model config fails with the installed Keras version, so the rebuild-and-load-weights path is retained.
- Image preprocessing remains RGB conversion, resize to `224 × 224`, float32 conversion, and division by `255`.
- The image tensor is sent to both image branches. The four numerical features are still initialized to zeros and transformed by the included scaler; the original training/feature-extraction pipeline is not in this folder, so replacement values cannot be established.
- A score aligned to the selected class is shown for continuity with the existing result card. It is **not** a validated or calibrated probability.
- Visual classification cannot confirm chemical treatment or establish food safety. The model filename is not evidence of a verified accuracy figure; no accuracy claim is made here.

## Performance notes

Model resources are cached for the Streamlit process and loaded only when **Predict** is first pressed. Uploaded-image preprocessing is cached by the exact uploaded bytes, so a different file cannot reuse another image's prepared tensor. Timing for resource loading, uncached image preprocessing, scaler transformation, and inference is written to the server log, not displayed as debug content on the page.

On this Windows/Python 3.11 CPU environment, fresh-process resource loading took **14.3–25.4 s** across runs; TensorFlow/Keras initialization plus architecture construction accounted for most of it (about **13.7–24.7 s**), versus **0.55–1.87 s** for weights and **0.06–0.31 s** for the scaler. A full Streamlit test measured the first inference at **14.2 s** and its repeat at **1.48 s**. In a separate warmed comparison, the existing `model.predict()` path had a **0.228 s median** over three synthetic-image calls. A warmed direct `model(inputs, training=False)` comparison had a **0.446 s median**, so it was not adopted. The two APIs returned exactly the same output (`0.9999988`) for the same generated test image. These are local performance checks, not latency guarantees or validation on real mango photos. TensorFlow initialization and first use account for much more delay than image resizing (about **0.003 s** for the small test file); subsequent predictions are faster. The original code already cached the model; this version preserves that cache and defers loading until the first prediction instead of loading on entry to Detection.

## Checks

Run the preprocessing, inference-contract, result-mapping, and Streamlit workflow tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The `test_load.py` script provides a separate saved-model/scaler smoke check:

```powershell
.\.venv\Scripts\python.exe test_load.py
```

## Project files

- `app.py` — Streamlit interface, navigation/theme state, preprocessing, cached resource loading, and inference.
- `best_mango_model_95acc.h5` — existing model weights and legacy model config.
- `mango_scaler.pkl` — existing four-feature scaler.
- `requirements.txt` — supported runtime dependencies.
- `run_app.bat` — Windows setup and launch script.
- `test_load.py` — model/scaler inference smoke check.
- `tests/test_app.py` — pipeline and end-to-end Streamlit workflow tests.
- `Dockerfile` — optional container configuration.

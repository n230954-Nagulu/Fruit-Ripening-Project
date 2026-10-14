"""Streamlit interface for the Mango Ripeness Detector."""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_mango_model_95acc.h5"
SCALER_PATH = PROJECT_DIR / "mango_scaler.pkl"
IMG_SIZE = (224, 224)
FEATURE_COUNT = 4
DECISION_THRESHOLD = 0.5

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
PREDICTION_LOCK = threading.Lock()


def build_mango_model() -> Any:
    """Build the existing inference architecture for loading the saved weights."""
    import tensorflow as tf

    layers = tf.keras.layers

    mobile_input = layers.Input(shape=(*IMG_SIZE, 3), name="mobile_input")
    mobilenet = tf.keras.applications.MobileNetV3Small(
        weights=None,
        include_top=False,
        input_tensor=mobile_input,
    )
    mobile_features = layers.GlobalAveragePooling2D()(mobilenet.output)
    mobile_features = layers.Dense(256, activation="relu")(mobile_features)
    mobile_features = layers.BatchNormalization()(mobile_features)
    mobile_features = layers.Dropout(0.5)(mobile_features)

    cnn_input = layers.Input(shape=(*IMG_SIZE, 3), name="cnn_input")
    cnn_features = layers.Conv2D(32, (3, 3), activation="relu")(cnn_input)
    cnn_features = layers.MaxPooling2D()(cnn_features)
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.Conv2D(64, (3, 3), activation="relu")(cnn_features)
    cnn_features = layers.MaxPooling2D()(cnn_features)
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.Conv2D(128, (3, 3), activation="relu")(cnn_features)
    cnn_features = layers.MaxPooling2D()(cnn_features)
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
    cnn_features = layers.Dense(128, activation="relu")(cnn_features)
    cnn_features = layers.Dropout(0.4)(cnn_features)

    feature_input = layers.Input(shape=(FEATURE_COUNT,), name="feat_input")
    numeric_features = layers.Dense(64, activation="relu")(feature_input)
    numeric_features = layers.BatchNormalization()(numeric_features)
    numeric_features = layers.Dropout(0.3)(numeric_features)
    numeric_features = layers.Dense(32, activation="relu")(numeric_features)

    combined = layers.Concatenate()(
        [mobile_features, cnn_features, numeric_features]
    )
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation="sigmoid")(combined)

    return tf.keras.models.Model(
        inputs=[mobile_input, cnn_input, feature_input],
        outputs=output,
    )


@st.cache_resource(show_spinner=False)
def load_resources() -> tuple[Any, Any]:
    """Load the saved weights and scaler once per Streamlit process."""
    missing_assets = [
        str(path.name) for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()
    ]
    if missing_assets:
        raise FileNotFoundError(
            f"Required project assets are missing: {', '.join(missing_assets)}"
        )

    started = time.perf_counter()
    model = build_mango_model()
    architecture_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.load_weights(str(MODEL_PATH))
    weights_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scaler = joblib.load(SCALER_PATH)
    scaler_seconds = time.perf_counter() - started

    if getattr(scaler, "n_features_in_", None) != FEATURE_COUNT:
        raise ValueError(
            f"The saved scaler must accept {FEATURE_COUNT} features."
        )

    LOGGER.info(
        "Model resources loaded: architecture=%.3fs weights=%.3fs scaler=%.3fs total=%.3fs",
        architecture_seconds,
        weights_seconds,
        scaler_seconds,
        time.perf_counter() - started + architecture_seconds + weights_seconds,
    )
    return model, scaler


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_uploaded_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode once and return an RGB preview plus the original model input."""
    started = time.perf_counter()
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        rgb_image = source.convert("RGB")
        preview = np.asarray(rgb_image).copy()

        resized = rgb_image.resize(IMG_SIZE)
        image_array = np.array(resized).astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

    LOGGER.info(
        "Image preprocessing completed in %.3fs (input=%d bytes)",
        time.perf_counter() - started,
        len(image_bytes),
    )
    return preview, image_batch


def predict_image(model: Any, scaler: Any, image_batch: np.ndarray) -> float:
    """Run the existing three-input inference pipeline without parallel calls."""
    if not PREDICTION_LOCK.acquire(blocking=False):
        raise RuntimeError("A prediction is already running. Please wait.")

    try:
        feature_started = time.perf_counter()
        base_features = np.zeros((1, FEATURE_COUNT), dtype=np.float32)
        scaled_features = scaler.transform(base_features)
        LOGGER.info(
            "Feature scaling completed in %.3fs",
            time.perf_counter() - feature_started,
        )

        inputs = {
            "mobile_input": image_batch,
            "cnn_input": image_batch,
            "feat_input": scaled_features,
        }

        prediction_started = time.perf_counter()
        output = model.predict(inputs, verbose=0)
        prediction_output = np.asarray(output)
        if prediction_output.shape != (1, 1):
            raise ValueError("The model returned an unexpected output shape.")
        prediction = float(prediction_output[0, 0])
        if not np.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
            raise ValueError("The model returned an invalid classification score.")

        LOGGER.info(
            "Model inference completed in %.3fs",
            time.perf_counter() - prediction_started,
        )
        return prediction
    finally:
        PREDICTION_LOCK.release()


def interpret_prediction(prediction: float) -> tuple[str, float, str]:
    """Preserve the original class threshold and class-aligned score."""
    if prediction > DECISION_THRESHOLD:
        return "Artificially Ripened Mango", prediction, "artificial"
    return "Naturally Ripened Mango", 1.0 - prediction, "natural"


def render_styles(theme: str) -> None:
    """Render the compact shared visual system for the selected theme."""
    palettes = {
        "light": {
            "page": "#f5f6f1",
            "surface": "#ffffff",
            "surface_alt": "#eef3ed",
            "ink": "#1f3026",
            "muted": "#58685d",
            "line": "#dce5dc",
            "green": "#2b704b",
            "green_hover": "#205b3c",
            "button_green": "#2b704b",
            "button_hover": "#205b3c",
            "green_soft": "#e8f1e9",
            "accent": "#bc8429",
            "warning_bg": "#fff7e8",
            "warning_line": "#ead9b6",
        },
        "dark": {
            "page": "#141a15",
            "surface": "#1d261f",
            "surface_alt": "#253128",
            "ink": "#edf2ed",
            "muted": "#b6c2b8",
            "line": "#39483c",
            "green": "#77b68e",
            "green_hover": "#92c9a4",
            "button_green": "#2d7049",
            "button_hover": "#245d3d",
            "green_soft": "#263b2d",
            "accent": "#e0b75e",
            "warning_bg": "#332b1c",
            "warning_line": "#5a4a2b",
        },
    }
    colors = palettes[theme]
    st.markdown(
        f"""
<style>
:root {{
    --page: {colors["page"]};
    --surface: {colors["surface"]};
    --surface-alt: {colors["surface_alt"]};
    --ink: {colors["ink"]};
    --muted: {colors["muted"]};
    --line: {colors["line"]};
    --green: {colors["green"]};
    --green-hover: {colors["green_hover"]};
    --button-green: {colors["button_green"]};
    --button-hover: {colors["button_hover"]};
    --green-soft: {colors["green_soft"]};
    --accent: {colors["accent"]};
    --warning-bg: {colors["warning_bg"]};
    --warning-line: {colors["warning_line"]};
}}
html, body, [class*="css"] {{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.stApp {{ background: var(--page); color: var(--ink); }}
[data-testid="stHeader"] {{ background: var(--page); }}
.block-container {{ max-width: 1080px; padding: 1.25rem 1.5rem 2.5rem; }}
h1, h2, h3 {{ color: var(--ink) !important; letter-spacing: -.035em; }}
h1 {{ font-size: clamp(1.9rem, 4vw, 2.7rem) !important; font-weight: 740 !important; }}
h2 {{ font-size: clamp(1.35rem, 2.3vw, 1.75rem) !important; font-weight: 700 !important; }}
div[data-testid="stMarkdownContainer"] p {{ color: var(--muted); line-height: 1.65; }}
.app-header {{ display:flex; align-items:center; gap:12px; min-height:44px; }}
.wordmark {{ color:var(--ink); font-weight:750; font-size:.93rem; letter-spacing:.025em; line-height:1.2; }}
.wordmark-subtitle {{ color:var(--muted); font-size:.69rem; letter-spacing:.09em; text-transform:uppercase; margin-top:4px; }}
.brand-mark {{ display:grid; place-items:center; width:34px; height:34px; flex:0 0 34px; border:1px solid var(--line); border-radius:9px; color:var(--green); font-size:.72rem; font-weight:800; }}
.route-label {{ margin-top:4px; color:var(--green); font-size:.68rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; }}
[data-testid="stButton"] button {{
    box-sizing: border-box;
    min-height:2.65rem;
    padding:.58rem 1rem;
    border:1px solid var(--line) !important;
    border-radius:9px;
    background:var(--surface) !important;
    color:var(--ink) !important;
    font-size:.88rem;
    font-weight:620;
    box-shadow:0 1px 2px rgba(16,35,22,.08);
    transition:background-color .2s ease, border-color .2s ease, color .2s ease, box-shadow .2s ease, transform .2s ease;
}}
[data-testid="stButton"] button p,
[data-testid="stButton"] button span {{ color:inherit !important; }}
[data-testid="stButton"] button:hover {{
    border-color:var(--green) !important;
    background:var(--green-soft) !important;
    color:var(--green) !important;
    box-shadow:0 3px 8px rgba(16,35,22,.12);
    transform:translateY(-1px);
}}
[data-testid="stButton"] button[kind="primary"], [data-testid="stButton"] button[data-testid="stBaseButton-primary"] {{
    border-color:var(--button-green) !important;
    background:var(--button-green) !important;
    color:#fff !important;
    font-weight:700;
    box-shadow:0 2px 5px rgba(16,35,22,.18);
}}
[data-testid="stButton"] button[kind="primary"]:hover, [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {{
    border-color:var(--button-hover) !important;
    background:var(--button-hover) !important;
    color:#fff !important;
    box-shadow:0 4px 10px rgba(16,35,22,.2);
}}
[data-testid="stButton"] button[kind="primary"] p,
[data-testid="stButton"] button[kind="primary"] span,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"] p,
[data-testid="stButton"] button[data-testid="stBaseButton-primary"] span {{
    color:#fff !important;
}}
[data-testid="stButton"] button:disabled {{
    border-color:var(--line) !important;
    background:var(--surface-alt) !important;
    color:var(--muted) !important;
    box-shadow:none;
    transform:none;
    cursor:not-allowed;
}}
[data-testid="stButton"] button:disabled p,
[data-testid="stButton"] button:disabled span {{ color:var(--muted) !important; }}
.st-key-dark_theme label > div:first-of-type {{ background:var(--line) !important; }}
.st-key-dark_theme label[data-selected="true"] > div:first-of-type {{ background:var(--button-green) !important; }}
.st-key-dark_theme label > div:first-of-type > div {{ background:var(--surface) !important; }}
.st-key-dark_theme label p {{ color:var(--ink) !important; }}
button:focus-visible, a:focus-visible, [role="switch"]:focus-visible {{
    outline:3px solid var(--accent) !important; outline-offset:3px !important;
}}
[data-testid="stToggle"] label, [data-testid="stWidgetLabel"] {{ color:var(--ink) !important; }}
[data-testid="stDivider"] {{ border-color:var(--line); }}
.hero, .content-section, .result-card {{ animation: appear .24s ease-out both; }}
@keyframes appear {{ from {{ opacity:0; transform:translateY(7px); }} to {{ opacity:1; transform:translateY(0); }} }}
.hero {{ border:1px solid var(--line); border-radius:14px; padding:clamp(24px,4vw,42px); background:var(--surface); }}
.eyebrow {{ color:var(--green); font-size:.7rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:12px; }}
.hero-title {{ color:var(--ink) !important; max-width:660px; font-size:clamp(2.15rem,5vw,3.65rem) !important; line-height:1.08 !important; letter-spacing:-.055em !important; margin:0 0 14px !important; }}
.hero-title span {{ color:var(--green); }}
.hero-copy {{ max-width:650px; color:var(--muted) !important; font-size:1rem; line-height:1.7; margin:0 0 4px; }}
.hero-aside {{ height:100%; border-left:2px solid var(--green); padding:8px 0 8px 20px; }}
.hero-aside-title {{ color:var(--ink); font-size:1rem; font-weight:700; margin-bottom:9px; }}
.hero-aside-copy {{ color:var(--muted); font-size:.9rem; line-height:1.65; }}
.secondary-link {{ display:inline-block; padding:.55rem .1rem; color:var(--green) !important; font-size:.9rem; font-weight:680; text-decoration:none; }}
.secondary-link:hover {{ color:var(--green-hover) !important; text-decoration:underline; text-underline-offset:3px; }}
.section-title {{ color:var(--ink); font-size:1.42rem; font-weight:730; letter-spacing:-.035em; margin:0 0 6px; }}
.section-copy {{ color:var(--muted) !important; margin:0; }}
.step-card {{ height:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:11px; padding:20px; background:var(--surface); }}
.step-index {{ color:var(--accent); font-size:.73rem; font-weight:800; letter-spacing:.1em; margin-bottom:12px; }}
.step-title {{ color:var(--ink); font-size:.98rem; font-weight:700; margin-bottom:5px; }}
.step-copy {{ color:var(--muted); font-size:.88rem; line-height:1.6; }}
.limitation-note {{ box-sizing:border-box; padding:15px 17px; border:1px solid var(--warning-line); border-radius:10px; background:var(--warning-bg); color:var(--ink); font-size:.86rem; line-height:1.6; }}
.page-intro {{ margin:0 0 18px; }}
.page-subtitle {{ color:var(--muted) !important; margin:0; font-size:.96rem; }}
.stFileUploader {{ color:var(--ink); }}
.stFileUploader > div {{ box-sizing:border-box; border:1px dashed var(--line); border-radius:11px; background:var(--surface-alt); padding:15px; }}
.stFileUploader label, .stFileUploader [data-testid="stWidgetLabel"] {{ color:var(--ink) !important; font-weight:650; }}
.stFileUploader button {{ border:1px solid var(--line); border-radius:8px; color:var(--green); }}
[data-testid="stImage"] img {{ border:1px solid var(--line); border-radius:12px; }}
.preview-empty {{ display:grid; place-items:center; min-height:205px; box-sizing:border-box; border:1px dashed var(--line); border-radius:11px; background:var(--surface-alt); color:var(--muted); text-align:center; padding:22px; font-size:.9rem; }}
.result-card {{ box-sizing:border-box; margin-top:16px; border:1px solid var(--line); border-radius:13px; padding:clamp(22px,4vw,34px); background:var(--surface); text-align:center; }}
.result-kicker {{ color:var(--muted); font-size:.7rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:10px; }}
.result-class {{ color:var(--ink); font-size:clamp(1.4rem,3vw,2rem); font-weight:760; letter-spacing:-.035em; line-height:1.2; margin-bottom:22px; }}
.result-natural .result-class {{ color:var(--green); }}
.result-artificial .result-class {{ color:var(--accent); }}
.score-value {{ color:var(--ink); font-size:1.15rem; font-weight:750; margin:6px 0 12px; }}
.score-track {{ height:9px; max-width:580px; margin:0 auto; border-radius:8px; overflow:hidden; background:var(--surface-alt); }}
.score-fill {{ height:100%; background:var(--green); border-radius:8px; }}
.result-artificial .score-fill {{ background:var(--accent); }}
.limitation-inline {{ color:var(--muted) !important; font-size:.82rem; text-align:center; margin:12px auto 0; max-width:720px; }}
.technical-note {{ color:var(--muted) !important; font-size:.8rem; text-align:center; }}
.app-footer {{ border-top:1px solid var(--line); margin-top:34px; padding-top:14px; color:var(--muted); font-size:.77rem; }}
[data-testid="stAlert"] {{ border-radius:10px; }}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ scroll-behavior:auto !important; animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }}
}}
@media (max-width:700px) {{
    .block-container {{ padding:1rem .9rem 2rem; }}
    .hero {{ padding:23px 19px; }}
    .hero-aside {{ margin-top:18px; }}
    .step-card {{ padding:16px; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header(page: str, dark_theme: bool) -> None:
    """Render navigation and theme controls shared by both pages."""
    active_nav = "nav_home" if page == "home" else "nav_detection"
    st.markdown(
        f"""
<style>
.st-key-{active_nav} [data-testid="stButton"] button {{
    border-color: var(--green) !important;
    background: var(--green-soft) !important;
    color: var(--green) !important;
    font-weight: 720 !important;
}}
.st-key-{active_nav} [data-testid="stButton"] button:hover {{
    border-color: var(--green-hover) !important;
    background: var(--green-soft) !important;
    color: var(--green-hover) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )
    brand_column, home_column, detect_column, theme_column = st.columns(
        [4.1, 1, 1.35, 1.55],
        vertical_alignment="center",
    )
    with brand_column:
        route_label = "Home" if page == "home" else "Detection"
        st.markdown(
            f"""
<div class="app-header">
  <span class="brand-mark" aria-hidden="true">MR</span>
  <div><div class="wordmark">Mango Ripeness Study</div>
  <div class="route-label" aria-current="page">{route_label}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with home_column:
        if st.button(
            "Home",
            key="nav_home",
            type="secondary",
            width="stretch",
            help="Go to the project home page",
        ) and page != "home":
            st.session_state.page = "home"
            st.rerun()

    with detect_column:
        if st.button(
            "Detection",
            key="nav_detection",
            type="secondary",
            width="stretch",
            help="Open the mango image detector",
        ) and page != "detection":
            st.session_state.page = "detection"
            st.rerun()

    with theme_column:
        st.toggle(
            "Dark theme",
            key="dark_theme",
            value=dark_theme,
            help="Switch between the light and dark color themes",
        )


def render_home() -> None:
    st.markdown(
        """
<section class="hero">
  <div class="eyebrow">Agricultural computer vision · Academic project</div>
  <h1 class="hero-title">A closer look at<br><span>mango ripening.</span></h1>
  <p class="hero-copy">Explore an image-based classification demonstration built around the project's existing trained model. Upload a mango image to view the model's classification and score.</p>
</section>
""",
        unsafe_allow_html=True,
    )

    action_column, aside_column = st.columns([1.25, 0.75], gap="large")
    with action_column:
        if st.button(
            "Start Detection",
            key="start_detection",
            type="primary",
            width="stretch",
        ):
            st.session_state.page = "detection"
            st.rerun()
        st.markdown(
            '<a class="secondary-link" href="#how-it-works">How it works <span aria-hidden="true">↓</span></a>',
            unsafe_allow_html=True,
        )

    with aside_column:
        st.markdown(
            """
<div class="hero-aside">
  <div class="hero-aside-title">About the project</div>
  <div class="hero-aside-copy">A Python and Streamlit presentation layer for an existing image-classification model. Results are model outputs for demonstration, not verified food-safety findings.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<section class="content-section">
  <div class="eyebrow">The process</div>
  <div class="section-title">Three steps, from image to result</div>
  <p class="section-copy">A direct path through the existing detection workflow.</p>
</section>
""",
        unsafe_allow_html=True,
    )
    st.write("")
    first, second, third = st.columns(3, gap="medium")
    steps = (
        ("01", "Upload an image", "Choose a clear mango photo in JPG, JPEG, or PNG format."),
        ("02", "Model analysis", "The image is resized and passed into the saved model."),
        ("03", "Review the result", "See the existing class label and class-aligned model score."),
    )
    for column, (number, title, description) in zip(
        (first, second, third),
        steps,
        strict=True,
    ):
        with column:
            st.markdown(
                f"""
<div class="step-card">
  <div class="step-index">{number}</div>
  <div class="step-title">{title}</div>
  <div class="step-copy">{description}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
<div class="limitation-note">
  <strong>Project limitation:</strong> this folder does not include the original training pipeline or feature extraction. The current application supplies four zero-valued numerical features, as before. Visual classification cannot establish chemical treatment or food safety.
</div>
""",
        unsafe_allow_html=True,
    )


def render_detection() -> None:
    st.markdown(
        """
<div class="page-intro">
  <div class="eyebrow">Image classification</div>
  <h1>Detection</h1>
  <p class="page-subtitle">Upload a clear photo of one mango, then run the existing model.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    upload_column, preview_column = st.columns([1.1, 0.9], gap="large")
    with upload_column:
        st.markdown("**Choose an image**")
        st.caption("Supported formats: JPG, JPEG, PNG. Use a clear, well-lit photo.")
        uploaded_file = st.file_uploader(
            "Mango image",
            type=["jpg", "jpeg", "png"],
            key="mango_image",
            label_visibility="collapsed",
        )

    preview: np.ndarray | None = None
    image_batch: np.ndarray | None = None
    image_digest: str | None = None
    image_error = False

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        try:
            preview, image_batch = prepare_uploaded_image(image_bytes)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            image_error = True
            st.error(
                "This file could not be read as a valid image. Please choose a JPG, "
                "JPEG, or PNG image."
            )
        except Exception:
            LOGGER.exception("Unexpected error while decoding the uploaded image.")
            image_error = True
            st.error("The image could not be prepared. Please try a different image.")

    with preview_column:
        st.markdown("**Image preview**")
        if preview is None:
            st.markdown(
                '<div class="preview-empty">Your selected mango image will appear here.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.image(preview, caption="Selected mango image", width="stretch")

    if image_digest is None or (
        st.session_state.get("last_prediction") is not None
        and st.session_state.last_prediction["image_digest"] != image_digest
    ):
        st.session_state.last_prediction = None


    if image_batch is not None and not image_error:
        prediction_running = st.session_state.get("prediction_running", False)
        if st.button(
            "Predict",
            key="predict",
            type="primary",
            width="stretch",
            disabled=prediction_running,
        ):
            st.session_state.prediction_running = True
            st.session_state.last_prediction = None
            try:
                with st.spinner(
                    "Preparing the trained model and analyzing the image...",
                ):
                    model, scaler = load_resources()
                    prediction = predict_image(model, scaler, image_batch)
                st.session_state.last_prediction = {
                    "image_digest": image_digest,
                    "raw_prediction": prediction,
                }
            except RuntimeError as error:
                st.error(str(error))
            except (FileNotFoundError, OSError, ValueError):
                LOGGER.exception("Could not load resources or complete prediction.")
                st.error(
                    "The model could not complete this prediction. Check that the "
                    "model and scaler files are present and compatible."
                )
            except Exception:
                LOGGER.exception("Unexpected error during model loading or prediction.")
                st.error(
                    "Prediction could not be completed. Please try again or check "
                    "the application logs."
                )
            finally:
                st.session_state.prediction_running = False

    result = st.session_state.get("last_prediction")
    if result is not None and result["image_digest"] == image_digest:
        label, class_score, category = interpret_prediction(
            result["raw_prediction"]
        )
        st.markdown(
            f"""
<section class="result-card result-{category}" role="status" aria-live="polite">
  <div class="result-kicker">Analysis result</div>
  <div class="result-class">{label}</div>
  <div class="result-kicker">Class-aligned model score · not a calibrated probability</div>
  <div class="score-value">{class_score:.2%}</div>
  <div class="score-track" role="progressbar" aria-label="Class-aligned model score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{class_score * 100:.0f}">
    <div class="score-fill" style="width:{class_score * 100:.2f}%"></div>
  </div>
</section>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="technical-note">Raw model output: {result["raw_prediction"]:.4f} · Existing decision threshold: {DECISION_THRESHOLD:.1f}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="limitation-inline">This visual result does not verify ripening chemicals, treatment history, or food safety.</p>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Mango Ripeness Study",
        page_icon="🌿",
        layout="wide",
    )
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("dark_theme", False)
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("prediction_running", False)

    page = st.session_state.page
    dark_theme = st.session_state.dark_theme
    render_styles("dark" if dark_theme else "light")
    render_header(page, dark_theme)
    st.divider()

    if page == "home":
        render_home()
    else:
        render_detection()

    st.markdown(
        '<footer class="app-footer">Mango Ripeness Study · Python · Streamlit · TensorFlow</footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

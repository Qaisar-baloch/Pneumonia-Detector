"""
PneumoScan — Clinical Streamlit UI
Educational / portfolio project only. Not a medical device.

The UI is intentionally designed as a clinical/radiology-style workstation:
- large X-ray viewer
- structured AI readout
- model probability vs. confidence distinction
- Grad-CAM explanation
- technical image-quality checks
- explicit clinical safety notice
"""
import hashlib
import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(APP_DIR, "models", "pneumonia_resnet18.pt"),
)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PNEUMONIA_THRESHOLD = float(os.environ.get("PNEUMONIA_THRESHOLD", "0.85"))

HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID")
HF_MODEL_FILENAME = os.environ.get(
    "HF_MODEL_FILENAME",
    "pneumonia_resnet18.pt",
)

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan — AI-Assisted Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Clinical UI styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ps-bg: #0f151a;
    --ps-panel: #182128;
    --ps-panel-2: #202b33;
    --ps-border: #2b3841;
    --ps-text: #e9f0f2;
    --ps-muted: #9aa8af;
    --ps-faint: #687780;
    --ps-teal: #55c2b4;
    --ps-teal-soft: rgba(85,194,180,.13);
    --ps-warn: #e4a26f;
    --ps-danger: #dc806e;
    --ps-radius: 12px;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: var(--ps-bg);
    color: var(--ps-text);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.ps-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--ps-border);
    padding: 0 0 18px 0;
    margin-bottom: 24px;
}

.ps-brand {
    display: flex;
    align-items: center;
    gap: 11px;
}

.ps-mark {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    background: var(--ps-teal);
    box-shadow: 0 0 14px rgba(85,194,180,.28);
}

.ps-brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: .01em;
}

.ps-brand-sub {
    color: var(--ps-muted);
    font-size: .82rem;
    margin-left: 4px;
}

.ps-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 11px;
    border: 1px solid var(--ps-border);
    border-radius: 999px;
    color: var(--ps-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem;
    letter-spacing: .06em;
    text-transform: uppercase;
}

.ps-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
}

.ps-section-label {
    color: var(--ps-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .67rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin: 0 0 8px 2px;
}

.ps-panel {
    background: var(--ps-panel);
    border: 1px solid var(--ps-border);
    border-radius: var(--ps-radius);
    padding: 20px;
}

.ps-viewer {
    min-height: 520px;
    background: #e8edf0;
    border: 1px solid var(--ps-border);
    border-radius: var(--ps-radius);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.ps-empty {
    text-align: center;
    color: #59666d;
    padding: 55px 30px;
}

.ps-empty-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}

.ps-empty-title {
    font-size: .98rem;
    margin-bottom: 4px;
}

.ps-empty-sub {
    font-family: 'IBM Plex Mono', monospace;
    color: #7b878d;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.ps-readout {
    min-height: 520px;
}

.ps-result-kicker {
    color: var(--ps-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .67rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.ps-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 18px;
}

.ps-normal { color: var(--ps-teal); }
.ps-pneumonia { color: var(--ps-danger); }

.ps-gauge {
    position: relative;
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #3c9e93 0%, #34424a 50%, #b96758 100%);
    margin: 4px 0 8px 0;
}

.ps-gauge-marker {
    position: absolute;
    top: -5px;
    width: 3px;
    height: 18px;
    border-radius: 3px;
    background: #f5fafb;
    box-shadow: 0 0 7px rgba(255,255,255,.4);
}

.ps-gauge-labels {
    display: flex;
    justify-content: space-between;
    color: var(--ps-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .63rem;
    letter-spacing: .05em;
    margin-bottom: 22px;
}

.ps-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
}

.ps-stat {
    background: var(--ps-panel-2);
    border: 1px solid var(--ps-border);
    border-radius: 9px;
    padding: 13px;
}

.ps-stat-label {
    color: var(--ps-faint);
    font-size: .72rem;
    margin-bottom: 5px;
}

.ps-stat-value {
    color: var(--ps-text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .9rem;
}

.ps-divider {
    border-top: 1px solid var(--ps-border);
    margin: 20px 0;
}

.ps-note {
    color: var(--ps-muted);
    font-size: .84rem;
    line-height: 1.55;
}

.ps-note strong {
    color: var(--ps-text);
}

.ps-safety {
    border-left: 3px solid var(--ps-teal);
    background: var(--ps-teal-soft);
    border-radius: 0 8px 8px 0;
    padding: 13px 15px;
    color: var(--ps-muted);
    font-size: .78rem;
    line-height: 1.55;
    margin-top: 20px;
}

.ps-safety strong {
    color: var(--ps-text);
}

.ps-quality-good {
    color: var(--ps-teal);
}

.ps-meta {
    color: var(--ps-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .64rem;
    letter-spacing: .03em;
    line-height: 1.7;
    margin: 8px 0;
}

div[data-testid="stFileUploader"] {
    background: var(--ps-panel);
    border: 1px dashed #42515a;
    border-radius: var(--ps-radius);
    padding: 5px;
    margin-bottom: 12px;
}

div[data-testid="stFileUploader"]:hover {
    border-color: var(--ps-teal);
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #3c8f86;
    background: var(--ps-teal-soft);
    color: var(--ps-text);
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    border-color: var(--ps-teal);
    color: white;
}

div[data-testid="stTabs"] button {
    color: var(--ps-muted);
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--ps-teal);
}

@media (max-width: 850px) {
    .ps-topbar { align-items: flex-start; gap: 12px; }
    .ps-brand-sub { display: none; }
    .ps-stat-grid { grid-template-columns: 1fr; }
    .ps-viewer { min-height: 360px; }
    .ps-readout { min-height: auto; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = self.model(input_tensor)
            output.backward(torch.ones_like(output))

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM could not collect model activations.")

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * acts, dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        prob = torch.sigmoid(output.detach()).item()
        return cam.cpu().numpy(), prob


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return None

    if not HF_MODEL_REPO_ID:
        return (
            f"No model file found at {MODEL_PATH}. "
            "Set HF_MODEL_REPO_ID or place pneumonia_resnet18.pt in app/models/."
        )

    try:
        from huggingface_hub import hf_hub_download
        import shutil

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
        )
        shutil.copy(downloaded_path, MODEL_PATH)
        return None
    except Exception as exc:
        return f"Failed to download model from Hugging Face Hub: {exc}"


@st.cache_resource
def load_model():
    error = ensure_model_downloaded()
    if error:
        return None, None, error

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        class_to_idx = checkpoint.get(
            "class_to_idx",
            {"NORMAL": 0, "PNEUMONIA": 1},
        )
        idx_to_class = {value: key for key, value in class_to_idx.items()}
        gradcam = GradCAM(model, model.layer4)
        return model, gradcam, idx_to_class
    except Exception as exc:
        return None, None, f"Failed to load model: {exc}"


eval_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Image / explanation helpers
# ---------------------------------------------------------------------------
def quadrant_focus(cam_resized):
    height, width = cam_resized.shape
    ys, xs = np.mgrid[0:height, 0:width]
    total = cam_resized.sum() + 1e-8
    cy = (ys * cam_resized).sum() / total
    cx = (xs * cam_resized).sum() / total

    vertical = "upper" if cy < height / 2 else "lower"
    horizontal = "left" if cx < width / 2 else "right"
    return f"{vertical} {horizontal} lung field"


def jet_colormap(values):
    values = np.clip(values, 0, 1)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_gradcam_overlay(pil_img, cam):
    base = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    cam_img = Image.fromarray(
        (cam * 255).astype(np.uint8)
    ).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0
    heatmap = jet_colormap(cam_resized)
    base_arr = np.asarray(base).astype(np.float32)

    overlay = np.uint8(0.55 * base_arr + 0.45 * heatmap)
    return base, Image.fromarray(overlay), cam_resized


def technical_quality_check(image):
    gray = np.asarray(image.convert("L"))
    width, height = image.size
    issues = []

    if min(width, height) < 224:
        issues.append("Low image resolution for the model input size.")

    if gray.std() < 12:
        issues.append("Low grayscale contrast; image may be very flat or blank.")

    aspect_ratio = width / max(height, 1)
    if aspect_ratio < 0.45 or aspect_ratio > 1.8:
        issues.append("Unusual image aspect ratio; verify the uploaded study.")

    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "issues": issues,
    }


def image_hash(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
model, gradcam, model_status = load_model()
model_ready = model is not None

status_text = "model ready" if model_ready else "model unavailable"
status_color = "var(--ps-teal)" if model_ready else "var(--ps-danger)"

st.markdown(
    f"""
<div class="ps-topbar">
  <div class="ps-brand">
    <span class="ps-mark"></span>
    <span class="ps-brand-name">PneumoScan</span>
    <span class="ps-brand-sub">AI-Assisted Chest X-Ray Analysis</span>
  </div>
  <div class="ps-status">
    <span class="ps-status-dot" style="background:{status_color};"></span>
    {status_text}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not model_ready:
    st.error(model_status)
    st.markdown(
        """
<div class="ps-safety">
<strong>System unavailable.</strong> No inference is performed until the model is loaded successfully.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown('<div class="ps-section-label">X-ray viewer</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="PNG or JPEG chest X-ray image.",
    )

    if uploaded_file is None:
        st.markdown(
            """
<div class="ps-viewer">
  <div class="ps-empty">
    <div class="ps-empty-icon">🫁</div>
    <div class="ps-empty-title">Drop a chest X-ray here, or use the uploader above</div>
    <div class="ps-empty-sub">JPG / PNG · technical quality check before analysis</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        raw_bytes = uploaded_file.getvalue()
        current_hash = image_hash(raw_bytes)

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be read as an image.")
            st.stop()

        quality = technical_quality_check(pil_img)

        if st.session_state.uploaded_hash != current_hash:
            st.session_state.analysis = None
            st.session_state.uploaded_hash = current_hash

        st.markdown(
            f"""
<div class="ps-meta">
STUDY FILE · {uploaded_file.name}<br>
IMAGE · {quality["width"]} × {quality["height"]} px · RGB
</div>
            """,
            unsafe_allow_html=True,
        )

        if quality["issues"]:
            st.warning(
                "Technical image-quality check: "
                + " ".join(quality["issues"])
            )
        else:
            st.markdown(
                '<div class="ps-meta ps-quality-good">✓ Basic technical image checks passed</div>',
                unsafe_allow_html=True,
            )

        analysis = st.session_state.analysis

        if analysis is None:
            st.image(pil_img, use_container_width=True)

            if st.button("Run PNEUMOSCAN Analysis", type="primary"):
                input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

                with st.status("Analyzing chest X-ray...", expanded=True) as status:
                    st.write("Preprocessing image...")
                    st.write("Running ResNet18 inference...")
                    try:
                        cam, prob = gradcam.generate(input_tensor)
                        base_img, overlay_img, cam_resized = make_gradcam_overlay(
                            pil_img, cam
                        )
                        pred_idx = 1 if prob >= PNEUMONIA_THRESHOLD else 0
                        pred_label = idx_to_class.get(
                            pred_idx,
                            "PNEUMONIA" if pred_idx == 1 else "NORMAL",
                        )
                        confidence = prob if pred_idx == 1 else 1 - prob
                        focus = quadrant_focus(cam_resized)

                        st.session_state.analysis = {
                            "prob": prob,
                            "confidence": confidence,
                            "prediction": pred_label,
                            "base_img": base_img,
                            "overlay_img": overlay_img,
                            "focus": focus,
                        }
                        status.update(
                            label="Analysis complete",
                            state="complete",
                            expanded=False,
                        )
                        st.rerun()
                    except Exception as exc:
                        status.update(
                            label="Analysis failed",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Inference failed: {exc}")
        else:
            tabs = st.tabs(["Original", "AI Attention / Grad-CAM"])

            with tabs[0]:
                st.image(
                    analysis["base_img"],
                    use_container_width=True,
                    caption="Original uploaded image",
                )

            with tabs[1]:
                st.image(
                    analysis["overlay_img"],
                    use_container_width=True,
                    caption=(
                        "Grad-CAM overlay — highlights image regions "
                        "that influenced the model output"
                    ),
                )

            if st.button("Load a different image"):
                st.session_state.analysis = None
                st.session_state.uploaded_hash = None
                st.rerun()


with right:
    st.markdown('<div class="ps-section-label">AI readout</div>', unsafe_allow_html=True)

    analysis = st.session_state.analysis

    if analysis is None:
        st.markdown(
            """
<div class="ps-panel ps-readout">
  <div style="display:flex;align-items:center;height:100%;">
    <div>
      <div class="ps-result-kicker">Prediction</div>
      <div class="ps-note">
        Results will appear here once an X-ray has been uploaded and analyzed.
      </div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        is_pneumonia = analysis["prediction"].upper() == "PNEUMONIA"
        verdict_class = "ps-pneumonia" if is_pneumonia else "ps-normal"

        probability_pct = analysis["prob"] * 100
        confidence_pct = analysis["confidence"] * 100
        marker_pct = min(max(probability_pct, 0), 100)

        st.markdown(
            f"""
<div class="ps-panel ps-readout">
  <div class="ps-result-kicker">Model assessment</div>
  <div class="ps-verdict {verdict_class}">{analysis["prediction"]}</div>

  <div class="ps-result-kicker">Pneumonia model probability</div>
  <div class="ps-gauge">
    <div class="ps-gauge-marker" style="left:calc({marker_pct}% - 1px);"></div>
  </div>
  <div class="ps-gauge-labels">
    <span>LOWER</span>
    <span>{probability_pct:.1f}% model probability</span>
    <span>HIGHER</span>
  </div>

  <div class="ps-stat-grid">
    <div class="ps-stat">
      <div class="ps-stat-label">Model confidence</div>
      <div class="ps-stat-value">{confidence_pct:.1f}%</div>
    </div>
    <div class="ps-stat">
      <div class="ps-stat-label">Decision threshold</div>
      <div class="ps-stat-value">{PNEUMONIA_THRESHOLD:.2f}</div>
    </div>
    <div class="ps-stat">
      <div class="ps-stat-label">Grad-CAM focus</div>
      <div class="ps-stat-value">{analysis["focus"]}</div>
    </div>
    <div class="ps-stat">
      <div class="ps-stat-label">Model</div>
      <div class="ps-stat-value">ResNet18</div>
    </div>
  </div>

  <div class="ps-divider"></div>

  <div class="ps-result-kicker">AI interpretation</div>
  <div class="ps-note">
    The model output is a <strong>machine-learning prediction</strong>, not a
    clinical diagnosis. Review the original radiograph, the AI attention map,
    and relevant clinical information before drawing conclusions.
  </div>

  <div class="ps-safety">
    <strong>Clinical decision-support notice.</strong>
    PneumoScan is an educational/portfolio AI project and is not validated
    for clinical use. Its output should not be used as a standalone diagnosis
    or treatment decision. Consult a qualified healthcare professional.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
<div style="margin-top:24px;border-top:1px solid #2b3841;padding-top:16px;text-align:center;"
     class="ps-meta">
PNEUMOSCAN · EDUCATIONAL / PORTFOLIO PROJECT · NOT A DIAGNOSTIC DEVICE
</div>
    """,
    unsafe_allow_html=True,
)

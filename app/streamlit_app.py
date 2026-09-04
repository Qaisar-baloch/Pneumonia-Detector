"""PneumoScan — clinical Streamlit workstation.
Educational / portfolio project only. Not a medical device.
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_DIR, "models", "pneumonia_resnet18.pt"))
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PNEUMONIA_THRESHOLD = float(os.environ.get("PNEUMONIA_THRESHOLD", "0.85"))
HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "pneumonia_resnet18.pt")
device = torch.device("cpu")

st.set_page_config(
    page_title="PneumoScan — AI-Assisted Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#0f151a;--panel:#182128;--panel2:#202b33;--border:#2b3841;--text:#e9f0f2;--muted:#9aa8af;--faint:#687780;--teal:#55c2b4;--tealsoft:rgba(85,194,180,.13);--danger:#dc806e;--radius:12px}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}.stApp{background:var(--bg);color:var(--text)}
.block-container{max-width:1320px;padding-top:1.2rem;padding-bottom:3rem}#MainMenu,footer{visibility:hidden}header[data-testid="stHeader"]{background:transparent}
.ps-top{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:18px;margin-bottom:24px}.ps-brand{display:flex;align-items:center;gap:11px}.ps-mark{width:11px;height:11px;border-radius:3px;background:var(--teal);box-shadow:0 0 14px rgba(85,194,180,.28)}.ps-name{font-family:'Space Grotesk';font-weight:700;font-size:1.25rem}.ps-sub{color:var(--muted);font-size:.82rem}.ps-status{display:flex;align-items:center;gap:8px;padding:6px 11px;border:1px solid var(--border);border-radius:999px;color:var(--muted);font: .68rem 'IBM Plex Mono';text-transform:uppercase;letter-spacing:.06em}.ps-dot{width:7px;height:7px;border-radius:50%}
.ps-label{color:var(--faint);font:.67rem 'IBM Plex Mono';letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px 2px}.ps-panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:20px}.ps-viewer{min-height:500px;background:#e8edf0;border:1px solid var(--border);border-radius:var(--radius);display:flex;align-items:center;justify-content:center;overflow:hidden}.ps-empty{text-align:center;color:#59666d;padding:50px}.ps-empty-icon{font-size:2rem}.ps-empty-title{font-size:.98rem;margin:8px 0 4px}.ps-empty-sub,.ps-meta{font: .65rem 'IBM Plex Mono';color:#7b878d;text-transform:uppercase;letter-spacing:.05em}.ps-readout{min-height:500px}.ps-kicker{color:var(--faint);font:.67rem 'IBM Plex Mono';letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}.ps-verdict{font-family:'Space Grotesk';font-size:2rem;font-weight:700;margin-bottom:18px}.normal{color:var(--teal)}.pneumonia{color:var(--danger)}
.ps-gauge{position:relative;height:8px;border-radius:99px;background:linear-gradient(90deg,#3c9e93,#34424a 50%,#b96758);margin:4px 0 8px}.ps-marker{position:absolute;top:-5px;width:3px;height:18px;border-radius:3px;background:#f5fafb;box-shadow:0 0 7px rgba(255,255,255,.4)}.ps-gauge-labels{display:flex;justify-content:space-between;color:var(--faint);font:.63rem 'IBM Plex Mono';margin-bottom:22px}.ps-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.ps-stat{background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:13px}.ps-stat-label{color:var(--faint);font-size:.72rem;margin-bottom:5px}.ps-stat-value{font:.88rem 'IBM Plex Mono';color:var(--text)}.ps-divider{border-top:1px solid var(--border);margin:20px 0}.ps-note{color:var(--muted);font-size:.84rem;line-height:1.55}.ps-note strong{color:var(--text)}.ps-safety{border-left:3px solid var(--teal);background:var(--tealsoft);border-radius:0 8px 8px 0;padding:13px 15px;color:var(--muted);font-size:.78rem;line-height:1.55;margin-top:20px}.ps-safety strong{color:var(--text)}
div[data-testid="stFileUploader"]{background:var(--panel);border:1px dashed #42515a;border-radius:var(--radius);padding:5px;margin-bottom:12px}div[data-testid="stFileUploader"]:hover{border-color:var(--teal)}.stButton>button{width:100%;border-radius:8px;border:1px solid #3c8f86;background:var(--tealsoft);color:var(--text);font-weight:600;min-height:42px}.stButton>button:hover{border-color:var(--teal);color:#fff}div[data-testid="stTabs"] button{color:var(--muted)}div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--teal)}
@media(max-width:850px){.ps-top{align-items:flex-start;gap:12px}.ps-sub{display:none}.ps-grid{grid-template-columns:1fr}.ps-viewer,.ps-readout{min-height:360px}}
</style>
""", unsafe_allow_html=True)


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate(self, tensor):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = self.model(tensor)
            output.backward(torch.ones_like(output))
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM could not collect model activations.")
        grads, acts = self.gradients[0], self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * acts, dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.cpu().numpy(), torch.sigmoid(output.detach()).item()


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return None
    if not HF_MODEL_REPO_ID:
        return f"No model file found at {MODEL_PATH}. Set HF_MODEL_REPO_ID or place pneumonia_resnet18.pt in app/models/."
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
        shutil.copy(downloaded, MODEL_PATH)
        return None
    except Exception as exc:
        return f"Failed to download model from Hugging Face Hub: {exc}"


@st.cache_resource
def load_model():
    error = ensure_model_downloaded()
    if error:
        return None, None, None, error
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        class_to_idx = checkpoint.get("class_to_idx", {"NORMAL": 0, "PNEUMONIA": 1})
        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
        gradcam = GradCAM(model, model.layer4)
        return model, gradcam, idx_to_class, None
    except Exception as exc:
        return None, None, None, f"Failed to load model: {exc}"


eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def quadrant_focus(cam):
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = cam.sum() + 1e-8
    cy, cx = (ys * cam).sum() / total, (xs * cam).sum() / total
    return f"{'upper' if cy < h/2 else 'lower'} {'left' if cx < w/2 else 'right'} lung field"


def jet_colormap(v):
    v = np.clip(v, 0, 1)
    rgb = np.stack([
        np.clip(1.5 - np.abs(4*v - 3), 0, 1),
        np.clip(1.5 - np.abs(4*v - 2), 0, 1),
        np.clip(1.5 - np.abs(4*v - 1), 0, 1),
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_overlay(image, cam):
    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255
    heat = jet_colormap(cam_arr)
    base_arr = np.asarray(base).astype(np.float32)
    overlay = np.uint8(.55 * base_arr + .45 * heat)
    return base, Image.fromarray(overlay), cam_arr


def quality_check(image):
    gray = np.asarray(image.convert("L"))
    w, h = image.size
    issues = []
    if min(w, h) < 224: issues.append("low resolution")
    if gray.std() < 12: issues.append("low grayscale contrast")
    ratio = w / max(h, 1)
    if ratio < .45 or ratio > 1.8: issues.append("unusual aspect ratio")
    return w, h, issues


if "analysis" not in st.session_state: st.session_state.analysis = None
if "uploaded_hash" not in st.session_state: st.session_state.uploaded_hash = None

model, gradcam, idx_to_class, model_status = load_model()
model_ready = model is not None
status_color = "#55c2b4" if model_ready else "#dc806e"

st.markdown(f"""
<div class="ps-top"><div class="ps-brand"><span class="ps-mark"></span><span class="ps-name">PneumoScan</span><span class="ps-sub">AI-Assisted Chest X-Ray Analysis</span></div><div class="ps-status"><span class="ps-dot" style="background:{status_color}"></span>{'model ready' if model_ready else 'model unavailable'}</div></div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error(model_status)
    st.stop()

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown('<div class="ps-label">X-ray viewer</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded is None:
        st.markdown('<div class="ps-viewer"><div class="ps-empty"><div class="ps-empty-icon">🫁</div><div class="ps-empty-title">Drop a chest X-ray here, or use the uploader above</div><div class="ps-empty-sub">JPG / PNG · technical quality check before analysis</div></div></div>', unsafe_allow_html=True)
    else:
        raw = uploaded.getvalue()
        file_hash = hashlib.sha256(raw).hexdigest()
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be read as an image.")
            st.stop()

        if st.session_state.uploaded_hash != file_hash:
            st.session_state.analysis = None
            st.session_state.uploaded_hash = file_hash

        w, h, issues = quality_check(image)
        st.markdown(f'<div class="ps-meta">STUDY FILE · {uploaded.name}<br>IMAGE · {w} × {h} px · RGB</div>', unsafe_allow_html=True)
        if issues: st.warning("Technical image-quality check: " + "; ".join(issues) + ".")
        else: st.markdown('<div class="ps-meta" style="color:#55c2b4">✓ Basic technical image checks passed</div>', unsafe_allow_html=True)

        if st.session_state.analysis is None:
            st.image(image, use_container_width=True)
            if st.button("Run PNEUMOSCAN Analysis", type="primary"):
                tensor = eval_transform(image).unsqueeze(0).to(device)
                with st.status("Analyzing chest X-ray...", expanded=True) as status:
                    st.write("Preprocessing image...")
                    st.write("Running ResNet18 inference...")
                    try:
                        cam, prob = gradcam.generate(tensor)
                        base, overlay, cam_resized = make_overlay(image, cam)
                        pred_idx = 1 if prob >= PNEUMONIA_THRESHOLD else 0
                        pred_label = idx_to_class.get(pred_idx, "PNEUMONIA" if pred_idx == 1 else "NORMAL")
                        confidence = prob if pred_idx == 1 else 1 - prob
                        st.session_state.analysis = {"prob": prob, "confidence": confidence, "prediction": pred_label, "base": base, "overlay": overlay, "focus": quadrant_focus(cam_resized)}
                        status.update(label="Analysis complete", state="complete", expanded=False)
                        st.rerun()
                    except Exception as exc:
                        status.update(label="Analysis failed", state="error", expanded=True)
                        st.error(f"Inference failed: {exc}")
        else:
            a = st.session_state.analysis
            original, explanation = st.tabs(["Original", "AI Attention / Grad-CAM"])
            with original: st.image(a["base"], use_container_width=True, caption="Original uploaded image")
            with explanation: st.image(a["overlay"], use_container_width=True, caption="Grad-CAM overlay — regions that influenced the model output")
            st.caption("Grad-CAM is an explainability aid, not a clinically validated lesion-localization method.")
            if st.button("Load a different image"):
                st.session_state.analysis = None
                st.session_state.uploaded_hash = None
                st.rerun()

with right:
    st.markdown('<div class="ps-label">AI readout</div>', unsafe_allow_html=True)
    a = st.session_state.analysis
    if a is None:
        st.markdown('<div class="ps-panel ps-readout"><div class="ps-kicker">Prediction</div><div class="ps-note">Results will appear here once an X-ray has been uploaded and analyzed.</div></div>', unsafe_allow_html=True)
    else:
        pneumonia = a["prediction"].upper() == "PNEUMONIA"
        cls = "pneumonia" if pneumonia else "normal"
        prob_pct = a["prob"] * 100
        conf_pct = a["confidence"] * 100
        marker = min(max(prob_pct, 0), 100)
        st.markdown(f"""
<div class="ps-panel ps-readout">
<div class="ps-kicker">Model assessment</div><div class="ps-verdict {cls}">{a['prediction']}</div>
<div class="ps-kicker">Pneumonia model probability</div><div class="ps-gauge"><div class="ps-marker" style="left:calc({marker}% - 1px)"></div></div>
<div class="ps-gauge-labels"><span>LOWER</span><span>{prob_pct:.1f}% model probability</span><span>HIGHER</span></div>
<div class="ps-grid"><div class="ps-stat"><div class="ps-stat-label">Model confidence</div><div class="ps-stat-value">{conf_pct:.1f}%</div></div><div class="ps-stat"><div class="ps-stat-label">Decision threshold</div><div class="ps-stat-value">{PNEUMONIA_THRESHOLD:.2f}</div></div><div class="ps-stat"><div class="ps-stat-label">Grad-CAM focus</div><div class="ps-stat-value">{a['focus']}</div></div><div class="ps-stat"><div class="ps-stat-label">Model</div><div class="ps-stat-value">ResNet18</div></div></div>
<div class="ps-divider"></div><div class="ps-kicker">AI interpretation</div><div class="ps-note">The output is a <strong>machine-learning prediction</strong>, not a clinical diagnosis. Review the original radiograph, AI attention map, and relevant clinical information before drawing conclusions.</div>
<div class="ps-safety"><strong>Clinical decision-support notice.</strong> PneumoScan is an educational/portfolio AI project and is not validated for clinical use. Do not use it as a standalone diagnosis or treatment decision. Consult a qualified healthcare professional.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ps-meta" style="text-align:center;border-top:1px solid #2b3841;padding-top:16px;margin-top:24px">PNEUMOSCAN · EDUCATIONAL / PORTFOLIO PROJECT · NOT A DIAGNOSTIC DEVICE</div>', unsafe_allow_html=True)

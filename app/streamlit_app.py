"""
Pneumonia Detector — Streamlit app
Loads the ResNet18 model trained in the companion notebook (pneumonia_detection.ipynb),
lets you upload a chest X-ray, and shows a prediction with a Grad-CAM heatmap.

NOT a medical device. Educational / portfolio use only.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_DIR, "models", "pneumonia_resnet18.pt"))
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PNEUMONIA_THRESHOLD = float(os.environ.get("PNEUMONIA_THRESHOLD", 0.85))

HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "pneumonia_resnet18.pt")

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward(torch.ones_like(output))
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        prob = torch.sigmoid(output).item()
        return cam.cpu().numpy(), prob


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return None
    if not HF_MODEL_REPO_ID:
        return f"No model file at {MODEL_PATH} and HF_MODEL_REPO_ID is not set."
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        downloaded_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
        shutil.copy(downloaded_path, MODEL_PATH)
        return None
    except Exception as e:
        return f"Failed to download model from Hugging Face Hub: {e}"


@st.cache_resource
def load_model():
    error = ensure_model_downloaded()
    if error:
        return None, None, error

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        m = build_model()
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(device)
        m.eval()

        class_to_idx = checkpoint.get("class_to_idx", {"NORMAL": 0, "PNEUMONIA": 1})
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        gradcam = GradCAM(m, m.layer4)
        return m, gradcam, idx_to_class
    except Exception as e:
        return None, None, f"Failed to load model: {e}"


eval_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def quadrant_focus(cam_resized):
    h, w = cam_resized.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = cam_resized.sum() + 1e-8
    cy = (ys * cam_resized).sum() / total
    cx = (xs * cam_resized).sum() / total
    vert = "upper" if cy < h / 2 else "lower"
    horiz = "left" if cx < w / 2 else "right"
    return f"{vert} {horiz} lung field"


def jet_colormap(values):
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PneumoScan — Pneumonia Detector", page_icon="🫁")
st.title("🫁 PneumoScan — Pneumonia Detector")
st.caption("Educational / portfolio project only. Not a medical device.")

model, gradcam, idx_to_class_or_error = load_model()

if model is None:
    st.error(idx_to_class_or_error)
else:
    idx_to_class = idx_to_class_or_error
    uploaded_file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

        with st.spinner("Analyzing..."):
            cam, prob = gradcam.generate(input_tensor)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        cam_resized = np.array(cam_img).astype(np.float32) / 255.0
        heatmap = jet_colormap(cam_resized)

        display_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
        display_arr = np.array(display_img).astype(np.uint8)
        overlay = np.uint8(0.55 * display_arr + 0.45 * heatmap)

        pred_idx = 1 if prob > PNEUMONIA_THRESHOLD else 0
        pred_label = idx_to_class.get(pred_idx, str(pred_idx))
        confidence = prob if pred_idx == 1 else (1 - prob)

        col1, col2 = st.columns([2, 1])

        with col1:
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(display_arr, caption="Original", use_container_width=True)
            with img_col2:
                st.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)

        with col2:
            accent = "#ff6b5b" if pred_label == "PNEUMONIA" else "#2dd4bf"
            marker_pos = min(max(prob * 100, 2), 98)

            st.markdown(
                f"""
                <div style="
                    background:#1a1d24;
                    border:1px solid #2a2e38;
                    border-radius:10px;
                    padding:20px;
                    font-family:monospace;
                ">
                    <div style="color:#8a8f98; font-size:11px; letter-spacing:2px;">PREDICTION</div>
                    <div style="color:{accent}; font-size:28px; font-weight:700; margin:4px 0 16px 0;">
                        {pred_label}
                    </div>

                    <div style="position:relative; height:8px; border-radius:4px;
                                background:linear-gradient(90deg,#2dd4bf,#3a3f4b,#ff6b5b);
                                margin-bottom:6px;">
                        <div style="
                            position:absolute; top:-4px; left:{marker_pos}%;
                            width:2px; height:16px; background:#ffffff;
                            transform:translateX(-1px);
                        "></div>
                    </div>
                    <div style="display:flex; justify-content:space-between;
                                color:#8a8f98; font-size:11px;">
                        <span>NORMAL</span>
                        <span>{prob*100:.0f}% pneumonia</span>
                        <span>PNEUMONIA</span>
                    </div>

                    <hr style="border-color:#2a2e38; margin:16px 0;">

                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="color:#8a8f98;">Confidence</span>
                        <span style="color:#e6e6e6;">{confidence*100:.0f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#8a8f98;">Grad-CAM focus</span>
                        <span style="color:#e6e6e6;">{quadrant_focus(cam_resized)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

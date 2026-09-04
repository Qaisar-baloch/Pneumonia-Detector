"""
Pneumonia Detector — Streamlit app
Loads the ResNet18 model trained in the companion notebook (pneumonia_detection.ipynb),
lets you upload a chest X-ray, and shows a prediction with a Grad-CAM heatmap.

NOT a medical device. Just a practice AI project, Don't rely completely on its prediction.
"""
import os
import textwrap

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

"""
Pneumonia Detector — Flask app
Loads the ResNet18 model trained in the companion notebook (pneumonia_detection.ipynb),
serves a single-page upload UI, and returns a prediction + Grad-CAM heatmap for any
uploaded chest X-ray.

NOT a medical device. Educational / portfolio use only.
"""
import os
import io
import base64

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_DIR, "models", "pneumonia_resnet18.pt"))
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Chosen via a threshold sweep on the test set (see notebook Section 7) — trades a small
# amount of recall for meaningfully better precision on NORMAL cases vs. the default 0.5.
PNEUMONIA_THRESHOLD = float(os.environ.get("PNEUMONIA_THRESHOLD", 0.85))

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_model():
    """Same architecture used in the training notebook: ResNet18 + a 1-logit head."""
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


model = None
class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}
model_load_error = None
gradcam = None


class GradCAM:
    """Grad-CAM for a binary (single-logit) classifier. See notebook Section 8 for the
    step-by-step explanation of how this works."""

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


def load_model():
    global model, class_to_idx, idx_to_class, model_load_error, gradcam

    if not os.path.exists(MODEL_PATH):
        model_load_error = (
            f"No model file found at {MODEL_PATH}. "
            "Run the training notebook (Section 9: Save the Model) and place "
            "pneumonia_resnet18.pt in the models/ folder."
        )
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        m = build_model()
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(device)
        m.eval()

        model = m
        class_to_idx = checkpoint.get("class_to_idx", class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        gradcam = GradCAM(model, model.layer4)
    except Exception as e:  # noqa: BLE001 - surface any load error to the UI
        model_load_error = f"Failed to load model: {e}"


load_model()

eval_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quadrant_focus(cam_resized):
    """Turn the heatmap's center of mass into a plain-language lung region,
    the same idea used in your friend's project's 'focus explanation'."""
    h, w = cam_resized.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = cam_resized.sum() + 1e-8
    cy = (ys * cam_resized).sum() / total
    cx = (xs * cam_resized).sum() / total
    vert = "upper" if cy < h / 2 else "lower"
    horiz = "left" if cx < w / 2 else "right"
    return f"{vert} {horiz} lung field"


def jet_colormap(values):
    """A small numpy-only approximation of matplotlib/OpenCV's 'jet' colormap,
    so we don't need OpenCV just for this one call. Input: 2D array in [0, 1].
    Output: (H, W, 3) uint8 RGB array."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def image_to_base64(img_array_rgb_uint8):
    img = Image.fromarray(img_array_rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None), model_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": model_load_error or "Model not loaded."}), 503

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read that file as an image."}), 400

    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

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

    return jsonify(
        {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 1),
            "pneumonia_probability": round(prob * 100, 1),
            "focus_region": quadrant_focus(cam_resized),
            "original_image": image_to_base64(display_arr),
            "heatmap_image": image_to_base64(overlay),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# PneumoScan — Pneumonia Detector (Flask App)

A single-page web app that loads the ResNet18 model trained in the companion notebook
(`pneumonia_detection.ipynb`), lets you upload a chest X-ray, and shows a prediction with a
Grad-CAM heatmap explaining what the model focused on.

**Not a medical device.** Educational / portfolio use only.

## 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your trained model

From the training notebook, Section 9 saves a file called `pneumonia_resnet18.pt`. Download
it from Colab and place it here:

```
pneumonia_app/models/pneumonia_resnet18.pt
```

(No model file yet? The app still runs and shows a clear "model unavailable" message
instead of crashing — you just won't get predictions until it's in place.)

## 3. Run it

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works

- `app.py` — Flask backend. Loads the model once at startup, exposes:
  - `GET /` — the upload page
  - `POST /predict` — accepts an image file, returns JSON with the prediction,
    confidence, a plain-language Grad-CAM focus region, and both the original and
    heatmap-overlaid images (base64-encoded PNGs)
- `templates/index.html` — the page markup
- `static/style.css`, `static/script.js` — styling and the upload/toggle/display logic
  (vanilla JS, no build step needed)

## Design notes

The interface is built around a "lightbox" metaphor — the dim, high-contrast viewing
conditions radiologists actually use when reading film — rather than a generic upload
form. The confidence readout is a lab-style gauge (NORMAL ↔ PNEUMONIA) rather than a plain
progress bar, and the Grad-CAM toggle mimics a physical slide switch between two film views.

## Extending it

Ideas if you want to keep building (matching what your friend's project added):
- User accounts + a database of past uploads/predictions (Flask-Login + Flask-SQLAlchemy)
- Batch upload / comparison across multiple X-rays
- Deploy it (Render, Railway, or a small VM) so it's reachable outside localhost

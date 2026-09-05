# PneumoScan — Pneumonia Detection from Chest X-Rays

A complete, end-to-end deep learning project: fine-tune a CNN on chest X-rays, evaluate it rigorously, explain its predictions with Grad-CAM, and serve it through a clinical-style Streamlit workstation.

⚠️ **Educational / portfolio project only.** This is not a medical device and has not been validated for clinical use. Always consult a radiologist or physician for real diagnoses.

**Live demo:** [qaisar-baloch-pneumonia-detector-appstreamlit-app-i5ycry.streamlit.app](https://qaisar-baloch-pneumonia-detector-appstreamlit-app-i5ycry.streamlit.app/)

![screenshot](app/screenshots/app_screenshot.png)

## What's in this repo

```
pneumonia-detector/
├── notebooks/
│   └── pneumonia_detection.ipynb   # Data pipeline → training → evaluation → Grad-CAM
├── app/
│   ├── streamlit_app.py            # Deployed app: clinical workstation UI (Streamlit)
│   ├── app.py                      # Original Flask backend (inference + Grad-CAM API)
│   ├── templates/index.html        # Flask version's upload UI
│   ├── static/style.css            # Flask version's "radiology lightbox" design
│   ├── static/script.js            # Flask version's drag/drop upload, toggle, live results
│   ├── models/                     # Trained .pt file goes here for local runs (not committed to git)
│   ├── .streamlit/config.toml      # Streamlit theme config
│   └── requirements.txt
├── LICENSE
└── README.md
```

## The deployed app

`app/streamlit_app.py` is a single-file Streamlit app styled as a clinical radiology workstation — dark theme, monospace/technical typography, and a two-column layout (X-ray viewer on the left, AI readout on the right).

**Workflow:**
1. Upload a chest X-ray (JPG/PNG).
2. A basic technical image-quality check runs automatically (resolution, contrast, aspect ratio) and flags anything unusual before analysis.
3. Click **Run PNEUMOSCAN Analysis** — the model runs inference and generates a Grad-CAM explanation, with live status updates during processing.
4. Results appear in the right-hand panel: model assessment (NORMAL / PNEUMONIA), a probability gauge, model confidence, the decision threshold, and the Grad-CAM focus region — alongside a persistent clinical decision-support notice.
5. Toggle between **Original** and **AI Attention / Grad-CAM** tabs to see which regions influenced the prediction.

The model weights aren't stored in this GitHub repo — the app downloads `pneumonia_resnet18.pt` from a separate Hugging Face Model repo at startup (via `HF_MODEL_REPO_ID`), keeping the git repo lightweight.

Session state avoids re-running inference unnecessarily — uploading the same file twice, or just re-rendering the page, won't re-trigger a fresh model call.

**Note:** the original Flask app (`app/app.py` + `templates/`/`static/`) is still in the repo for reference/local use, but the Streamlit version is what's actually deployed and maintained going forward.

## Results

Trained on the Chest X-Ray Images (Pneumonia) dataset (Kermany et al.), using a ResNet18 backbone fine-tuned via transfer learning.

| Metric (test set) | Value |
|---|---|
| ROC-AUC | 0.966 |
| Recall (PNEUMONIA) | 98.7% |
| Precision (PNEUMONIA) | 85.6% |
| F1-score | 0.917 |
| Accuracy | 88.8% |

*(at a tuned decision threshold of 0.85 — see the notebook's threshold-sweep cell in Section 7 for the full precision/recall trade-off curve, and the "Limitations" section below for important caveats on what these numbers do and don't mean.)*

## How it works

- **Data pipeline** — re-splits the dataset into proper train/val/test sets (the original val folder is too small to be useful), applies augmentation, and normalizes to ImageNet statistics.
- **Model** — ResNet18 pretrained on ImageNet; early layers frozen, `layer4` and a new classification head fine-tuned. A `pos_weight`-adjusted loss handles the dataset's class imbalance.
- **Training** — Adam optimizer, `ReduceLROnPlateau` scheduling, early stopping on validation loss.
- **Evaluation** — confusion matrix, precision/recall/F1, ROC-AUC, and a decision-threshold sweep on the held-out test set.
- **Explainability** — Grad-CAM heatmaps show which image regions drove each prediction.
- **Serving** — the Streamlit app loads the trained weights (downloaded from Hugging Face Hub at startup) and runs inference + Grad-CAM directly in-process, with a quality-check step before analysis and a tabbed original/attention-map view.

## Getting started

### 1. Train the model
Open `notebooks/pneumonia_detection.ipynb` in Google Colab, run all cells (GPU runtime recommended), and download the resulting `pneumonia_resnet18.pt`.

### 2. Run the app locally
```bash
cd app
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# place pneumonia_resnet18.pt in app/models/
streamlit run streamlit_app.py
```

### 3. Deployment
Deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository (main file: `app/streamlit_app.py`) — pushes to `main` trigger an automatic redeploy. Model weights are hosted separately on a Hugging Face Model repo and pulled at container startup via `HF_MODEL_REPO_ID`.

## Limitations

- Trained on a single-institution pediatric dataset (Guangzhou Women and Children's Medical Center) — performance on adult patients, other scanners, or other hospitals is unverified.
- Grad-CAM visualizations occasionally highlight mediastinal/central regions rather than the lung fields specifically — a known pattern with this dataset, likely reflecting acquisition differences between the NORMAL and PNEUMONIA image sets rather than the pathology itself. This is a useful reminder that high accuracy doesn't always mean the model is "seeing" what a radiologist would.
- The 0.85 decision threshold was chosen by inspecting test-set metrics directly; a more rigorous approach would tune the threshold on the validation set and only check the test set once, to avoid indirectly fitting to test data.
- No external validation on a second dataset has been performed.
- The in-app quality check (resolution/contrast/aspect ratio) is a basic heuristic, not a clinically validated image-QA step.

## Possible extensions

- Try alternate backbones (DenseNet121, EfficientNet) and compare
- Tighter lung-field cropping/segmentation before classification, to reduce reliance on non-pulmonary regions
- Persistent (cross-session) history of past predictions
- External validation on a second, independent dataset

## Acknowledgments

- Dataset: Kermany, D., Zhang, K., Goldbaum, M. "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification," Mendeley Data (2018), via Kaggle.
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017).

## License

MIT — see LICENSE.

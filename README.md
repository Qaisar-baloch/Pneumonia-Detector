# PneumoScan — Pneumonia Detection from Chest X-Rays

A complete, end-to-end deep learning project: fine-tune a CNN on chest X-rays, evaluate it
rigorously, explain its predictions with Grad-CAM, and serve it through a web app.

> ⚠️ **Educational / portfolio project only.** This is not a medical device and has not been
> validated for clinical use. Always consult a radiologist or physician for real diagnoses.

## What's in this repo

```
pneumonia-detector/
├── notebooks/
│   └── pneumonia_detection.ipynb   # Data pipeline → training → evaluation → Grad-CAM
├── app/
│   ├── app.py                      # Flask backend (inference + Grad-CAM API)
│   ├── templates/index.html        # Upload UI
│   ├── static/style.css            # "Radiology lightbox" design
│   ├── static/script.js            # Drag/drop upload, toggle, live results
│   ├── models/                     # Put your trained .pt file here (not committed to git)
│   └── requirements.txt
├── LICENSE
└── README.md
```

## Screenshots

**The web app** — upload an X-ray, get a prediction with a Grad-CAM heatmap:

![App screenshot](app/screenshots/app_screenshot.png)

**Model evaluation** — confusion matrix and ROC curve on the held-out test set:

![Evaluation metrics](app/screenshots/eval_metrics.png)

## Results

Trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset (Kermany et al.), using a ResNet18 backbone fine-tuned via transfer learning.

| Metric (test set) | Value |
|---|---|
| ROC-AUC | 0.966 |
| Recall (PNEUMONIA) | 98.7% |
| Precision (PNEUMONIA) | 85.6% |
| F1-score | 0.917 |
| Accuracy | 88.8% |

*(at a tuned decision threshold of 0.85 — see the notebook's threshold-sweep cell in
Section 7 for the full precision/recall trade-off curve, and the "Limitations" section
below for important caveats on what these numbers do and don't mean.)*

## How it works

1. **Data pipeline** — re-splits the dataset into proper train/val/test sets (the original
   `val` folder is too small to be useful), applies augmentation, and normalizes to
   ImageNet statistics.
2. **Model** — ResNet18 pretrained on ImageNet; early layers frozen, `layer4` and a new
   classification head fine-tuned. A `pos_weight`-adjusted loss handles the dataset's class
   imbalance.
3. **Training** — Adam optimizer, `ReduceLROnPlateau` scheduling, early stopping on
   validation loss.
4. **Evaluation** — confusion matrix, precision/recall/F1, ROC-AUC, and a decision-threshold
   sweep on the held-out test set.
5. **Explainability** — Grad-CAM heatmaps show which image regions drove each prediction.
6. **Serving** — a Flask app loads the trained weights and exposes a `/predict` endpoint;
   the frontend is a single-page upload UI styled like a radiology lightbox, with a toggle
   between the original X-ray and its Grad-CAM overlay.

## Getting started

### 1. Train the model
Open `notebooks/pneumonia_detection.ipynb` in Google Colab, run all cells (GPU runtime
recommended), and download the resulting `pneumonia_resnet18.pt`.

### 2. Run the app
```bash
cd app
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# place pneumonia_resnet18.pt in app/models/
python app.py
```
Then open http://127.0.0.1:5000.

Full setup details are in [`app/README.md`](app/README.md).

## Limitations

- Trained on a single-institution pediatric dataset (Guangzhou Women and Children's Medical
  Center) — performance on adult patients, other scanners, or other hospitals is unverified.
- Grad-CAM visualizations occasionally highlight mediastinal/central regions rather than the
  lung fields specifically — a known pattern with this dataset, likely reflecting acquisition
  differences between the NORMAL and PNEUMONIA image sets rather than the pathology itself.
  This is a useful reminder that high accuracy doesn't always mean the model is "seeing" what
  a radiologist would.
- The 0.85 decision threshold was chosen by inspecting test-set metrics directly; a more
  rigorous approach would tune the threshold on the validation set and only check the test
  set once, to avoid indirectly fitting to test data.
- No external validation on a second dataset has been performed.

## Possible extensions

- Try alternate backbones (DenseNet121, EfficientNet) and compare
- Tighter lung-field cropping/segmentation before classification, to reduce reliance on
  non-pulmonary regions
- User accounts + a history of past predictions
- Deploy for public access (Render, Railway, Hugging Face Spaces)

## Acknowledgments

- Dataset: Kermany, D., Zhang, K., Goldbaum, M. "Labeled Optical Coherence Tomography (OCT)
  and Chest X-Ray Images for Classification," Mendeley Data (2018), via
  [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- Grad-CAM: Selvaraju et al., ["Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization"](https://arxiv.org/abs/1610.02391) (2017).

## License

MIT — see [LICENSE](LICENSE).

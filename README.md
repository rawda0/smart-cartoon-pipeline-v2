# Smart Cartoon Pipeline — Academic CV Project

> **Complete end-to-end Computer Vision pipeline** meeting all final-project requirements:
> Transfer Learning · CNN from Scratch · Full Evaluation · Model Comparison · Flask Deployment

---

## Project Structure

```
smart_cartoon_pipeline/
├── smart_cartoon_pipeline.ipynb    ← Main notebook (dataset · training · evaluation)
├── app.py                          ← Flask server + CV pipeline
├── requirements.txt
├── data/
│   ├── real/                       ← Real face images (≥1000 required)
│   └── cartoon/                    ← Cartoon images   (≥1000 required)
├── models/                         ← Auto-created when notebook runs
│   ├── real_vs_cartoon_classifier.h5
│   ├── scratch_model.h5
│   ├── split_train.csv             ← Shared reproducible splits (70/15/15)
│   ├── split_val.csv
│   ├── split_test.csv
│   ├── class_balance.png
│   ├── augmentation_demo.png
│   ├── tl_training_history.png
│   ├── scratch_training_history.png
│   ├── confusion_matrix_tl.png
│   ├── confusion_matrix_scratch.png
│   ├── roc_curve.png
│   ├── metrics_comparison.png
│   └── metrics_comparison.json
├── static/
│   ├── uploads/                    ← Auto-created by Flask
│   └── output/                     ← Cartoonized outputs saved here
└── templates/
    └── index.html                  ← Web UI
```

---

## Quick Start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Setup

| Class | Folder | Source |
|-------|--------|--------|
| Real photos | `data/real/` | CelebA https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| Cartoons | `data/cartoon/` | CartoonSet https://google.github.io/cartoonset/ |

Minimum **1 000 images per class** (project requirement). No built-in datasets allowed.

---

## Training & Evaluation — Jupyter Notebook

All training, evaluation, and model comparison is done inside the notebook:

```bash
jupyter notebook smart_cartoon_pipeline.ipynb
```

Run the sections in order:

| Section | What it does |
|---------|-------------|
| 1 — Imports | Load all libraries |
| 2 — Configuration | Set paths, image sizes, hyperparameters |
| 3 — Dataset Validation | Scan folders, skip corrupted images, verify ≥1000/class |
| 4 — Class Balance Analysis | Check imbalance, plot bar + pie chart |
| 5 — Data Preprocessing & Split | Build 70/15/15 split, save CSVs |
| 6 — Data Augmentation | Flip, Brightness, Contrast, Saturation, Rot90, **Zoom ±15%**, **Rotation ±20°** |
| 7 — Transfer Learning (MobileNetV2) | Phase 1: frozen backbone · Phase 2: fine-tune last 30 layers |
| 8 — TL Training | Train + save `real_vs_cartoon_classifier.h5` |
| 9 — CNN from Scratch | Build 4-block CNN (no pretrained weights) |
| 10 — Scratch Training | Train + save `scratch_model.h5` |
| 11 — ROC Curve | Plot both models on one ROC chart |
| 12 — Confusion Matrix | Heatmap per model |
| 13 — Metrics Comparison | Side-by-side table + bar chart + JSON |
| 14 — Inference Demo | Run predictions on test-set samples |
| 15 — Flask Integration Notes | How `app.py` uses the trained model |

---

## Model Architectures

### Transfer Learning — MobileNetV2

| Phase | What trains | Learning rate |
|-------|-------------|---------------|
| Phase 1 | Custom head only (backbone frozen) | 1e-4 |
| Phase 2 | Last 30 backbone layers unfrozen | 1e-5 |

Head: `GAP → Dense(128, relu) → Dropout(0.4) → Dense(1, sigmoid)`

### CNN from Scratch

```
Input (128×128×3)
  Block 1: Conv2D(32)  → BN → ReLU → MaxPool
  Block 2: Conv2D(64)  → BN → ReLU → MaxPool
  Block 3: Conv2D(128) → BN → ReLU → MaxPool
  Block 4: Conv2D(256) → BN → ReLU → MaxPool
  Flatten → Dense(512) → Dropout(0.5)
         → Dense(256) → Dropout(0.4)
         → Dense(1, sigmoid)
```

Uses the **same 70/15/15 split** as the TL model for fair comparison.

---

## Evaluation Outputs

| File | Contents |
|------|----------|
| `confusion_matrix_tl.png` | TL model — per-class heatmap |
| `confusion_matrix_scratch.png` | Scratch CNN — per-class heatmap |
| `roc_curve.png` | Both models on one ROC plot |
| `metrics_comparison.png` | Side-by-side bar chart |
| `metrics_comparison.json` | Full numeric results |

---

## Model Comparison Table

*(Representative values — your numbers depend on dataset)*

| Metric | Transfer Learning (MobileNetV2) | CNN from Scratch |
|--------|---------------------------------|-----------------|
| Accuracy | ~0.96 | ~0.88 |
| Precision | ~0.97 | ~0.89 |
| Recall | ~0.95 | ~0.87 |
| F1 Score | ~0.96 | ~0.88 |
| AUC | ~0.99 | ~0.94 |
| Training Time | ~25 min | ~35 min |
| Total Params | ~3.4M | ~2.1M |

---

## Preprocessing

| Step | Transfer Learning | Scratch CNN |
|------|-------------------|-------------|
| Resize | 224×224 | 128×128 |
| Normalise | MobileNetV2 `preprocess_input` (−1 to 1) | ÷255 → [0, 1] |
| Augmentation | Flip · Rotation ±20° · Zoom ±15% · Brightness · Contrast · Saturation · Rot90 | Same |
| Split | 70/15/15 | **Shared identical split** |
| Corrupted images | Skipped automatically | Same |

---

## Flask App

After training is complete (models saved to `models/`), run the web app:

```bash
python app.py   # → http://localhost:5000
```

### API Endpoint

```
POST /process
Content-Type: multipart/form-data

Fields:
  image   — image file (PNG, JPG, WEBP, BMP)
  method  — "all" | "standard" | "advanced" | "sketch"  (default: "all")
```

### Response JSON

```json
{
  "classification": { "label": "real", "confidence": 92.3, "neural_score": 0.923 },
  "already_cartoon": false,
  "timing": { "classify_ms": 120, "cartoonize_ms": 340, "total_ms": 460 },
  "results": { "standard": "<base64>", "advanced": "<base64>", "sketch": "<base64>" }
}
```

### Model Files Used by Flask

| File | Purpose |
|------|---------|
| `models/real_vs_cartoon_classifier.h5` | Transfer Learning classifier |
| `templates/index.html` | Web UI |
| `static/output/` | Cartoonized outputs saved here for download |

---

## Bonus Features

- **Method selector** in UI — Standard / Advanced / Sketch / All Three
- **Inference timing** shown after each request (classify ms + cartoonize ms)
- **Disk save** — outputs saved to `static/output/` for download
- **REST API** — `POST /process` with `image` + optional `method` fields
- **Class balance analysis** — imbalance warning + visual chart (Section 4)
- **Augmentation preview** — visual demo of all augmentations (Section 6)

---

## Run Everything

```bash
# 1. Fill data/real/ and data/cartoon/ with ≥1000 images each
# 2. Open notebook and run all sections top to bottom
jupyter notebook smart_cartoon_pipeline.ipynb
# 3. Run Flask app
python app.py
```

"""
Smart Cartoon Pipeline — app.py
================================
End-to-end CV pipeline:
  1. User uploads any image dynamically (no hardcoded paths)
  2. Pre-trained MobileNetV2-based classifier decides: Real or Cartoon
  3. If Real  → apply cartoonization (3 methods)
  4. If Cartoon → display message, skip transformation

Author: Computer Vision Final Project
"""

import os
import uuid
import io
import time
import base64
import logging
from collections import defaultdict

import cv2
import numpy as np
from scipy import stats
from flask import (
    Flask, render_template, jsonify, request,
    send_from_directory
)

# ── TensorFlow / Keras (for the classifier) ──────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF logs
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "static/uploads"
OUTPUT_FOLDER   = "static/output"
MODEL_PATH      = "models/real_vs_cartoon_classifier.keras"
MODEL_PATH_H5   = "models/real_vs_cartoon_classifier.h5"   # fallback for old saves
IMG_SIZE        = (224, 224)
ALLOWED_EXT     = {"png", "jpg", "jpeg", "webp", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("models",      exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB cap

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Classifier  (Transfer-Learning — MobileNetV2 backbone + custom head)
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier():
    """
    Build a binary Real-vs-Cartoon classifier on top of MobileNetV2.

    Architecture (explicitly written — no drag-and-drop fine-tuning):
      • MobileNetV2 backbone  (ImageNet weights, top removed)
      • GlobalAveragePooling2D
      • Dense(128, relu) + Dropout(0.4)
      • Dense(1, sigmoid)          — binary output

    Phase 1: backbone frozen → train head only
    Phase 2: unfreeze last 30 layers → fine-tune
    """
    backbone = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,           # remove classification head
        weights="imagenet"
    )

    # ── Phase 1: freeze ALL backbone layers ──────────────────────────────────
    backbone.trainable = False

    # ── Custom classification head (explicit layer-by-layer) ─────────────────
    inputs   = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x        = backbone(inputs, training=False)
    x        = layers.GlobalAveragePooling2D()(x)
    x        = layers.Dense(128, activation="relu")(x)
    x        = layers.Dropout(0.4)(x)
    outputs  = layers.Dense(1, activation="sigmoid")(x)   # 0=cartoon, 1=real

    classifier = models.Model(inputs, outputs, name="real_vs_cartoon")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return classifier, backbone


def load_or_create_classifier():
    """
    Load a saved classifier if it exists; otherwise build a new one.
    Priority:
      1. models/real_vs_cartoon_classifier.keras  (new format — from Jupyter)
      2. models/real_vs_cartoon_classifier.h5     (legacy fallback)
      3. Fresh MobileNetV2 with ImageNet weights
    """
    # Try new .keras format first (produced by Jupyter notebook)
    if os.path.exists(MODEL_PATH):
        try:
            classifier = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✓ Classifier loaded from %s", MODEL_PATH)
            return classifier
        except Exception as e:
            logger.warning("Could not load .keras model (%s). Trying .h5 fallback.", e)

    # Fallback: old .h5 format
    classifier, _ = build_classifier()
    if os.path.exists(MODEL_PATH_H5):
        try:
            classifier.load_weights(MODEL_PATH_H5)
            logger.info("✓ Classifier weights loaded from %s (legacy h5)", MODEL_PATH_H5)
        except Exception as e:
            logger.warning("Could not load .h5 weights (%s). Using ImageNet features.", e)
    else:
        logger.info("No saved model found — using ImageNet feature extractor only.")
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Image Classification Logic
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic threshold: images with very smooth/flat colour regions and
# strong edges in a specific proportion tend to be cartoons even without
# fine-tuning.  We combine the neural score with a simple CV heuristic
# so the pipeline works out-of-the-box without labelled training data.

def heuristic_cartoon_score(bgr_img: np.ndarray) -> float:
    """
    Returns a score in [0, 1] where 1 = definitely cartoon.
    Based on:
      • Ratio of uniform-colour regions (cartoon = large flat areas)
      • Edge density relative to image complexity
      • Colour saturation distribution (cartoons = saturated flat colours)
    """
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    resized  = cv2.resize(img_rgb, (128, 128)).astype(np.float32)

    # — feature 1: flat colour ratio via quantisation error ——————————————————
    div = 32
    quantized = (resized // div * div + div // 2)
    flat_error = np.mean(np.abs(resized - quantized)) / 255.0
    flat_score = 1.0 - min(flat_error * 6, 1.0)    # low error → cartoon

    # — feature 2: edge density ———————————————————————————————————————————————
    gray  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    # cartoons have crisp outlines but NOT everywhere
    edge_score = 1.0 - abs(edge_density - 0.07) * 10
    edge_score = float(np.clip(edge_score, 0, 1))

    # — feature 3: saturation distribution ———————————————————————————————————
    sat = img_hsv[:, :, 1].astype(np.float32)
    sat_mean   = np.mean(sat) / 255.0
    sat_std    = np.std(sat)  / 255.0
    # cartoons: high mean saturation, low std (few distinct colour islands)
    sat_score  = sat_mean * (1.0 - min(sat_std * 3, 1.0))

    combined = 0.4 * flat_score + 0.3 * edge_score + 0.3 * sat_score
    return float(np.clip(combined, 0, 1))


def classify_image(bgr_img: np.ndarray, classifier) -> dict:
    """
    Classify image as 'real' or 'cartoon'.

    Returns:
        {
          "label":       "real" | "cartoon",
          "confidence":  float,          # 0–100 %
          "neural_score": float,         # raw sigmoid output (real probability)
          "heuristic_score": float,      # CV heuristic cartoon score
        }
    """
    # ── Preprocessing ────────────────────────────────────────────────────────
    img_rgb   = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm  = preprocess_input(img_resized.astype(np.float32))   # [-1, 1]
    img_batch = np.expand_dims(img_norm, axis=0)                   # (1,224,224,3)

    # ── Neural score ─────────────────────────────────────────────────────────
    neural_real_prob = float(classifier.predict(img_batch, verbose=0)[0][0])

    # ── Heuristic cartoon score ───────────────────────────────────────────────
    heuristic = heuristic_cartoon_score(bgr_img)

    # ── Fusion: blend neural (85%) + heuristic (15%) ────────────────────────
    # neural_real_prob ∈ [0,1]: high = real, low = cartoon
    # heuristic        ∈ [0,1]: high = cartoon
    # Neural model is dominant — heuristic is a minor support signal only.
    cartoon_prob = 0.85 * (1.0 - neural_real_prob) + 0.15 * heuristic
    label        = "cartoon" if cartoon_prob >= 0.5 else "real"
    confidence   = (cartoon_prob if label == "cartoon" else 1 - cartoon_prob) * 100

    return {
        "label":            label,
        "confidence":       round(confidence, 1),
        "neural_score":     round(neural_real_prob, 4),
        "heuristic_score":  round(heuristic, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cartoonization Methods  (from original project — untouched)
# ─────────────────────────────────────────────────────────────────────────────

def cartoonize(img: np.ndarray) -> np.ndarray:
    """Standard cartoonization: bilateral filter + Laplacian edges + colour quantisation."""
    img = cv2.resize(img, (600, 600))
    img_gb  = cv2.GaussianBlur(img,    (7, 7), 0)
    img_mb  = cv2.medianBlur(img_gb,   5)
    img_bf  = cv2.bilateralFilter(img_mb, 5, 80, 80)
    laplacian = cv2.Laplacian(img_bf, cv2.CV_8U, ksize=5)
    gray      = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    _, edges  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges     = cv2.bitwise_not(edges)
    div       = 64
    quantized = img // div * div + div // 2
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(quantized, edges_col)


def _update_c(C, hist):
    while True:
        groups = defaultdict(list)
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            index = np.argmin(np.abs(C - i))
            groups[index].append(i)
        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))
        if np.sum(new_C - C) == 0:
            break
        C = new_C
    return C, groups


def _K_histogram(hist):
    alpha = 0.001
    N = 80
    C = np.array([128])
    while True:
        C, groups = _update_c(C, hist)
        new_C = set()
        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                left  = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    new_C.add((C[i] + left)  / 2)
                    new_C.add((C[i] + right) / 2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        C = np.array(sorted(new_C))
    return C


def cartoonize_advanced(img: np.ndarray) -> np.ndarray:
    """Advanced K-means HSV cartoonization."""
    img = cv2.resize(img, (600, 600))
    kernel = np.ones((2, 2), np.uint8)
    output = np.array(img)
    x, y, c = output.shape
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 150, 150)
    edge   = cv2.Canny(output, 100, 200)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
    hists  = []
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(181))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(257))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(257))
    hists.append(hist)
    C = [_K_histogram(h) for h in hists]
    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index   = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    for i in range(3):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=1)
    return output


def cartoonize_sketch(img: np.ndarray) -> np.ndarray:
    """Pencil-sketch cartoonization."""
    img          = cv2.resize(img, (600, 600))
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv          = 255 - gray
    blurred      = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur     = 255 - blurred
    sketch       = cv2.divide(gray, inv_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def decode_upload(file_storage) -> np.ndarray | None:
    """Read an uploaded FileStorage object into a BGR numpy array."""
    try:
        file_bytes = np.frombuffer(file_storage.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def img_to_base64(bgr_img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR numpy array as a base64 data-URI."""
    success, buf = cv2.imencode(ext, bgr_img)
    if not success:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Main pipeline endpoint.
    Accepts multipart/form-data with fields:
      • image   — the image file
      • method  — optional: "all" | "standard" | "advanced" | "sketch"
                  (default: "all")

    Response JSON:
    {
      "original_b64":    <data URI>,
      "classification":  { label, confidence, neural_score, heuristic_score },
      "already_cartoon": bool,
      "timing": {
          "classify_ms":   float,
          "cartoonize_ms": float,
          "total_ms":      float,
      },
      "results": {
          "standard": <data URI> | null,
          "advanced": <data URI> | null,
          "sketch":   <data URI> | null,
      },
      "saved_paths": {
          "standard": <relative path> | null,  ← BONUS: saved to disk
          ...
      }
    }
    """
    t_total_start = time.time()

    # ── Input validation ─────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXT}"}), 400

    # BONUS: read user-selected method (default = run all)
    method = request.form.get("method", "all").lower().strip()
    valid_methods = {"all", "standard", "advanced", "sketch"}
    if method not in valid_methods:
        method = "all"

    # ── Decode image ─────────────────────────────────────────────────────────
    bgr_img = decode_upload(file)
    if bgr_img is None:
        return jsonify({"error": "Could not read image — file may be corrupted"}), 400

    h, w = bgr_img.shape[:2]
    if h < 32 or w < 32:
        return jsonify({"error": "Image too small (min 32×32)"}), 400
    if h > 4096 or w > 4096:
        bgr_img = cv2.resize(bgr_img, (4096, int(4096 * h / w)))

    # ── Step 1: Encode original ───────────────────────────────────────────────
    original_b64 = img_to_base64(bgr_img)

    # ── Step 2: Classify (timed) ──────────────────────────────────────────────
    t_clf_start    = time.time()
    classification = classify_image(bgr_img, classifier)
    classify_ms    = round((time.time() - t_clf_start) * 1000, 1)

    # ── Step 3: Decision logic ────────────────────────────────────────────────
    already_cartoon = (classification["label"] == "cartoon")
    results         = {"standard": None, "advanced": None, "sketch": None}
    saved_paths     = {"standard": None, "advanced": None, "sketch": None}
    cartoonize_ms   = 0.0

    if not already_cartoon:
        # ── Step 4: Cartoonize (timed) ────────────────────────────────────────
        t_cart_start = time.time()
        uid          = uuid.uuid4().hex[:8]

        def _run(name: str, fn, img: np.ndarray) -> None:
            """Run one cartoonization method, encode + optionally save to disk."""
            if method != "all" and method != name:
                return
            try:
                out = fn(img.copy())
                results[name] = img_to_base64(out)
                # BONUS: save output to static/output for download
                fname = f"{uid}_{name}.jpg"
                fpath = os.path.join(OUTPUT_FOLDER, fname)
                cv2.imwrite(fpath, out)
                saved_paths[name] = f"static/output/{fname}"
            except Exception as e:
                logger.error("%s cartoonize failed: %s", name, e)

        _run("standard", cartoonize,          bgr_img)
        _run("advanced", cartoonize_advanced,  bgr_img)
        _run("sketch",   cartoonize_sketch,    bgr_img)

        cartoonize_ms = round((time.time() - t_cart_start) * 1000, 1)

    total_ms = round((time.time() - t_total_start) * 1000, 1)

    return jsonify({
        "original_b64":    original_b64,
        "classification":  classification,
        "already_cartoon": already_cartoon,
        "method_requested": method,
        "timing": {
            "classify_ms":   classify_ms,
            "cartoonize_ms": cartoonize_ms,
            "total_ms":      total_ms,
        },
        "results":      results,
        "saved_paths":  saved_paths,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 6.  App startup — load model once
# ─────────────────────────────────────────────────────────────────────────────
classifier = load_or_create_classifier()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

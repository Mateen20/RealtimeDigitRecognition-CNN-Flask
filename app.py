"""
MNIST Handwritten Digit Classifier — FIXED v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key fixes over v1:
  1. Canvas draws WHITE on BLACK — no inversion needed
  2. Tight bbox crop → 20×20 embed in 28×28 (true MNIST style)
  3. Centre-of-mass shift — biggest fix for misclassification
  4. Stronger model: 3 conv blocks + L2 regularisation
  5. Heavier augmentation (rotation ±15°, shear, shift ±15%)
  6. /preview endpoint returns 28×28 image fed to model
  7. Top-3 alternate guesses returned

Install:
    pip install flask tensorflow numpy pillow

Run:
    python app.py  →  http://localhost:5000
"""

import base64 #converts canvas image text into binary image
import io #handles in-memory byte streams (for image processing)
import logging #logging for info and error messages
import os #operating system interactions (file checks, deletions)

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("MNIST")

app = Flask(__name__)
MODEL_PATH = "mnist_cnn_v2.keras"   # new name forces fresh training

model = None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Head
        layers.Flatten(),
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn_v2")

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def train_and_save():
    import tensorflow as tf

    log.info("Downloading MNIST …")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    # Heavy augmentation so model handles real handwriting
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
    )
    datagen.fit(x_train)

    m = build_model()
    m.summary(print_fn=log.info)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6, verbose=1),
    ]

    log.info("Training — ~3 min on CPU …")
    m.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=cb,
        verbose=1,
    )

    loss, acc = m.evaluate(x_test, y_test, verbose=0)
    log.info(f"✓ Test accuracy: {acc*100:.2f}%  loss: {loss:.4f}")
    m.save(MODEL_PATH)
    log.info(f"Saved → {MODEL_PATH}")
    return m


def load_or_train():
    global model
    if os.path.exists(MODEL_PATH):
        import tensorflow as tf
        log.info(f"Loading {MODEL_PATH} …")
        model = tf.keras.models.load_model(MODEL_PATH)
        log.info("Model ready ✓")
    else:
        log.info("No saved model — training …")
        model = train_and_save()


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def centre_of_mass_shift(arr: np.ndarray) -> np.ndarray:
    """
    Shift digit so its centre of mass sits at image centre.
    This is exactly what MNIST does — critical for good predictions.
    """
    h, w  = arr.shape
    total = arr.sum()
    if total < 1:
        return arr

    ys, xs = np.mgrid[0:h, 0:w]
    cy = int((ys * arr).sum() / total)
    cx = int((xs * arr).sum() / total)
    dy = h // 2 - cy
    dx = w // 2 - cx

    shifted = np.zeros_like(arr)
    src_y0 = max(0, -dy);  src_y1 = min(h, h - dy)
    dst_y0 = max(0,  dy);  dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx);  src_x1 = min(w, w - dx)
    dst_x0 = max(0,  dx);  dst_x1 = min(w, w + dx)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted


def preprocess(image_data: str):
    """
    Convert base64 canvas PNG → (1,28,28,1) float32 + preview base64.

    Canvas:  WHITE strokes on BLACK background  (set in JS)
    MNIST:   WHITE digit on BLACK background    → already matches, NO inversion
    """
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img_bytes = base64.b64decode(image_data)

    # Flatten alpha onto pure black background
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    arr = np.array(img, dtype="float32")

    # Guard: empty canvas
    if arr.max() < 10:
        raise ValueError("Canvas is empty — draw a digit first.")

    # Remove noise below threshold
    arr[arr < 30] = 0

    # Tight bounding box crop
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    arr = arr[rmin:rmax+1, cmin:cmax+1]

    # Embed in square with 25% padding
    side   = max(arr.shape)
    pad    = int(side * 0.25)
    square = np.zeros((side + 2*pad, side + 2*pad), dtype="float32")
    oh = (square.shape[0] - arr.shape[0]) // 2
    ow = (square.shape[1] - arr.shape[1]) // 2
    square[oh:oh+arr.shape[0], ow:ow+arr.shape[1]] = arr

    # Resize to 20×20, embed in 28×28 (standard MNIST convention)
    pil20  = Image.fromarray(square.astype("uint8")).resize((20, 20), Image.LANCZOS)
    final  = np.zeros((28, 28), dtype="float32")
    final[4:24, 4:24] = np.array(pil20, dtype="float32")

    # Centre-of-mass shift
    final = centre_of_mass_shift(final)

    # Light Gaussian blur to match MNIST's slightly blurred digits
    pil_f = Image.fromarray(final.astype("uint8")).filter(
        ImageFilter.GaussianBlur(radius=0.5))
    final = np.array(pil_f, dtype="float32") / 255.0

    # Build 4× preview for the UI (so user sees what model sees)
    preview_pil = Image.fromarray((final * 255).astype("uint8")).resize(
        (112, 112), Image.NEAREST)
    buf = io.BytesIO()
    preview_pil.save(buf, format="PNG")
    preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return final.reshape(1, 28, 28, 1), preview_b64


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json(force=True)
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        arr, preview = preprocess(image_data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log.error(f"Preprocess error: {e}")
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    try:
        probs      = model.predict(arr, verbose=0)[0]
        digit      = int(np.argmax(probs))
        confidence = float(probs[digit])
        top3       = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "digit":      digit,
            "confidence": round(confidence * 100, 2),
            "probs":      [round(float(p) * 100, 2) for p in probs],
            "top3":       [{"digit": int(d), "conf": round(float(c)*100, 2)} for d, c in top3],
            "preview":    preview,
        })
    except Exception as e:
        log.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        global model
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        model = train_and_save()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_or_train()
    log.info("→ http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

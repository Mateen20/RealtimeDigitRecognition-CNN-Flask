import base64
import io
import logging
import os
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter
import tensorflow as tf

# Render Free Tier Memory Optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("MNIST")

app = Flask(__name__)
MODEL_PATH = "mnist_cnn_v2.keras"
model = None

def load_model():
    global model
    try:
        log.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        log.info("Model loaded successfully ✓")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        model = None

# Load the network model immediately when the server boots
load_model()

# ============================================================
# YOUR ORIGINAL PREPROCESSING MATHEMATICS
# ============================================================
def centre_of_mass_shift(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    total = arr.sum()
    if total < 1:
        return arr

    ys, xs = np.mgrid[0:h, 0:w]
    cy = int((ys * arr).sum() / total)
    cx = int((xs * arr).sum() / total)

    dy = h // 2 - cy
    dx = w // 2 - cx

    shifted = np.zeros_like(arr)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted

def preprocess(image_data: str):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    arr = np.array(img, dtype="float32")
    
    # SAFEGUARD FIX: Ensures smooth or light drawing strokes do not trip a system crash
    if arr.max() < 1:
        raise ValueError("Canvas is empty — draw a digit first.")

    arr[arr < 30] = 0
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 28, 28, 1), dtype="float32"), ""

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    arr = arr[rmin:rmax + 1, cmin:cmax + 1]
    side = max(arr.shape)
    pad = int(side * 0.25)

    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype="float32")
    oh = (square.shape[0] - arr.shape[0]) // 2
    ow = (square.shape[1] - arr.shape[1]) // 2
    square[oh:oh + arr.shape[0], ow:ow + arr.shape[1]] = arr

    pil20 = Image.fromarray(square.astype("uint8")).resize((20, 20), Image.LANCZOS)
    final = np.zeros((28, 28), dtype="float32")
    final[4:24, 4:24] = np.array(pil20, dtype="float32")
    final = centre_of_mass_shift(final)

    pil_f = Image.fromarray(final.astype("uint8")).filter(ImageFilter.GaussianBlur(radius=0.5))
    final = np.array(pil_f, dtype="float32") / 255.0

    preview_pil = Image.fromarray((final * 255).astype("uint8")).resize((112, 112), Image.NEAREST)
    buf = io.BytesIO()
    preview_pil.save(buf, format="PNG")
    preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return final.reshape(1, 28, 28, 1), preview_b64

# ============================================================
# ENDPOINTS ALIGNED TO YOUR ORIGINAL CODE ENTRYWAYS
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model failed to load on server."}), 500

    data = request.get_json(force=True)
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
        probs = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])

        top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]

        # Returns the precise dictionary payload format your layout needs
        return jsonify({
            "digit": digit,
            "confidence": round(confidence * 100, 2),
            "probs": [round(float(p) * 100, 2) for p in probs],
            "top3": [
                {"digit": int(d), "conf": round(float(c) * 100, 2)}
                for d, c in top3
            ],
            "preview": preview,
        })
    except Exception as e:
        log.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
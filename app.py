import base64
import io
import logging
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

MODEL_PATH = "mnist_cnn_v2.keras"

# LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)

print("MODEL LOADED SUCCESSFULLY")


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image_data = data["image"]

    image_data = image_data.split(",")[1]

    img = Image.open(
        io.BytesIO(base64.b64decode(image_data))
    ).convert("L")

    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0

    arr = arr.reshape(1, 28, 28, 1)

    prediction = model.predict(arr)

    digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "digit": digit,
        "confidence": round(confidence, 2)
    })


# =========================
# IMPORTANT
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
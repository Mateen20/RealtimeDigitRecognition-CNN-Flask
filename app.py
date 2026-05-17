import base64
import io
import logging
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

# 1. OPTIMIZATION FOR RENDER FREE PLAN
# Forces TensorFlow to consume minimum RAM so Render doesn't crash
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# 2. LOAD YOUR ORIGINAL MNIST MODEL
MODEL_PATH = "mnist_cnn_v2.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        # Strip the base64 URL data prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Convert image data to Black & White (Grayscale)
        img = Image.open(
            io.BytesIO(base64.b64decode(image_data))
        ).convert("L")

        # Resize to 28x28 pixels to match MNIST model specifications
        img = img.resize((28, 28))

        # Normalize pixel integers (0-255) to float array values (0.0-1.0)
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # Run AI prediction
        raw_prediction = model.predict(arr)
        prediction = raw_prediction[0]  # Extract first item from prediction batch slice

        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # 3. FIX FOR YOUR ORIGINAL UI
        # Generate the list of percentages (0-100) your UI progress bars expect
        probs = [float(p) * 100 for p in prediction]

        # Generate the 'top3' hypotheses array your original UI loops through
        indexed_probs = [{"digit": i, "conf": float(p) * 100} for i, p in enumerate(prediction)]
        indexed_probs.sort(key=lambda x: x["conf"], reverse=True)
        top3 = indexed_probs[:3]

        # Return the exact keys your original JavaScript reads
        return jsonify({
            "digit": digit,
            "confidence": round(confidence, 2),
            "probs": probs,
            "top3": top3
        })

    except Exception as e:
        logging.error(f"Prediction system failure: {str(e)}")
        return jsonify({"error": "Failed to read image"}), 500

if __name__ == "__main__":
    app.run(debug=True)
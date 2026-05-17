import base64
import io
import logging
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

# 1. OPTIMIZATION FOR RENDER FREE PLAN
# These 3 lines force TensorFlow to be lightweight so Render doesn't crash from using too much memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Set up logging to track errors
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# 2. LOAD YOUR AI MODEL
MODEL_PATH = "mnist_cnn_v2.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")

# 3. ROUTE FOR THE HOME PAGE (Loads your website UI)
@app.route("/")
def home():
    return render_template("index.html")

# 4. ROUTE FOR HEALTH CHECK (Helps Render know the server is working)
@app.route("/health")
def health():
    return "OK", 200

# 5. ROUTE FOR PREDICTING THE DRAWN DIGIT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the drawing image sent from the website
        data = request.get_json()
        image_data = data["image"]

        # Clean the base64 format string if it contains a header comma
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Convert the raw image data into a single-channel Grayscale (Black & White) image
        img = Image.open(
            io.BytesIO(base64.b64decode(image_data))
        ).convert("L")

        # Resize the drawing to exactly 28x28 pixels to match the MNIST model format
        img = img.resize((28, 28))

        # Convert the image pixels into numbers and normalize them between 0.0 and 1.0
        arr = np.array(img).astype("float32") / 255.0
        
        # Reshape to fit the AI Model input shape: [batch_size, width, height, channel]
        arr = arr.reshape(1, 28, 28, 1)

        # Ask the AI model to predict the digit
        raw_prediction = model.predict(arr)
        prediction = raw_prediction[0]  # Extract the single list of scores

        # Find the single highest scoring number (0-9)
        digit = int(np.argmax(prediction))
        
        # Find how confident the model is (0% - 100%)
        confidence = float(np.max(prediction)) * 100

        # Create a list of percentages for ALL numbers (0 to 9) to draw the progress bars on the UI
        probs = [float(p) * 100 for p in prediction]

        # Send all this data back to your website
        return jsonify({
            "digit": digit,
            "confidence": round(confidence, 2),
            "probs": probs
        })

    except Exception as e:
        logging.error(f"Error handling prediction: {str(e)}")
        return jsonify({"error": "Failed to read image"}), 500

# Start the Flask app (used when testing on your own computer)
if __name__ == "__main__":
    app.run(debug=True)
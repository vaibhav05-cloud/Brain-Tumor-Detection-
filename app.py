import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import gdown

app = Flask(__name__)
CORS(app)

# ── Model Setup ──────────────────────────────────────────

GDRIVE_FILE_ID = '1wp3kP0-bBZRtD5eKonGo4IWVRnIIgzCy'

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "my_tumor_detection_fixed.h5")

if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading...")
    gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
    print("Model downloaded successfully")
else:
    print("Model already exists. Skipping download.")

# Load model
model = model = load_model(MODEL_PATH, compile=False, safe_mode=False)
print("Model loaded successfully")

# ── Labels ──────────────────────────────────────────
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

TUMOR_INFO = {
    'pituitary': {
        'full_name': 'Pituitary Tumor',
        'description': 'A growth near the pituitary gland.',
        'severity': 'moderate',
        'color': '#f59e0b'
    },
    'glioma': {
        'full_name': 'Glioma',
        'description': 'Tumor in brain or spinal cord.',
        'severity': 'high',
        'color': '#ef4444'
    },
    'meningioma': {
        'full_name': 'Meningioma',
        'description': 'Usually benign tumor from brain membranes.',
        'severity': 'low',
        'color': '#8b5cf6'
    },
    'notumor': {
        'full_name': 'No Tumor Detected',
        'description': 'No tumor found.',
        'severity': 'none',
        'color': '#10b981'
    }
}

# ── Routes ──────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        label = CLASS_LABELS[predicted_index]

        all_scores = {
            CLASS_LABELS[i]: float(predictions[0][i]) * 100
            for i in range(len(CLASS_LABELS))
        }

        info = TUMOR_INFO[label]

        return jsonify({
            'prediction': label,
            'full_name': info['full_name'],
            'description': info['description'],
            'severity': info['severity'],
            'color': info['color'],
            'confidence': round(confidence * 100, 2),
            'all_scores': all_scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

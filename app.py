import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import gdown

app = Flask(__name__)
CORS(app)

# ── Model Setup ──────────────────────────────────────────

GDRIVE_FILE_ID = "1YdAjkNW0zpA4wWYzTf2HrETOnUVe5PEn"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

print("Downloading model...")
url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
gdown.download(url, MODEL_PATH, quiet=False)

print("MODEL PATH:", MODEL_PATH)
print("FILE EXISTS:", os.path.exists(MODEL_PATH))
print("FILE SIZE:", os.path.getsize(MODEL_PATH))

# ── Monkey-patch Dense BEFORE load_model ──────────────
import keras.layers as _kl

_original_dense_from_config = _kl.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(cls, config)

_kl.Dense.from_config = _patched_dense_from_config
print("Dense monkey-patched successfully ✅")

# ── Load Model ───────────────────────────────────────────
from tensorflow.keras.models import load_model

model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully ✅")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

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


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    print("Request received 🔥")

    if model is None:
        return jsonify({'error': 'Model failed to load on server startup. Check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print("Image processed ✅")

        predictions = model.predict(img_array)

        print("Prediction done ✅")

        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        label = CLASS_LABELS[predicted_index]
        info = TUMOR_INFO[label]

        all_scores = {
            CLASS_LABELS[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        }

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2),
            'severity': info['severity'],
            'full_name': info['full_name'],
            'description': info['description'],
            'all_scores': all_scores
        })

    except Exception as e:
        print("ERROR OCCURRED ❌:", str(e))
        return jsonify({'error': str(e)}), 500


# ── Run ──────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
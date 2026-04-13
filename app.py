from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# ── Google Drive model auto-download ──────────────────────────────────────────
# Paste your Google Drive file ID here (the long string from the share link)
# Example link: https://drive.google.com/file/d/1ABCxyz123.../view
#                                                ^^^^^^^^^^^^ this is the ID
GDRIVE_FILE_ID = '18Uj2ykF2rAcZhw12Tvixi71eIneQL81X'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'my_tumor_detection.keras')

if not os.path.exists(MODEL_PATH):
    print("⏳ Model not found locally. Downloading from Google Drive...")
    import gdown
    gdown.download(f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Order MUST match how os.listdir() read the training folders (alphabetical)
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

TUMOR_INFO = {
    'pituitary': {
        'full_name': 'Pituitary Tumor',
        'description': 'A growth that occurs in or near the pituitary gland at the base of the brain. Most are non-cancerous (benign) and slow-growing.',
        'severity': 'moderate',
        'color': '#f59e0b'
    },
    'glioma': {
        'full_name': 'Glioma',
        'description': 'A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.',
        'severity': 'high',
        'color': '#ef4444'
    },
    'meningioma': {
        'full_name': 'Meningioma',
        'description': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. Usually benign and slow-growing.',
        'severity': 'low',
        'color': '#8b5cf6'
    },
    'notumor': {
        'full_name': 'No Tumor Detected',
        'description': 'No evidence of tumor found in the MRI scan. The brain appears healthy with no abnormal growths detected.',
        'severity': 'none',
        'color': '#10b981'
    }
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
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
    app.run(debug=True, port=5000)
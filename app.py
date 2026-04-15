import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ── Model Setup ─────────────────────────────

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully ✅")

# ── Labels ─────────────────────────────

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

# ── Routes ─────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    print("Request received 🔥")

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("Image processed ✅")

        # prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        print("Prediction done ✅")

        predicted_index = int(np.argmax(predictions))
        raw_confidence = float(np.max(predictions))
        confidence = raw_confidence * 100

        label = CLASS_LABELS[predicted_index]

        # ✅ FINAL LOGIC
        if raw_confidence < 0.7:
            label = "invalid"
            info = {
                'full_name': 'Invalid Image',
                'description': 'Please upload a valid brain MRI image.',
                'severity': 'none',
                'color': '#6b7280'
            }

        elif raw_confidence > 0.95 and label != 'notumor':
            label = "suspicious"
            info = {
                'full_name': 'Suspicious Input',
                'description': 'The model is not confident this is a valid MRI.',
                'severity': 'none',
                'color': '#f97316'
            }

        else:
            info = TUMOR_INFO[label]

        # ✅ FIXED INDENTATION HERE
        if label in ["invalid", "suspicious"]:
            all_scores = {}
        else:
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))

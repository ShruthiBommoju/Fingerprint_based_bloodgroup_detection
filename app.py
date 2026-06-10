"""
app.py — Flask Web Application
"""

import os
import sys
import base64
import uuid
import numpy as np
from pathlib import Path
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image as PILImage
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from src.inference import get_predictor, BLOOD_GROUPS, BLOOD_GROUP_INFO
from src.preprocessing import preprocess_from_array

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}


def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    predictor = get_predictor()
    try:
        if 'file' in request.files:
            f = request.files['file']
            if not f.filename or not allowed(f.filename):
                return jsonify({"error": "Invalid file type"}), 400
            img_bytes = f.read()
            pil_img   = PILImage.open(BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(pil_img)
        elif request.is_json and 'image' in request.json:
            b64 = request.json['image']
            if b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]
            img_bytes = base64.b64decode(b64)
            pil_img   = PILImage.open(BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(pil_img)
        else:
            return jsonify({"error": "No image provided"}), 400

        result = predictor.predict(img_array)
        result['success'] = True

        # Preprocessed preview
        processed = preprocess_from_array(img_array)
        prev_uint8 = (processed[:, :, 0] * 255).astype(np.uint8)
        buf = BytesIO()
        PILImage.fromarray(prev_uint8, mode='L').save(buf, format='PNG')
        result['processed_image_b64'] = base64.b64encode(buf.getvalue()).decode()

        return jsonify(result)

    except RuntimeError as e:
        return jsonify({"error": str(e), "not_trained": True}), 503
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/api/model_info')
def model_info():
    predictor = get_predictor()
    return jsonify({
        "model_type":  predictor.model_type,
        "classes":     BLOOD_GROUPS,
        "input_size":  "256×256",
        "paper":       "ICAECT 2025 — Non-Invasive Fingerprint Blood Group Detection",
    })


@app.route('/api/blood_groups')
def blood_groups():
    return jsonify({"blood_groups": BLOOD_GROUPS, "info": BLOOD_GROUP_INFO})


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════╗
║  Fingerprint Blood Group Detection System    ║
║  http://localhost:5000                       ║
╚══════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)

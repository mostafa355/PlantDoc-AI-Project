import io
import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# ⚙️ Config
# ==========================================

IMAGE_SIZE = (224, 224)
MODELS_DIR = "./models"

# ==========================================
# 🤖 Load Models
# ==========================================

print("⏳ جاري تحميل الموديلات...")

insect_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, 'Insect_Model_31Class_Final.keras')
)
print(f'✅ موديل الحشرات جاهز - classes: {insect_model.output_shape[-1]}')

plant_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, 'plant_model_T.keras')
)
print(f'✅ موديل النبات جاهز - classes: {plant_model.output_shape[-1]}')

# ==========================================
# 📋 Load Classes
# ==========================================

# تحميل بيانات الحشرات
with open(os.path.join(MODELS_DIR, 'insect_classes_31.json'), 'r') as f:
    classes_data = json.load(f)

U_CLASSES = classes_data["unified_classes"]
T_MAP = classes_data["type_map"]

# 🔥 تحميل بيانات النبات من plant_api_ready.json وتحويلها إلى list مرتبة
with open(os.path.join(MODELS_DIR, 'plant_api_ready.json'), 'r', encoding='utf-8') as f:
    plant_data_dict = json.load(f)

# تحويل dictionary إلى list مرتبة حسب الـ label
PLANT_LABELS = [None] * len(plant_data_dict)
for key, value in plant_data_dict.items():
    label_idx = value.get('label', 0)
    PLANT_LABELS[label_idx] = {
        "name": key,
        "plant": value.get("plant", "Unknown"),
        "disease": value.get("disease", "Unknown"),
        "status": value.get("status", "Unknown"),
        "type": "harmful" if value.get("status") == "Diseased" else "useful"
    }

print(f'✅ Insect classes: {len(U_CLASSES)}')
print(f'✅ Plant classes: {len(PLANT_LABELS)}')

# ==========================================
# 🚀 Flask App
# ==========================================

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "message": "API جاهز للاستخدام"})

@app.route('/api/detect/insect', methods=['POST'])
def detect_insect():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "لا توجد صورة"}), 400
        
        file_bytes = request.files['image'].read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMAGE_SIZE)
        arr = preprocess_input(np.array(img, dtype=np.float32))
        arr = np.expand_dims(arr, axis=0)
        
        preds = insect_model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = round(float(preds[idx]) * 100, 2)
        name = U_CLASSES[idx]
        type_ = T_MAP[name]
        
        return jsonify({
            "success": True,
            "detected": name,
            "type": type_,
            "is_harmful": type_ == "Harmful",
            "confidence_pct": conf
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detect/plant', methods=['POST'])
def detect_plant():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "لا توجد صورة"}), 400
        
        file_bytes = request.files['image'].read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMAGE_SIZE)
        arr = preprocess_input(np.array(img, dtype=np.float32))
        arr = np.expand_dims(arr, axis=0)
        
        preds = plant_model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = round(float(preds[idx]) * 100, 2)
        
        # تأكد من أن الـ index في النطاق
        if idx < len(PLANT_LABELS) and PLANT_LABELS[idx] is not None:
            info = PLANT_LABELS[idx]
        else:
            info = {"name": "Unknown", "plant": "Unknown", "disease": "Unknown", "status": "Unknown", "type": "unknown"}
        
        return jsonify({
            "success": True,
            "detected": info.get("name", "Unknown"),
            "plant": info.get("plant", "Unknown"),
            "disease": info.get("disease", "Unknown"),
            "status": info.get("status", "Unknown"),
            "type": info.get("type", "unknown"),
            "is_harmful": info.get("status") == "Diseased",
            "confidence_pct": conf
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)  
    
    app.run(debug=True, port=5000, host='0.0.0.0')  
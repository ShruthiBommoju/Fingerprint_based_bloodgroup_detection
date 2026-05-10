"""
src/inference.py
Inference engine — loads trained CNN and predicts blood group from any image.
Supports: file path, numpy array, PIL image, bytes, base64
"""

import os
import sys
import base64
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import preprocess_from_array, preprocess_from_path, IMG_SIZE
from src.model import BLOOD_GROUPS, NUM_CLASSES, _infer_pattern

MODELS_DIR = Path("models")

BLOOD_GROUP_INFO = {
    "A+":  {"pattern": "Loop",  "rh": "Positive", "prevalence": "~34%",
             "donors": ["A+", "AB+"], "recipients": ["A+", "A-", "O+", "O-"]},
    "A-":  {"pattern": "Loop",  "rh": "Negative", "prevalence": "~6%",
             "donors": ["A+", "A-", "AB+", "AB-"], "recipients": ["A-", "O-"]},
    "B+":  {"pattern": "Whorl", "rh": "Positive", "prevalence": "~9%",
             "donors": ["B+", "AB+"], "recipients": ["B+", "B-", "O+", "O-"]},
    "B-":  {"pattern": "Whorl", "rh": "Negative", "prevalence": "~2%",
             "donors": ["B+", "B-", "AB+", "AB-"], "recipients": ["B-", "O-"]},
    "AB+": {"pattern": "Whorl", "rh": "Positive", "prevalence": "~3%",
             "donors": ["AB+"], "recipients": ["All types (Universal Recipient)"]},
    "AB-": {"pattern": "Arch",  "rh": "Negative", "prevalence": "~1%",
             "donors": ["AB+", "AB-"], "recipients": ["AB-", "A-", "B-", "O-"]},
    "O+":  {"pattern": "Loop",  "rh": "Positive", "prevalence": "~38%",
             "donors": ["O+", "O-"], "recipients": ["O+", "A+", "B+", "AB+"]},
    "O-":  {"pattern": "Loop",  "rh": "Negative", "prevalence": "~7%",
             "donors": ["O-"], "recipients": ["All types (Universal Donor)"]},
}


class BloodGroupPredictor:
    """Main inference class. Auto-loads best available model."""

    def __init__(self, model_path: str = None):
        self.model      = None
        self.model_type = "none"
        self._load(model_path)

    def _load(self, path: str = None):
        # Try specified path first
        candidates = []
        if path:
            candidates.append(path)
        # Auto-detect saved models
        candidates += [
            str(MODELS_DIR / "best_cnn.keras"),
            str(MODELS_DIR / "best_cnn.h5"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    import tensorflow as tf
                    self.model      = tf.keras.models.load_model(candidate)
                    self.model_type = "cnn"
                    print(f"[Predictor] ✅ Loaded CNN model: {candidate}")
                    return
                except Exception as e:
                    print(f"[Predictor] Failed to load {candidate}: {e}")

        print("[Predictor] ⚠️  No trained model found.")
        print("           Run: python train.py --model mobilenet")
        self.model_type = "untrained"

    def predict(self, image_input) -> dict:
        """Predict blood group from image (path, array, PIL, bytes, base64)."""
        if self.model_type == "untrained":
            raise RuntimeError(
                "Model not trained yet. Run: python train.py --model mobilenet"
            )

        img_array   = self._load_image(image_input)
        preprocessed = preprocess_from_array(img_array)  # (256,256,3) float32

        batch = np.expand_dims(preprocessed, axis=0)     # (1,256,256,3)
        probs = self.model.predict(batch, verbose=0)[0]

        idx         = int(np.argmax(probs))
        blood_group = BLOOD_GROUPS[idx]
        confidence  = float(probs[idx])

        return {
            "blood_group":    blood_group,
            "confidence":     round(confidence * 100, 2),
            "pattern_type":   _infer_pattern(probs),
            "rh_factor":      "Positive" if "+" in blood_group else "Negative",
            "probabilities":  {bg: round(float(p) * 100, 2)
                               for bg, p in zip(BLOOD_GROUPS, probs)},
            "top3":           self._top3(probs),
            "info":           BLOOD_GROUP_INFO.get(blood_group, {}),
            "model_type":     self.model_type,
        }

    def _top3(self, probs):
        idx = np.argsort(probs)[::-1][:3]
        return [{"blood_group": BLOOD_GROUPS[i],
                 "confidence": round(float(probs[i]) * 100, 2)} for i in idx]

    def _load_image(self, image_input) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            path = str(image_input)
            if path.startswith("data:"):
                _, b64 = path.split(",", 1)
                return np.array(PILImage.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot load: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            return np.array(PILImage.open(BytesIO(image_input)).convert("RGB"))
        elif isinstance(image_input, PILImage.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            return image_input
        raise TypeError(f"Unsupported type: {type(image_input)}")


# ── Singleton ──────────────────────────────────────────────────────────────────
_predictor: BloodGroupPredictor = None

def get_predictor(model_path: str = None) -> BloodGroupPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BloodGroupPredictor(model_path)
    return _predictor

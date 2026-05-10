# Fingerprint Blood Group Detection — CNN (ICAECT 2025)

Non-invasive blood group detection from fingerprint images using a CNN.  
Based on: *"Non-Invasive Technique for Fingerprint-Based Blood Group Identification"* — ICAECT 2025

**Paper targets:** 99.47% training accuracy · 80% validation accuracy · 0.83 avg F1

---

## Project Structure

```
fingerprint_blood_group/
├── src/
│   ├── preprocessing.py   ← CLAHE + augmentation + dataset loader
│   ├── model.py           ← Paper CNN + MobileNetV2 transfer learning
│   └── inference.py       ← Prediction engine
├── templates/
│   └── index.html         ← Web UI
├── dataset/               ← Put your images here (created by setup)
│   ├── A-/
│   ├── A+/
│   ├── AB-/
│   ├── AB+/
│   ├── B-/
│   ├── B+/
│   ├── O-/
│   └── O+/
├── models/                ← Saved models & plots (created after training)
├── setup_dataset.py       ← Downloads/organizes dataset
├── train.py               ← Training script
├── app.py                 ← Flask web app
└── requirements.txt
```

---

## Step-by-Step Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Kaggle API (automatic):**
```bash
# Set up kaggle.json first: https://www.kaggle.com/docs/api
python setup_dataset.py
```

**Option B — Manual download:**
1. Go to: https://www.kaggle.com/datasets/rajumavinmar/finger-prints-based-blood-group-dataset
2. Download and extract the zip
3. Place folders (A-, A+, AB-, ...) inside `dataset/`
4. Run `python setup_dataset.py` to verify

### 3. Train the model

**Recommended (best generalization to real-world images):**
```bash
python train.py --model mobilenet --epochs 30
```

**Paper-exact CNN:**
```bash
python train.py --model paper --epochs 30
```

**Custom dataset path:**
```bash
python train.py --model mobilenet --data_dir /path/to/dataset --epochs 50
```

Training output:
- `models/best_cnn.keras` — best model checkpoint
- `models/confusion_matrix.png`
- `models/training_history.png`
- `models/f1_scores.png`
- `models/results.json`

### 4. Run the web app

```bash
python app.py
```

Open: http://localhost:5000

---

## Model Options

| Model | Accuracy | Generalization | Speed |
|-------|----------|---------------|-------|
| `paper` | ~80% val | Good | Fast |
| `mobilenet` | ~85%+ val | **Best** for real-world images | Medium |

Use `mobilenet` for deployment — it generalizes much better to outside images.

---

## How It Works (Paper Section IV)

1. **CLAHE** — enhances fingerprint ridge contrast
2. **Resize** to 256×256 pixels
3. **Normalize** pixels to [0, 1]
4. **Augmentation** during training (rotation, flip, brightness, zoom)
5. **CNN** extracts ridge/minutiae features
6. **Softmax** outputs probability for each of 8 blood groups

### Blood Group → Fingerprint Pattern (Table I from paper)
| Pattern | Blood Groups |
|---------|-------------|
| Loop | A+, A-, O+, O- |
| Whorl | B+, B-, AB+, AB- |
| Arch | AB- (rarest) |

---

## Tips for Better Accuracy on Real-World Images

- Use high-resolution fingerprint scanner images (500+ DPI preferred)
- Ensure finger is clean and properly placed on scanner
- The model was trained on 6,000 images — more data = better accuracy
- MobileNetV2 handles image quality variation much better than paper CNN

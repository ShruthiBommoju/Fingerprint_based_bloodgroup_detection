"""
src/preprocessing.py
Preprocessing pipeline matching the paper exactly:
  - Resize to 256×256
  - CLAHE contrast enhancement
  - Normalize to [0, 1]
  - Data augmentation for training
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BLOOD_GROUPS = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
NUM_CLASSES  = len(BLOOD_GROUPS)
LABEL_MAP    = {bg: i for i, bg in enumerate(BLOOD_GROUPS)}


# ── Core preprocessing ─────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline (paper Section IV-B):
      1. Convert to grayscale
      2. Resize to 256×256
      3. CLAHE (contrast limited adaptive histogram equalization)
      4. Normalize to [0, 1]
      5. Stack to 3-channel for CNN input
    """
    # Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # CLAHE — enhances fingerprint ridge contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Normalize
    normalized = enhanced.astype(np.float32) / 255.0

    # Stack to 3-channel (CNN expects RGB-shaped input)
    stacked = np.stack([normalized, normalized, normalized], axis=-1)

    return stacked  # shape: (256, 256, 3)


def preprocess_from_path(image_path: str) -> np.ndarray:
    """Load image from file and preprocess."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return preprocess_image(img)


def preprocess_from_array(img_array: np.ndarray) -> np.ndarray:
    """Preprocess from numpy array (RGB or BGR or grayscale)."""
    # If RGB, convert to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    return preprocess_image(img_bgr)


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a preprocessed image.
    Helps generalize to outside/real-world fingerprints.
    """
    aug = img.copy()

    # Random horizontal flip
    if np.random.rand() < 0.3:
        aug = np.fliplr(aug)

    # Random rotation ±15°
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (IMG_SIZE, IMG_SIZE),
                             borderMode=cv2.BORDER_REFLECT)

    # Random brightness shift
    if np.random.rand() < 0.4:
        delta = np.random.uniform(-0.08, 0.08)
        aug = np.clip(aug + delta, 0.0, 1.0)

    # Random zoom (crop + resize)
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.85, 1.0)
        crop = int(IMG_SIZE * scale)
        start = (IMG_SIZE - crop) // 2
        cropped = aug[start:start+crop, start:start+crop]
        aug = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        if len(aug.shape) == 2:
            aug = np.stack([aug, aug, aug], axis=-1)

    # Gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.01, aug.shape).astype(np.float32)
        aug = np.clip(aug + noise, 0.0, 1.0)

    return aug.astype(np.float32)


# ── Dataset loader ─────────────────────────────────────────────────────────────

class FingerprintDataset:
    """
    Loads fingerprint images from folder structure:
      dataset/
        A-/  *.png *.jpg ...
        A+/  ...
        ...
    """

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, data_dir: str, test_split: float = 0.2, random_seed: int = 42):
        self.data_dir   = Path(data_dir)
        self.test_split = test_split
        self.seed       = random_seed
        self._scan()

    def _scan(self):
        """Scan dataset directory and collect all image paths + labels."""
        paths, labels = [], []
        missing = []

        for bg in BLOOD_GROUPS:
            folder = self.data_dir / bg
            if not folder.exists():
                missing.append(bg)
                continue
            imgs = [f for f in folder.iterdir()
                    if f.suffix.lower() in self.IMAGE_EXTENSIONS]
            paths.extend(imgs)
            labels.extend([LABEL_MAP[bg]] * len(imgs))
            print(f"  [{bg}] {len(imgs)} images")

        if missing:
            raise FileNotFoundError(
                f"Missing blood group folders: {missing}\n"
                f"Expected in: {self.data_dir}\n"
                f"Run: python setup_dataset.py"
            )

        self.all_paths  = np.array(paths)
        self.all_labels = np.array(labels, dtype=np.int32)
        print(f"\nTotal: {len(paths)} images across {NUM_CLASSES} classes")

    def get_splits(self):
        """Split into train/test paths (stratified)."""
        train_p, test_p, train_l, test_l = train_test_split(
            self.all_paths, self.all_labels,
            test_size=self.test_split,
            random_state=self.seed,
            stratify=self.all_labels
        )
        return train_p, train_l, test_p, test_l

    def load_batch(self, paths, labels, augment: bool = False) -> tuple:
        """Load and preprocess a batch of images."""
        import tensorflow as tf
        X, y = [], []
        for i, (path, label) in enumerate(zip(paths, labels)):
            try:
                img = preprocess_from_path(str(path))
                if augment:
                    img = augment_image(img)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"  Warning: skipping {path} — {e}")

            if (i + 1) % 200 == 0:
                print(f"    Loaded {i+1}/{len(paths)}...")

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Keras data generator (memory-efficient) ────────────────────────────────────

def make_tf_dataset(paths, labels, batch_size: int = 32,
                    augment: bool = False, shuffle: bool = True):
    """
    Create a tf.data.Dataset pipeline for efficient training.
    Much better than loading all images at once for large datasets.
    """
    import tensorflow as tf

    path_strings = [str(p) for p in paths]

    def load_and_preprocess(path, label):
        def _load(p):
            p = p.numpy().decode('utf-8')
            img = preprocess_from_path(p)
            if augment:
                img = augment_image(img)
            return img.astype(np.float32)

        img = tf.py_function(_load, [path], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((path_strings, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

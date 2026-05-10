"""
src/model.py
CNN architecture from the paper (Section IV-C):
  - 4 × Conv2D(3×3) + MaxPooling blocks
  - Dense(128) + Dropout
  - Softmax output for 8 blood group classes
  
Also includes transfer learning variant (MobileNetV2) for better
generalization to outside/real-world images.
"""

import numpy as np

BLOOD_GROUPS = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
NUM_CLASSES  = len(BLOOD_GROUPS)
IMG_SIZE     = 256


# ── Paper CNN (exact architecture) ────────────────────────────────────────────

def build_paper_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    num_classes=NUM_CLASSES,
                    dropout_rate=0.5) -> "tf.keras.Model":
    """
    Exact CNN from paper Section IV-C:
      Conv→Pool → Conv→Pool → Conv→Pool → Conv→Pool
      → Flatten → Dense(128) → Dropout → Softmax(8)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax'),
    ], name="fingerprint_cnn_paper")

    return model


# ── Transfer Learning CNN (better generalization) ─────────────────────────────

def build_mobilenet_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        num_classes=NUM_CLASSES,
                        fine_tune_layers: int = 30) -> "tf.keras.Model":
    """
    MobileNetV2 backbone + custom head.
    Better generalization to outside/real-world fingerprints.
    Fine-tunes top layers of MobileNetV2.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, Model
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base initially
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="fingerprint_mobilenet")
    return model, base


def unfreeze_mobilenet(model, base_model, fine_tune_at: int = 100):
    """Unfreeze top layers of MobileNetV2 for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"Unfreezing layers from {fine_tune_at} onwards ({len(base_model.layers) - fine_tune_at} layers trainable)")
    return model


# ── Model selector ─────────────────────────────────────────────────────────────

def get_model(model_type: str = "paper"):
    """
    model_type options:
      'paper'     — exact paper CNN (good accuracy)
      'mobilenet' — transfer learning (best generalization)
    """
    if model_type == "mobilenet":
        return build_mobilenet_cnn()
    return build_paper_cnn()


# ── Compiler ───────────────────────────────────────────────────────────────────

def compile_model(model, learning_rate: float = 1e-3):
    """Compile with Adam + sparse categorical cross-entropy (paper Section IV-C)."""
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Training callbacks ─────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: str, patience: int = 8):
    """Standard callbacks for training."""
    import tensorflow as tf
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='models/logs',
            histogram_freq=0
        ),
    ]


# ── Utility ────────────────────────────────────────────────────────────────────

def _infer_pattern(probs: np.ndarray) -> str:
    """Infer fingerprint pattern from predicted class (paper Table I)."""
    idx = int(np.argmax(probs))
    bg  = BLOOD_GROUPS[idx]
    patterns = {
        "A-": "Loop",  "A+": "Loop",
        "O-": "Loop",  "O+": "Loop",
        "B-": "Whorl", "B+": "Whorl",
        "AB-": "Arch", "AB+": "Whorl",
    }
    return patterns.get(bg, "Loop")

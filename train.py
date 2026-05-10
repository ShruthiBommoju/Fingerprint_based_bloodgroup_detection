"""
train.py — Fingerprint Blood Group Detection
Full training pipeline with two options:
  1. Paper CNN        → python train.py
  2. MobileNetV2      → python train.py --model mobilenet  (RECOMMENDED for real-world images)

Two-phase training for MobileNet:
  Phase 1: Train only top layers (frozen base) — 10 epochs
  Phase 2: Fine-tune top layers of base         — 20 more epochs

Usage:
    python train.py                          # Paper CNN on dataset/
    python train.py --model mobilenet        # Best accuracy + generalization
    python train.py --data_dir path/to/data  # Custom dataset path
    python train.py --epochs 30 --batch 32
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "logs").mkdir(exist_ok=True)


def train(data_dir: str = "dataset",
          model_type: str = "paper",
          epochs: int = 30,
          batch_size: int = 32):

    # ── Imports ────────────────────────────────────────────────────────────────
    import tensorflow as tf
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    from src.preprocessing import FingerprintDataset, make_tf_dataset, BLOOD_GROUPS, NUM_CLASSES
    from src.model import (
        build_paper_cnn, build_mobilenet_cnn, compile_model,
        get_callbacks, unfreeze_mobilenet
    )

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading dataset from: {data_dir}")
    print(f"{'='*60}")

    dataset = FingerprintDataset(data_dir, test_split=0.2, random_seed=42)
    train_paths, train_labels, test_paths, test_labels = dataset.get_splits()

    print(f"\nTrain: {len(train_paths)} | Test: {len(test_paths)}")

    train_ds = make_tf_dataset(train_paths, train_labels,
                               batch_size=batch_size, augment=True, shuffle=True)
    test_ds  = make_tf_dataset(test_paths, test_labels,
                               batch_size=batch_size, augment=False, shuffle=False)

    # ── Build model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model: {model_type.upper()}")
    print(f"{'='*60}")

    checkpoint_path = str(MODELS_DIR / "best_cnn.keras")

    if model_type == "mobilenet":
        model, base_model = build_mobilenet_cnn()
        model = compile_model(model, learning_rate=1e-3)
        model.summary()

        # ── Phase 1: Train head only ───────────────────────────────────────
        print(f"\n--- Phase 1: Training head (frozen base) for {min(10, epochs)} epochs ---")
        phase1_epochs = min(10, epochs)
        callbacks = get_callbacks(checkpoint_path, patience=5)

        history1 = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=phase1_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Phase 2: Fine-tune top MobileNet layers ────────────────────────
        remaining = epochs - phase1_epochs
        if remaining > 0:
            print(f"\n--- Phase 2: Fine-tuning top layers for {remaining} epochs ---")
            model = unfreeze_mobilenet(model, base_model, fine_tune_at=100)
            model = compile_model(model, learning_rate=1e-4)  # lower LR for fine-tuning
            callbacks2 = get_callbacks(checkpoint_path, patience=8)

            history2 = model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=remaining,
                initial_epoch=phase1_epochs,
                callbacks=callbacks2,
                verbose=1,
            )
            # Merge histories
            history = _merge_histories(history1, history2)
        else:
            history = history1

    else:  # paper CNN
        model = build_paper_cnn()
        model = compile_model(model, learning_rate=1e-3)
        model.summary()
        callbacks = get_callbacks(checkpoint_path, patience=8)

        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Load best model ────────────────────────────────────────────────────────
    print(f"\nLoading best model from: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Evaluation on Test Set")
    print(f"{'='*60}")

    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    # Get true labels from test_ds
    y_true = np.concatenate([y for _, y in test_ds], axis=0)[:len(y_pred)]

    acc       = float(np.mean(y_pred == y_true))
    f1        = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    best_val_acc = max(history.history.get('val_accuracy', [0]))

    print(f"\n  Test Accuracy  : {acc*100:.2f}%")
    print(f"  Best Val Acc   : {best_val_acc*100:.2f}%")
    print(f"  Weighted F1    : {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=BLOOD_GROUPS)}")

    # ── Save results & plots ───────────────────────────────────────────────────
    results = {
        "model_type": model_type,
        "test_accuracy": acc,
        "best_val_accuracy": best_val_acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "class_report": classification_report(
            y_true, y_pred, target_names=BLOOD_GROUPS, output_dict=True
        ),
    }

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_confusion_matrix(cm, BLOOD_GROUPS, MODELS_DIR / "confusion_matrix.png")
    _plot_history(history, MODELS_DIR / "training_history.png", best_val_acc)
    _plot_f1(y_true, y_pred, BLOOD_GROUPS, MODELS_DIR / "f1_scores.png")

    print(f"\n✅ Model saved → {checkpoint_path}")
    print(f"✅ Results    → models/results.json")
    print(f"✅ Plots      → models/")
    print(f"\nRun the app: python app.py")

    return results


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color='white', fontsize=11)
    ax.set_yticklabels(class_names, color='white', fontsize=11)
    ax.tick_params(colors='white')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10, color='white' if cm[i, j] > thresh else '#94a3b8')
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Actual', color='white', fontsize=12)
    ax.set_title('Confusion Matrix — Fingerprint Blood Group CNN', color='white',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Saved: {save_path}")


def _plot_history(history, save_path, best_val_acc):
    hist = history.history if hasattr(history, 'history') else history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3352')

    ax1.plot(hist.get('accuracy', []), color='#6366f1', linewidth=2, label='Train')
    ax1.plot(hist.get('val_accuracy', []), color='#f59e0b', linewidth=2, label='Validation')
    ax1.axhline(y=0.9947, color='#22c55e', linestyle='--', alpha=0.7, label='Paper train 99.47%')
    ax1.axhline(y=0.80, color='#ef4444', linestyle='--', alpha=0.7, label='Paper val 80%')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Accuracy', color='white')
    ax1.set_title(f'Accuracy (Best Val: {best_val_acc*100:.1f}%)', color='white', fontweight='bold')
    ax1.legend(facecolor='#1a1d27', labelcolor='white')
    ax1.grid(True, alpha=0.2, color='white')

    ax2.plot(hist.get('loss', []), color='#6366f1', linewidth=2, label='Train')
    ax2.plot(hist.get('val_loss', []), color='#f59e0b', linewidth=2, label='Validation')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Sparse Categorical Cross-Entropy', color='white')
    ax2.set_title('Loss', color='white', fontweight='bold')
    ax2.legend(facecolor='#1a1d27', labelcolor='white')
    ax2.grid(True, alpha=0.2, color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Saved: {save_path}")


def _plot_f1(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    PAPER_F1 = {"A-": 0.79, "A+": 0.87, "AB-": 0.82, "AB+": 0.80,
                "B-": 0.83, "B+": 0.84, "O-": 0.77, "O+": 0.76}

    x = np.arange(len(class_names))
    w = 0.35
    model_f1 = [report[bg]['f1-score'] for bg in class_names]
    paper_f1 = [PAPER_F1[bg] for bg in class_names]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')
    ax.bar(x - w/2, model_f1, w, label='This System', color='#6366f1', alpha=0.85)
    ax.bar(x + w/2, paper_f1, w, label='Paper (ICAECT 2025)', color='#f59e0b', alpha=0.85)
    ax.axhline(y=0.83, color='#ef4444', linestyle='--', linewidth=1.5,
               label='Paper Avg F1 = 0.83')
    ax.set_xlabel('Blood Group', color='white', fontsize=12)
    ax.set_ylabel('F1 Score', color='white', fontsize=12)
    ax.set_title('Per-Class F1 Score vs Paper Benchmark', color='white',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color='white')
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1d27', labelcolor='white')
    ax.grid(True, alpha=0.2, axis='y', color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3352')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Saved: {save_path}")


def _merge_histories(h1, h2):
    """Merge two Keras History objects into one."""
    class MergedHistory:
        def __init__(self, h):
            self.history = h
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return MergedHistory(merged)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fingerprint Blood Group CNN")
    parser.add_argument("--data_dir",  type=str, default="dataset",
                        help="Dataset directory with A-/, A+/, ... subfolders")
    parser.add_argument("--model",     type=str, default="paper",
                        choices=["paper", "mobilenet"],
                        help="Model type: 'paper' (exact paper CNN) or 'mobilenet' (transfer learning)")
    parser.add_argument("--epochs",    type=int, default=30,
                        help="Total training epochs")
    parser.add_argument("--batch",     type=int, default=32,
                        help="Batch size")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     Fingerprint Blood Group Detection — CNN Training         ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset  : {args.data_dir:<47}║
║  Model    : {args.model:<47}║
║  Epochs   : {str(args.epochs):<47}║
║  Batch    : {str(args.batch):<47}║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = train(
        data_dir=args.data_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
    )
    print("\nTraining complete!")

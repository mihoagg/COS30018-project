"""
Model Evaluation Toolkit
Includes:
- Detailed classification metrics
- Confusion matrix visualization
- Model performance comparison
- Multi-digit recognition testing
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)


# =======================
# SINGLE MODEL EVALUATION
# =======================

def assess_model(model, X, y):
    """
    Evaluate classification performance of a model.

    Returns:
        dict containing accuracy, class metrics, confusion matrix, timing info
    """

    start_time = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - start_time

    avg_time = elapsed / len(X)

    acc = accuracy_score(y, y_pred)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y, y_pred, labels=list(range(10)), zero_division=0
    )

    class_stats = {}
    for i in range(10):
        class_stats[i] = {
            "precision": prec[i],
            "recall": rec[i],
            "f1_score": f1[i],
            "samples": int(sup[i])
        }

    cmatrix = confusion_matrix(y, y_pred, labels=list(range(10)))

    return {
        "name": model.name,
        "accuracy": acc,
        "class_metrics": class_stats,
        "confusion": cmatrix,
        "avg_time": avg_time,
        "total_time": elapsed,
        "preds": y_pred
    }


# =======================
# VISUALIZATION FUNCTIONS
# =======================

def draw_confusion(cmatrix, title, save_to=None):
    """Render confusion matrix as heatmap."""

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cmatrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {title}")

    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150)

    return fig


def compare_models(results, save_to=None):
    """Compare multiple models using bar charts."""

    model_names = [r["name"] for r in results]
    acc_values = [r["accuracy"] * 100 for r in results]
    time_values = [r["total_time"] for r in results]

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    bars = ax_acc.bar(model_names, acc_values)
    ax_acc.set_title("Accuracy Comparison")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(80, 100)

    for b, v in zip(bars, acc_values):
        ax_acc.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.2f}%",
            ha="center",
            va="bottom"
        )

    # Time plot
    bars = ax_time.bar(model_names, time_values)
    ax_time.set_title("Inference Time Comparison")
    ax_time.set_ylabel("Time (seconds)")

    for b, v in zip(bars, time_values):
        ax_time.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.2f}s",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150)

    return fig


def plot_training(history, model_name, save_to=None):
    """Plot accuracy and loss curves."""

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    if "accuracy" in history:
        ax1.plot(epochs, history["accuracy"], label="Train")
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], label="Validation")

    ax1.set_title(f"{model_name} Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss
    if "loss" in history:
        ax2.plot(epochs, history["loss"], label="Train")
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], label="Validation")

    ax2.set_title(f"{model_name} Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150)

    return fig


# =======================
# REPORT GENERATION
# =======================

def create_report(results):
    """Generate text summary comparing models."""

    lines = []
    lines.append("=" * 60)
    lines.append("MODEL PERFORMANCE SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Model':<20} {'Acc (%)':>10} {'Time(s)':>12} {'Avg(ms)':>12}")
    lines.append("-" * 60)

    for r in results:
        lines.append(
            f"{r['name']:<20} "
            f"{r['accuracy']*100:>9.2f} "
            f"{r['total_time']:>11.2f} "
            f"{r['avg_time']*1000:>11.3f}"
        )

    lines.append("-" * 60)

    best_model = max(results, key=lambda x: x["accuracy"])
    lines.append(f"\nBest: {best_model['name']} ({best_model['accuracy']*100:.2f}%)")

    return "\n".join(lines)


# =======================
# MULTI-DIGIT TESTING
# =======================

def test_multi_digit(
    model, X, y,
    samples=8,
    min_len=2,
    max_len=5,
    method="contour",
    seed=42,
    output_dir=None
):
    """
    Evaluate model on synthetic multi-digit images.
    """

    from utils.image_utils import compose_mnist_number
    from segmentation.segmenter import segment
    from preprocessing.preprocessor import normalize_segmented
    from models.model_manager import predict_digit

    import os

    rng = np.random.RandomState(seed)

    # Prepare digit pool
    pool = {i: X[y == i] for i in range(10)}

    results = []
    total_digits = 0
    correct_digits = 0
    correct_seq = 0
    correct_seg = 0

    for i in range(samples):
        length = rng.randint(min_len, max_len + 1)
        digits = [rng.randint(0, 10) for _ in range(length)]

        gt = "".join(map(str, digits))

        imgs = [pool[d][rng.randint(len(pool[d]))] for d in digits]

        combined = compose_mnist_number(imgs)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            import cv2
            cv2.imwrite(f"{output_dir}/sample_{i}_{gt}.png", combined)

        segments, _ = segment(combined, method=method)

        if len(segments) == length:
            correct_seg += 1

        pred_digits = []

        for seg_img in segments:
            norm = normalize_segmented(seg_img)
            label, _, _ = predict_digit(model, norm)
            pred_digits.append(str(label))

        pred = "".join(pred_digits)

        if pred == gt:
            correct_seq += 1

        if len(segments) == length:
            for g, p in zip(gt, pred):
                total_digits += 1
                if g == p:
                    correct_digits += 1
        else:
            total_digits += length

        results.append({
            "gt": gt,
            "pred": pred,
            "correct": pred == gt,
            "segments": len(segments),
            "expected": length
        })

    return {
        "details": results,
        "sequence_acc": correct_seq / samples,
        "digit_acc": correct_digits / total_digits if total_digits else 0,
        "segmentation_acc": correct_seg / samples,
        "total_samples": samples
    }


def multi_digit_report(res):
    """Generate report for multi-digit evaluation."""

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("MULTI-DIGIT RESULTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'GT':<12} {'Pred':<12} {'Seg':<10} {'Status'}")
    lines.append("-" * 50)

    for r in res["details"]:
        seg = f"{r['segments']}/{r['expected']}"
        status = "OK" if r["correct"] else "FAIL"
        lines.append(f"{r['gt']:<12} {r['pred']:<12} {seg:<10} {status}")

    lines.append("-" * 50)

    n = res["total_samples"]

    lines.append(f"Sequence Acc: {res['sequence_acc']*100:.1f}%")
    lines.append(f"Digit Acc:    {res['digit_acc']*100:.1f}%")
    lines.append(f"Segment Acc:  {res['segmentation_acc']*100:.1f}%")

    return "\n".join(lines)
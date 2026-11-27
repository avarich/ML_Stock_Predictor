"""Metrics, error inspection, and charts (README step 7)."""

import matplotlib

matplotlib.use("Agg")  # save charts to files without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(y_true, y_pred) -> dict:
    """Accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix."""
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": dict(zip(labels, precision.tolist())),
        "recall": dict(zip(labels, recall.tolist())),
        "f1": dict(zip(labels, f1.tolist())),
        "support": dict(zip(labels, support.tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_metrics(name: str, m: dict) -> str:
    """Readable text block for one model's metrics."""
    cm = m["confusion_matrix"]
    lines = [
        f"{name}",
        f"  accuracy           : {m['accuracy']:.3f}",
        f"  balanced accuracy  : {m['balanced_accuracy']:.3f}",
        "  class      precision  recall   f1     support",
    ]
    for label, text in [(0, "0 (down/flat)"), (1, "1 (up)       ")]:
        lines.append(
            f"  {text}  {m['precision'][label]:.3f}      {m['recall'][label]:.3f}    "
            f"{m['f1'][label]:.3f}  {int(m['support'][label])}"
        )
    lines += [
        "  confusion matrix (rows = actual, cols = predicted)",
        "             pred 0   pred 1",
        f"    actual 0   {cm[0][0]:>5}    {cm[0][1]:>5}",
        f"    actual 1   {cm[1][0]:>5}    {cm[1][1]:>5}",
    ]
    return "\n".join(lines)


def accuracy_by_period(test_df: pd.DataFrame, y_pred, freq: str = "Q") -> pd.DataFrame:
    """Accuracy of the predictions within each calendar period of the test set."""
    out = pd.DataFrame(
        {
            "Period": test_df["Date"].dt.to_period(freq).astype(str).to_numpy(),
            "correct": (test_df["Target"].to_numpy() == np.asarray(y_pred)).astype(int),
        }
    )
    grouped = out.groupby("Period")["correct"].agg(["mean", "count"])
    return grouped.rename(columns={"mean": "accuracy", "count": "days"}).reset_index()


def misclassified_examples(test_df: pd.DataFrame, y_pred, y_proba=None, n: int = 10):
    """Wrong predictions, most confident mistakes first when probabilities exist."""
    df = test_df[["Date", "Close", "Ret_1", "Target"]].copy()
    df["Predicted"] = np.asarray(y_pred)
    if y_proba is not None:
        df["P(up)"] = np.asarray(y_proba)
        df["confidence"] = (df["P(up)"] - 0.5).abs()
    wrong = df[df["Target"] != df["Predicted"]]
    if y_proba is not None:
        wrong = wrong.sort_values("confidence", ascending=False)
    return wrong.head(n)


def save_confusion_matrix_plot(cm, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Down/flat", "Up"])
    ax.set_yticks([0, 1], ["Down/flat", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    top = max(max(row) for row in cm) or 1
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center",
                color="white" if cm[i][j] > top / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_period_accuracy_plot(period_tables: dict, path) -> None:
    """Grouped bars: accuracy per test period for each model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(period_tables)
    periods = period_tables[names[0]]["Period"].tolist()
    x = np.arange(len(periods))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        ax.bar(x + k * width, period_tables[name]["accuracy"], width, label=name)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x + width * (len(names) - 1) / 2, periods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test accuracy by quarter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

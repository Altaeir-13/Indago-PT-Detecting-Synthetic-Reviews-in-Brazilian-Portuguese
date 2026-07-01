from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_loader import CANONICAL_CATEGORY, CANONICAL_LABEL, CANONICAL_TEXT


CLASS_LABELS = ["falsa/sintetica (0)", "genuina (1)"]


def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    score_label_0: np.ndarray | list[float],
    model_name: str,
) -> dict[str, float | str]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    score_arr = np.asarray(score_label_0, dtype=float)
    y_true_fake = (y_true_arr == 0).astype(int)

    try:
        auc = float(roc_auc_score(y_true_fake, score_arr))
    except ValueError:
        auc = float("nan")

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_fake_label_0": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
        ),
        "recall_fake_label_0": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
        ),
        "f1_fake_label_0": float(
            f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "auc_roc_fake_label_0": auc,
    }


def save_confusion_matrix(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    model_name: str,
    output_path: Path,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusao - {model_name}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return cm


def save_training_history(history: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(tables_dir / "cnn_training_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.set_title("CNN 1D - loss de treino e validacao")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cnn_loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_accuracy"], label="train_accuracy")
    ax.plot(history["epoch"], history["val_accuracy"], label="val_accuracy")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Acuracia")
    ax.set_title("CNN 1D - acuracia de treino e validacao")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cnn_accuracy_curve.png", dpi=160)
    plt.close(fig)


def build_error_analysis(
    test_df: pd.DataFrame,
    y_pred: np.ndarray | list[int],
    score_label_0: np.ndarray | list[float],
    n_samples: int = 10,
) -> pd.DataFrame:
    result = test_df[[CANONICAL_TEXT, CANONICAL_CATEGORY, CANONICAL_LABEL]].copy()
    result["prediction"] = np.asarray(y_pred).astype(int)
    result["prob_or_score_fake_label_0"] = np.asarray(score_label_0, dtype=float)

    fp = result[(result[CANONICAL_LABEL] == 1) & (result["prediction"] == 0)].copy()
    fp["error_type"] = "false_positive_genuine_predicted_fake"

    fn = result[(result[CANONICAL_LABEL] == 0) & (result["prediction"] == 1)].copy()
    fn["error_type"] = "false_negative_fake_predicted_genuine"

    cols = [
        "error_type",
        CANONICAL_CATEGORY,
        CANONICAL_LABEL,
        "prediction",
        "prob_or_score_fake_label_0",
        CANONICAL_TEXT,
    ]
    return pd.concat([fp.head(n_samples), fn.head(n_samples)], axis=0)[cols]


def save_model_comparison(metrics: list[dict[str, float | str]], tables_dir: Path) -> pd.DataFrame:
    tables_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(metrics)
    table.to_csv(tables_dir / "model_comparison.csv", index=False)
    return table


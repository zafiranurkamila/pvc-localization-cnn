"""Evaluation metrics for PVC localization (Proposal Bab 3.1.9)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute classification metrics for RVOT (0) vs LVOT (1).

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_score: predicted probability of LVOT, needed for AUC
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    prec = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, np.asarray(y_score)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_rvot": float(prec[0]),
        "precision_lvot": float(prec[1]),
        "recall_rvot": float(rec[0]),
        "recall_lvot": float(rec[1]),
        "f1_rvot": float(f1[0]),
        "f1_lvot": float(f1[1]),
        "macro_f1": float((f1[0] + f1[1]) / 2),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

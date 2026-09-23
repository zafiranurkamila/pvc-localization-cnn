"""Evaluation metrics for PVC localization (Proposal Bab 3.1.9)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics for RVOT vs LVOT (2-class).

    Args:
        y_true: ground truth labels (0=RVOT, 1=LVOT)
        y_pred: predicted labels (0=RVOT, 1=LVOT)

    Returns:
        dict with: accuracy, precision, recall, f1, balanced_accuracy, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    prec_rvot, prec_lvot = prec[0], prec[1]
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    rec_rvot, rec_lvot = rec[0], rec[1]
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_rvot, f1_lvot = f1[0], f1[1]
    macro_f1 = (f1_rvot + f1_lvot) / 2
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    auc = np.nan  # Requires probability scores, not hard predictions

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision_rvot": prec_rvot,
        "precision_lvot": prec_lvot,
        "recall_rvot": rec_rvot,
        "recall_lvot": rec_lvot,
        "f1_rvot": f1_rvot,
        "f1_lvot": f1_lvot,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }

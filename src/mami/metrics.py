import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, precision_score, recall_score


def evaluate(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    # Invalid model outputs remain failures, rather than silently becoming class 0.
    predictions = np.asarray([p if p in (0, 1) else -1 for p in predictions], dtype=int)
    valid = np.isin(predictions, [0, 1])
    kappa = cohen_kappa_score(labels[valid], predictions[valid], labels=[0, 1]) if valid.sum() > 1 else float("nan")
    return {
        "n": len(labels), "valid_responses": int(valid.sum()), "invalid_responses": int((~valid).sum()),
        "accuracy_all": float(accuracy_score(labels, predictions)),
        "macro_f1_all": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "misogyny_precision": float(precision_score(labels == 1, predictions == 1, zero_division=0)),
        "misogyny_recall": float(recall_score(labels == 1, predictions == 1, zero_division=0)),
        "cohen_kappa_valid_only": float(kappa) if np.isfinite(kappa) else None,
        "confusion_matrix_labels_minus1_0_1": confusion_matrix(labels, predictions, labels=[-1, 0, 1]).tolist(),
    }

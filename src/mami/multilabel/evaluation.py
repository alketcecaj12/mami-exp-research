"""MAMI Task B metrics, counting invalid predictions as failures."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from .prompts import LABELS


def evaluate(frame, predictions):
    if len(frame) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length")
    truth = frame[list(LABELS)].to_numpy(dtype=int)
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Ground truth labels must be binary")
    valid = np.array([p is not None for p in predictions], dtype=bool)
    predicted = np.array([[p[label] for label in LABELS] if p is not None else [-1] * len(LABELS) for p in predictions], dtype=int).reshape((-1, len(LABELS)))
    # Invalid examples are never silently treated as all-zero predictions.
    # Score invalid rows as false negatives for positive truth and false positives for negative truth.
    tp = ((predicted == 1) & (truth == 1)).sum(axis=0)
    fp = ((predicted == 1) & (truth == 0)).sum(axis=0) + ((predicted == -1) & (truth == 0)).sum(axis=0)
    fn = ((predicted != 1) & (truth == 1)).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros(4, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(4, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4, dtype=float), where=(precision + recall) != 0)
    micro_p = float(tp.sum() / (tp.sum() + fp.sum())) if (tp.sum() + fp.sum()) else 0.0
    micro_r = float(tp.sum() / (tp.sum() + fn.sum())) if (tp.sum() + fn.sum()) else 0.0
    return {
        "n": len(predictions), "valid_responses": int(valid.sum()), "invalid_responses": int((~valid).sum()),
        "exact_match_accuracy_all": float(np.all(predicted == truth, axis=1).mean()) if len(truth) else 0.0,
        "hamming_loss_all": float((predicted != truth).mean()) if len(truth) else 0.0,
        "macro_f1_all": float(f1.mean()),
        "micro_f1_all": float(2 * micro_p * micro_r / (micro_p + micro_r)) if micro_p + micro_r else 0.0,
        "per_label": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(truth[:, i].sum())} for i, name in enumerate(LABELS)},
    }

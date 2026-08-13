"""Metrics shared by every model arm.

``macro_f1`` is the checkpoint-selection metric; the positive-class scores are
reported alongside it because claim detection is usually read as "how well do we
find claims", not "how well do we label both classes".
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

POSITIVE_LABEL = 1


def scores(y_true, y_pred) -> Dict[str, float]:
    """Accuracy, macro-F1, and precision/recall/F1 for the claim class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POSITIVE_LABEL], average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "f1_pos": float(f1),
    }


def compute_metrics(eval_pred) -> Dict[str, float]:
    """``Trainer``-compatible wrapper around :func:`scores`."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return scores(labels, np.asarray(logits).argmax(axis=-1))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predictions_frame(texts, y_true, logits: np.ndarray) -> pd.DataFrame:
    """Per-example predictions -- the input the McNemar test will need."""
    logits = np.asarray(logits)
    probs = softmax(logits)
    return pd.DataFrame(
        {
            "text": list(texts),
            "label": np.asarray(y_true).astype(int),
            "pred": logits.argmax(axis=-1).astype(int),
            "prob_1": probs[:, POSITIVE_LABEL],
        }
    )

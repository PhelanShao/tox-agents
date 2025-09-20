import numpy as np
from sklearn.metrics import f1_score


def best_threshold(y_true: np.ndarray,
                   preds: np.ndarray,
                   metric: str = 'f1'):
    """Scan thresholds and return the best one according to ``metric``.

    Parameters
    ----------
    y_true: np.ndarray
        Binary ground truth labels.
    preds: np.ndarray
        Predicted probabilities.
    metric: str, default "f1"
        Currently only ``"f1"`` is supported.
    """
    ts = np.linspace(0.01, 0.99, 99)
    if metric == 'f1':
        scores = [f1_score(y_true, preds >= t) for t in ts]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    i = int(np.argmax(scores))
    return ts[i], scores[i]

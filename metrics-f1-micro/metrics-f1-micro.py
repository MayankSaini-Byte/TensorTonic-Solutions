import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    try:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        classes = np.unique(np.concatenate([y_true, y_pred]))

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))

            total_tp += tp
            total_fp += fp
            total_fn += fn

        N = 2 * total_tp
        D = (2 * total_tp) + total_fp + total_fn

        if D == 0:
            return 0.0

        return float(N / D)
        
    except Exception as e:
        return 0.0

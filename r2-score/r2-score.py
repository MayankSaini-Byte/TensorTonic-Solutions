import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    Y = np.array(y_true, dtype=float)
    Y_hat = np.array(y_pred, dtype=float)

    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    if ss_tot == 0:
        return 1.0 if np.allclose(Y, Y_hat) else 0.0

    return 1 - (ss_res / ss_tot)
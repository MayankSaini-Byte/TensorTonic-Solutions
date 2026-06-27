import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype = float)
    m = X.mean(axis = 0)

    Xc = X - m

    n = X.shape[0] - 1
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    sum = Xc.T@Xc / (n)

    return sum
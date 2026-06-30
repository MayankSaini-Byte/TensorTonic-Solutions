import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    X = np.array(x)
    Q = np.array(q)

    return np.percentile(X, Q, method='linear')
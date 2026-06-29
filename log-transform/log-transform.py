import numpy as np
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here

    X = np.asarray(values, dtype = float)

    X_lp = np.log1p(X)

    return X_lp
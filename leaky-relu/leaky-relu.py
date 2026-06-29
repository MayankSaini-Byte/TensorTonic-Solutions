import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    X= np.asarray(x, dtype = float)
    return np.where(X>=0, X, alpha*X)
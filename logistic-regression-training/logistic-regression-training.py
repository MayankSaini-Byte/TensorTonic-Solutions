import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    m,n = X.shape

    w = np.zeros(n)
    b = 0

    for _ in range(steps):
        z = X@w + b
        y_pred  = _sigmoid(z)
        dw = X.T @ (y_pred - y)/m
        db = np.mean(_sigmoid(z) - y)

        w -= lr*dw
        b -= lr*db

    return w,b
import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    n = len(y_pred)

    yp = np.array(y_pred)
    yt = np.array(y_true)
    
    MSE = (1/n) * (np.sum((yp - yt)**2))
    return MSE
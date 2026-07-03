import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    X= np.array(x)
    Y= np.array(y)

    M = np.abs(X-Y)

    return float(np.sum(M))
        
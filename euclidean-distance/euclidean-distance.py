import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    a = np.array(x)
    b = np.array(y)

    diff = a - b
    summ = sum(c*c for c in diff)
    return np.sqrt(summ)
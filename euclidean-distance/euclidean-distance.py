import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    X = np.array(x)
    Y = np.array(y)
    
    # Calculate squared differences
    squared_diff = (X - Y) ** 2
    
    # Sum the squares and take the square root
    return float(np.sqrt(np.sum(squared_diff)))

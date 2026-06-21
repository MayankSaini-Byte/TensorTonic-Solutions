import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    Y = np.array(y)
    _, u = np.unique(Y, return_counts = True)

    if len(u) == 0:
        return 0.0

    p = u/len(Y)
    E = -np.sum(p * np.log2(p))

    return E
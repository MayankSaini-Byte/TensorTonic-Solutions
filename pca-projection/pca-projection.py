import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here

    X = np.array(X, dtype = float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    n = X.shape[0] - 1 

    m = X.mean(axis = 0)

    Xc = X - m

    C = Xc.T@ Xc/(n)

    e_value, e_vectors = np.linalg.eig(C)

    idx = np.argsort(e_value)[::-1]
    e_vectors_sorted = e_vectors[:,idx]
    W = e_vectors_sorted[:, :k]

    Xp = Xc@W

    return Xp
import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:

        A = np.asarray(matrix)
    except (ValueError, TypeError):
        return None

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    evals = np.linalg.eigvals(A)

    sorted_indices = np.lexsort((evals.imag, evals.real))
    sorted_eigenvalues = evals[sorted_indices]

    return np.array(sorted_eigenvalues)

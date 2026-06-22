import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    X = np.array(x)
    P = np.array(p)
    
    # Check if probabilities sum to 1
    if np.allclose(np.sum(P), 1):
        # Correctly multiply elements element-wise and sum them up
        E = np.sum(X * P)
        return float(E)
    else:
        raise ValueError("The probabilities must sum to 1.")
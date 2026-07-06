import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    
    # 1. Validate structural requirements (Must be a 2D matrix where N >= 2)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
        
    # 2. Center the dataset by subtracting the feature means
    X_centered = X - np.mean(X, axis=0)
    
    # 3. Compute the covariance matrix (unscaled by N-1)
    # Using dot product avoids loops and processes features in parallel
    covariance_matrix = np.dot(X_centered.T, X_centered)
    
    # 4. Extract squared deviations (variances) from the diagonal
    variances = np.diag(covariance_matrix)
    
    # 5. Compute the standard deviation outer product matrix: σ_i * σ_j
    # sqrt(variance_i * variance_j) is mathematically identical
    std_outer_product = np.sqrt(np.outer(variances, variances))
    
    # 6. Normalize covariance by standard deviations, avoiding zero-division warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = covariance_matrix / std_outer_product
        
    return correlation_matrix
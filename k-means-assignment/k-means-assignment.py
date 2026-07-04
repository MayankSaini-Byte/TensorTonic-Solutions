import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """

    P = np.array(points)
    C = np.array(centroids)
    A = []
        
    for p in P:
        best_dist = float('inf')
        best_idx = -1
        for idx, c in enumerate(C):
            dist = np.sum((p - c)**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        A.append(best_idx)
    return(A)
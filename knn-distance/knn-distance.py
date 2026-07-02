import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train, dtype = float)
    X_test = np.asarray(X_test, dtype = float)

    if X_train.ndim == 1 :
        X_train = X_train.reshape(-1,1)
    if X_test.ndim == 1 :
        X_test = X_test.reshape(-1,1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]


    diff = X_test[:,np.newaxis,:] - X_train[np.newaxis,:,:]
    dist = np.sqrt(np.sum(diff**2, axis = -1))

    sorted_indices = np.argsort(dist, axis=1)

    if k > n_train:
        padd = np.full((n_test, k-n_train), -1, dtype = int)
        return np.hstack((sorted_indices, padd))
    else:
        return sorted_indices[:,:k]
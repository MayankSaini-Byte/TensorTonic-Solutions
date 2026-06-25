import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    l = np.maximum(0, margin - y_true*y_score)
    # l = [a*b for a in y_true and b in y_score]
    # loss = np.array(np.max(0,c for c in l))
    if(reduction == 'mean'):
        return l.mean()
    elif(reduction == 'sum'):
        return l.sum()
    
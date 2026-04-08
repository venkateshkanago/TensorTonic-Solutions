import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    y_true: 
        - either class indices (n,)  → e.g. [0,2,1]
        - or one-hot encoded (n, C)

    y_pred:
        - predicted probabilities (n, C)
    """

    # Convert to numpy
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true)

    # Numerical stability
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n_samples = y_pred.shape[0]

    # Case 1: y_true is class indices
    if y_true.ndim == 1:
        correct_probs = y_pred[np.arange(n_samples), y_true]
        loss = -np.log(correct_probs)

    # Case 2: one-hot encoded
    else:
        loss = -np.sum(y_true * np.log(y_pred), axis=1)

    # Return mean loss as Python float
    return float(np.mean(loss))
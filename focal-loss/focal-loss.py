import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.

    p: predicted probabilities (n,) or (n,1)
    y: true labels (0 or 1) (n,) or (n,1)
    gamma: focusing parameter
    """

    # Convert to numpy arrays
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Numerical stability
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)

    # Compute p_t
    p_t = y * p + (1 - y) * (1 - p)

    # Focal loss (no alpha since not in signature)
    loss = -((1 - p_t) ** gamma) * np.log(p_t)

    # Return scalar (mean)
    return float(np.mean(loss))
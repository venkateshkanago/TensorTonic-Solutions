import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.

    y_true: (n,) or (n,1)
    y_pred: (n,) or (n,1)
    delta: threshold parameter
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    # Error
    error = y_true - y_pred
    abs_error = np.abs(error)

    # Huber loss calculation
    quadratic = 0.5 * (error ** 2)
    linear = delta * (abs_error - 0.5 * delta)

    loss = np.where(abs_error <= delta, quadratic, linear)

    # Return mean as Python float
    return float(np.mean(loss))
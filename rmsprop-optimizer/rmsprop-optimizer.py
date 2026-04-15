import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    Works for scalars, lists, or NumPy arrays.
    """

    # Convert inputs to NumPy arrays (handles lists/sequences safely)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # RMSProp cache update
    s = beta * s + (1 - beta) * (g ** 2)

    # Parameter update
    w = w - (lr * g) / (np.sqrt(s) + eps)

    # Return scalar if scalar input, else array/list-compatible values
    if w.ndim == 0:
        return float(w), float(s)
    
    return w, s
import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    Return (w_new, G_new).
    Supports scalars, lists, tuples, and NumPy arrays.
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # Accumulate squared gradients
    G_new = G + g**2

    # Update parameters
    # NOTE: epsilon is added inside sqrt to match standard test cases
    w_new = w - (lr * g) / np.sqrt(G_new + eps)

    # Return scalar if scalar input
    if w_new.ndim == 0:
        return float(w_new), float(G_new)

    return w_new.tolist(), G_new.tolist()
import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    
    x: list or np.ndarray
    Returns: np.ndarray (same shape)
    """

    # Convert to numpy array
    x = np.asarray(x, dtype=float)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))

    # Swish
    return x * sigmoid
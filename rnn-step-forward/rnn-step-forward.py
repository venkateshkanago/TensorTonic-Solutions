import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Single RNN step forward (tanh cell).

    x_t: (D,)
    h_prev: (H,)
    Wx: (D, H)
    Wh: (H, H)
    b: (H,)

    Returns:
    h_t: (H,)
    """

    # Convert to numpy arrays
    x_t = np.asarray(x_t, dtype=float).reshape(-1)
    h_prev = np.asarray(h_prev, dtype=float).reshape(-1)
    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    # Compute next hidden state
    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)

    return h_t
import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.

    x: (N, C_in, H, W)
    W: (C_out, C_in, K_h, K_w)
    b: (C_out,)
    
    Returns:
    out: (N, C_out, H_out, W_out)
    """

    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, C_in, H, W_in = x.shape
    C_out, _, K_h, K_w = W.shape

    # Output dimensions (valid padding)
    H_out = H - K_h + 1
    W_out = W_in - K_w + 1

    # Initialize output
    out = np.zeros((N, C_out, H_out, W_out))

    # Convolution
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    region = x[n, :, i:i+K_h, j:j+K_w]
                    out[n, c_out, i, j] = np.sum(region * W[c_out]) + b[c_out]

    return out
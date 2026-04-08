import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).

    x: input array
    gamma: scale parameter
    beta: shift parameter
    eps: small constant for numerical stability
    """

    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # Case 1: Fully connected (N, D)
    if x.ndim == 2:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        out = gamma * x_hat + beta
        return out

    # Case 2: Convolutional (N, C, H, W)
    elif x.ndim == 4:
        # Compute mean/var across (N, H, W) for each channel
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # Reshape gamma and beta for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        out = gamma * x_hat + beta
        return out

    else:
        raise ValueError("Input must be 2D or 4D")
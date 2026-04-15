import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    Works for scalars, lists, or NumPy arrays.
    """

    # Convert inputs to NumPy arrays
    param = np.asarray(param, dtype=float)
    grad  = np.asarray(grad, dtype=float)
    m     = np.asarray(m, dtype=float)
    v     = np.asarray(v, dtype=float)

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Parameter update
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    # Return scalars if scalar input
    if param.ndim == 0:
        return float(param), float(m), float(v)

    return param, m, v
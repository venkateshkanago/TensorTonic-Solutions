import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    Must return NumPy arrays:
    (w_new, E_grad_sq_new, E_update_sq_new)
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)

    # Update running average of squared gradients
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * (grad ** 2)

    # Compute adaptive update
    delta = (np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_new + eps)) * grad

    # Update parameters
    w_new = w - delta

    # Update running average of squared parameter updates
    E_update_sq_new = rho * E_update_sq + (1 - rho) * (delta ** 2)

    return w_new, E_grad_sq_new, E_update_sq_new
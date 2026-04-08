import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.

    real_scores: critic outputs for real samples (n,)
    fake_scores: critic outputs for fake samples (n,)
    """

    # Convert to numpy arrays
    real_scores = np.asarray(real_scores, dtype=float).reshape(-1)
    fake_scores = np.asarray(fake_scores, dtype=float).reshape(-1)

    # WGAN critic loss: E[fake] - E[real]
    loss = np.mean(fake_scores) - np.mean(real_scores)

    # Return scalar
    return float(loss)
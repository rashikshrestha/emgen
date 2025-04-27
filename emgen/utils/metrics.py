import numpy as np
import torch
from scipy.stats import entropy


def compute_kl_divergence(p_actual, p_t):
    """
    Compute KL divergence between actual distribution and noise distribution at time t.

    Args:
        p_actual (torch.Tensor): Samples from the actual distribution
        p_t (torch.Tensor): Samples from the noisy distribution at time t

    Returns:
        float: KL divergence value
    """
    # Convert to numpy arrays for scipy entropy function
    p_actual_np = p_actual.detach().cpu().numpy()
    p_t_np = p_t.detach().cpu().numpy()

    # Estimate distributions using histograms
    bins = 100
    min_val = min(p_actual_np.min(), p_t_np.min())
    max_val = max(p_actual_np.max(), p_t_np.max())

    p_actual_hist, bin_edges = np.histogram(p_actual_np, bins=bins, range=(min_val, max_val), density=True)
    p_t_hist, _ = np.histogram(p_t_np, bins=bins, range=(min_val, max_val), density=True)

    # Add small constant to avoid division by zero
    p_actual_hist += 1e-10
    p_t_hist += 1e-10

    # Normalize
    p_actual_hist /= p_actual_hist.sum()
    p_t_hist /= p_t_hist.sum()

    # Compute KL divergence: KL(P_actual || P_t)
    kl_div = entropy(p_actual_hist, p_t_hist)

    return kl_div


def compute_Nd_kl_divergence(p_samples, q_samples, bins=100):
    """
    Estimate KL divergence between the samples of two multivariate distributions
    using histogram estimation.

    Args:
        p_samples (np.ndarray): Samples from P, shape (N, d)
        q_samples (np.ndarray): Samples from Q, shape (N, d)
        bins (int or list): Number of bins for each dimension

    Returns:
        float: KL divergence D(P || Q)
    """
    assert p_samples.shape[1] == q_samples.shape[1], "Dimension mismatch."

    d = p_samples.shape[1]

    # Determine histogram ranges
    min_val = np.minimum(p_samples.min(axis=0), q_samples.min(axis=0))
    max_val = np.maximum(p_samples.max(axis=0), q_samples.max(axis=0))
    ranges = [(min_val[i], max_val[i]) for i in range(d)]

    # Compute histograms
    p_hist, edges = np.histogramdd(p_samples, bins=bins, range=ranges, density=True)
    q_hist, _ = np.histogramdd(q_samples, bins=bins, range=ranges, density=True)

    # Flatten histograms
    p_hist = p_hist.flatten()
    q_hist = q_hist.flatten()

    # Add small constant to avoid division by zero
    p_hist += 1e-10
    q_hist += 1e-10

    # Normalize
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    # Compute KL divergence
    kl = entropy(p_hist, q_hist)

    return kl
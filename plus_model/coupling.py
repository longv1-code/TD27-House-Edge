"""Infer the shared lag-pattern distribution from observed training histories."""
import numpy as np
from scipy.special import softmax


def fit_joint(win_weights, amount_log_evidence, concentration=1.0, power=1.0):
    """EM for the joint lag prior, with symmetric Dirichlet pseudocounts.

    Inputs are likelihoods for each machine history, not target probabilities.
    Win weights contain a uniform pattern prior, which cancels in the E step.
    """
    win_weights = np.asarray(win_weights, dtype=float)
    evidence = np.asarray(amount_log_evidence, dtype=float)
    n, k = win_weights.shape
    if n == 0 or evidence.ndim != 2 or evidence.shape[0] != n:
        raise ValueError('Joint lag fitting requires aligned, nonempty histories')
    if concentration <= 0 or power <= 0:
        raise ValueError('Joint lag prior settings must be positive')
    l = evidence.shape[1]
    joint = np.full((k, l), 1.0 / (k * l))
    likelihood = np.log(np.maximum(win_weights, 1e-300))[:, :, None] + evidence[:, None, :]
    for _ in range(40):
        posterior = softmax((likelihood + np.log(joint)).reshape(n, -1), axis=1).reshape(n, k, l)
        joint = (posterior.sum(axis=0) + concentration) / (n + concentration * k * l)
    joint = joint ** power
    return joint / joint.sum()


def coupled_amount(win_weights, amount, amount_log_evidence, joint):
    """Use win-pattern evidence to resolve the sparser payout-size pattern."""
    joint = np.asarray(joint, dtype=float)
    likelihood = (np.log(np.maximum(win_weights, 1e-300))[:, :, None]
                  + amount_log_evidence[:, None, :] + np.log(joint)[None, :, :])
    posterior = softmax(likelihood.reshape(len(win_weights), -1), axis=1).reshape(likelihood.shape)
    amount_weights = posterior.sum(axis=1)
    state1 = (amount_weights * amount.last_details['state1']).sum(axis=1)
    return ((1 - state1[:, None]) * amount.templates[:, 0]
            + state1[:, None] * amount.templates[:, 1])

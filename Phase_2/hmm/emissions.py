"""HMM Emission Probability Module.

Computes log emission probabilities P(O_t | arc_i) from:
  1. Classifier predicted probability (primary, well-calibrated)
  2. Gaussian perpendicular-distance model (fallback / complement)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from typing import Dict, List


def gaussian_emission_log_prob(
    perp_dist: float,
    sigma_z: float = 20.0,
) -> float:
    """
    Gaussian emission: P(obs | arc) ∝ exp(-d² / (2 σ_z²))
    Returns the log probability (unnormalized).

    Args:
        perp_dist: Perpendicular distance from GPS point to candidate arc (meters).
        sigma_z:   Standard deviation of GPS measurement noise (meters). Typical: 10–30 m.
    """
    return -0.5 * (perp_dist / sigma_z) ** 2


def classifier_emission_log_probs(
    candidate_probs: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Derive emission log-probabilities from classifier predicted P(arc=match).

    Applies softmax normalization to prevent numerical collapse when all
    probabilities are low, then takes log.

    Args:
        candidate_probs: (K,) array of classifier P(match=1) for each candidate arc.

    Returns:
        log_emission: (K,) log-probabilities.
    """
    # Clip to avoid log(0)
    probs = np.clip(candidate_probs, eps, 1.0)

    # Normalize to a proper distribution (softmax-like re-normalization)
    probs = probs / (probs.sum() + eps)

    return np.log(probs + eps)


def combined_emission_log_probs(
    candidate_probs: np.ndarray,
    perp_dists: np.ndarray,
    sigma_z: float = 20.0,
    alpha: float = 0.7,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Combined emission: alpha * log P_classifier + (1-alpha) * log P_gaussian

    This blends the learned classifier signal with the geometric distance
    signal, which is useful when classifier confidence is low or for
    uncalibrated models.

    Args:
        candidate_probs: (K,) classifier probability for each candidate arc.
        perp_dists:      (K,) perpendicular distances (meters) for each candidate.
        sigma_z:         Gaussian noise standard deviation (meters).
        alpha:           Weight for classifier signal (0=pure distance, 1=pure classifier).

    Returns:
        log_emission: (K,) unnormalized log emission probabilities.
    """
    log_clf  = classifier_emission_log_probs(candidate_probs, eps)
    log_gauss = np.array([gaussian_emission_log_prob(d, sigma_z) for d in perp_dists])

    return alpha * log_clf + (1.0 - alpha) * log_gauss

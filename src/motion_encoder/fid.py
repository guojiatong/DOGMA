from __future__ import annotations

import numpy as np
from scipy import linalg


def fit_gaussian(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}")
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mean, cov


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariances have different shapes: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(diff @ diff + trace)

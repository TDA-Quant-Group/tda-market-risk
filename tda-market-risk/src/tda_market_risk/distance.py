"""Correlation to distance matrix conversion."""

from __future__ import annotations

import numpy as np


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to metric-like distance matrix.

    Formula: d_ij = sqrt(2 * (1 - rho_ij)).
    """
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Correlation matrix contains non-finite values")

    matrix = np.clip(matrix, -1.0, 1.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)

    dist = np.sqrt(np.clip(2.0 * (1.0 - matrix), 0.0, None))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def validate_distance_matrix(dist: np.ndarray, atol: float = 1e-8) -> bool:
    """Check basic structural properties of a distance matrix."""
    matrix = np.asarray(dist, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    if not np.allclose(matrix, matrix.T, atol=atol):
        return False
    if not np.allclose(np.diag(matrix), 0.0, atol=atol):
        return False
    if np.min(matrix) < -atol:
        return False
    if np.max(matrix) > 2.0 + atol:
        return False
    return True

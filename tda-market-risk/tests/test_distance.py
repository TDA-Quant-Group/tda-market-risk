"""Unit tests for distance conversion."""

from __future__ import annotations

import numpy as np
import pytest

from tda_market_risk.distance import correlation_to_distance, validate_distance_matrix


def test_correlation_to_distance_properties() -> None:
    corr = np.array(
        [
            [1.0, 0.5, -0.2],
            [0.5, 1.0, 0.1],
            [-0.2, 0.1, 1.0],
        ],
        dtype=float,
    )

    dist = correlation_to_distance(corr)

    assert dist.shape == corr.shape
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.min(dist) >= 0.0
    assert np.max(dist) <= 2.0
    assert validate_distance_matrix(dist)


def test_correlation_to_distance_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        correlation_to_distance(np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.1]]))

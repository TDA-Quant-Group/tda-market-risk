"""Unit tests for rolling correlation snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tda_market_risk.rolling import compute_rolling_snapshots


def test_rolling_known_correlation() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    base = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01], dtype=float)
    returns = pd.DataFrame(
        {
            "AAA": base,
            "BBB": 2.0 * base,
            "CCC": -1.0 * base,
        },
        index=dates,
    )

    snapshots = compute_rolling_snapshots(
        returns=returns,
        rolling_window_days=6,
        step_days=1,
        min_overlap_frac=1.0,
    )

    assert len(snapshots) == 1

    corr = snapshots[0].corr
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)

    ix_aaa = snapshots[0].tickers.index("AAA")
    ix_bbb = snapshots[0].tickers.index("BBB")
    ix_ccc = snapshots[0].tickers.index("CCC")

    assert np.isclose(corr[ix_aaa, ix_bbb], 1.0)
    assert np.isclose(corr[ix_aaa, ix_ccc], -1.0)

"""Cleaning and return computations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_dir, save_dataframe

LOGGER = logging.getLogger(__name__)


def preprocess_prices(
    prices: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Clean prices and compute log returns.

    Steps:
    1. Drop tickers that fail global non-NaN threshold.
    2. Align remaining tickers on intersection of dates (no forward-fill).
    3. Compute log returns and drop first row.
    """
    if prices.empty:
        raise ValueError("Input prices DataFrame is empty")

    ordered_existing = [t for t in config.tickers if t in prices.columns]
    if not ordered_existing:
        raise ValueError("None of configured tickers exist in downloaded prices")

    subset = prices.loc[:, ordered_existing].copy()

    non_nan_frac = subset.notna().mean(axis=0)
    keep_tickers = non_nan_frac[non_nan_frac >= config.min_non_nan_frac].index.tolist()
    dropped = [ticker for ticker in ordered_existing if ticker not in keep_tickers]
    if dropped:
        LOGGER.warning(
            "Dropping %d tickers below min_non_nan_frac=%.3f: %s",
            len(dropped),
            config.min_non_nan_frac,
            dropped,
        )

    if len(keep_tickers) < 2:
        raise ValueError("Need at least two tickers after missingness filter")

    cleaned = subset.loc[:, keep_tickers]
    aligned = cleaned.dropna(axis=0, how="any")
    if aligned.shape[0] < 2:
        raise ValueError("Not enough aligned rows after date intersection")

    returns = np.log(aligned / aligned.shift(1)).dropna(axis=0, how="any")
    if returns.empty:
        raise ValueError("Log return matrix is empty after preprocessing")

    return aligned, returns, keep_tickers


def save_processed_data(
    aligned_prices: pd.DataFrame,
    returns: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> None:
    """Persist processed aligned prices and returns."""
    processed_path = Path(processed_dir)
    ensure_dir(processed_path)

    aligned_parquet = processed_path / "aligned_prices.parquet"
    returns_parquet = processed_path / "returns.parquet"
    returns_csv = processed_path / "returns.csv"

    save_dataframe(aligned_prices, aligned_parquet)
    save_dataframe(returns, returns_parquet)
    save_dataframe(returns, returns_csv)

    LOGGER.info("Saved aligned prices to %s", aligned_parquet)
    LOGGER.info("Saved returns to %s and %s", returns_parquet, returns_csv)

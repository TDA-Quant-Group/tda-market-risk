"""Rolling correlation snapshot generation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .distance import correlation_to_distance
from .io import ensure_dir, save_npy

LOGGER = logging.getLogger(__name__)


@dataclass
class RollingSnapshot:
    """Single rolling correlation snapshot."""

    date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    tickers: list[str]
    corr: np.ndarray


def _window_correlation(
    window_returns: pd.DataFrame,
    min_overlap_frac: float,
) -> tuple[np.ndarray, list[str]] | None:
    """Compute a cleaned correlation matrix for one window."""
    if window_returns.empty:
        return None

    overlap = window_returns.notna().mean(axis=0)
    keep = overlap[overlap >= min_overlap_frac].index.tolist()
    if len(keep) < 2:
        return None

    candidate = window_returns.loc[:, keep]
    min_periods = max(2, int(np.ceil(len(candidate) * min_overlap_frac)))

    while candidate.shape[1] >= 2:
        corr = candidate.corr(min_periods=min_periods)
        corr = corr.loc[candidate.columns, candidate.columns]

        bad = corr.columns[corr.isna().any(axis=0)].tolist()
        if not bad:
            corr_values = corr.to_numpy(dtype=float)
            corr_values = 0.5 * (corr_values + corr_values.T)
            np.fill_diagonal(corr_values, 1.0)
            return corr_values, list(candidate.columns)

        candidate = candidate.drop(columns=bad)

    return None


def compute_rolling_snapshots(
    returns: pd.DataFrame,
    rolling_window_days: int,
    step_days: int,
    min_overlap_frac: float,
) -> list[RollingSnapshot]:
    """Compute rolling correlation snapshots from returns."""
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    if rolling_window_days < 2:
        raise ValueError("rolling_window_days must be >= 2")
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    if len(returns) < rolling_window_days:
        raise ValueError(
            f"Need at least {rolling_window_days} return rows, got {len(returns)}"
        )

    ordered_returns = returns.sort_index()
    snapshots: list[RollingSnapshot] = []

    for end_idx in range(rolling_window_days - 1, len(ordered_returns), step_days):
        window = ordered_returns.iloc[end_idx - rolling_window_days + 1 : end_idx + 1]
        result = _window_correlation(window_returns=window, min_overlap_frac=min_overlap_frac)
        if result is None:
            LOGGER.warning(
                "Skipping snapshot %s due to insufficient overlap",
                ordered_returns.index[end_idx].date().isoformat(),
            )
            continue

        corr, tickers = result
        snapshot = RollingSnapshot(
            date=pd.Timestamp(ordered_returns.index[end_idx]),
            window_start=pd.Timestamp(window.index[0]),
            window_end=pd.Timestamp(window.index[-1]),
            tickers=tickers,
            corr=corr,
        )
        snapshots.append(snapshot)

    if not snapshots:
        raise ValueError("No valid rolling snapshots were produced")

    return snapshots


def save_snapshots_and_manifest(
    snapshots: list[RollingSnapshot],
    corr_dir: str | Path = "outputs/correlation_matrices",
    dist_dir: str | Path = "outputs/distance_matrices",
    manifest_path: str | Path = "outputs/manifests/matrix_index.csv",
) -> pd.DataFrame:
    """Save correlation and distance matrices and build manifest CSV."""
    corr_dir_path = Path(corr_dir)
    dist_dir_path = Path(dist_dir)
    manifest_path_obj = Path(manifest_path)

    ensure_dir(corr_dir_path)
    ensure_dir(dist_dir_path)
    ensure_dir(manifest_path_obj.parent)

    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        date_str = snapshot.date.strftime("%Y%m%d")
        corr_path = corr_dir_path / f"corr_{date_str}.npy"
        dist_path = dist_dir_path / f"dist_{date_str}.npy"

        dist = correlation_to_distance(snapshot.corr)

        save_npy(snapshot.corr, corr_path)
        save_npy(dist, dist_path)

        rows.append(
            {
                "date": snapshot.date.date().isoformat(),
                "corr_path": corr_path.as_posix(),
                "dist_path": dist_path.as_posix(),
                "tickers_json": json.dumps(snapshot.tickers),
                "n_tickers": len(snapshot.tickers),
                "window_start": snapshot.window_start.date().isoformat(),
                "window_end": snapshot.window_end.date().isoformat(),
            }
        )

    manifest = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    manifest.to_csv(manifest_path_obj, index=False)
    LOGGER.info("Saved %d snapshots and manifest to %s", len(manifest), manifest_path_obj)
    return manifest

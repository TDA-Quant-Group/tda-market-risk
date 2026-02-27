"""Sanity checks and summary plotting for generated snapshots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import ensure_dir

LOGGER = logging.getLogger(__name__)


def _avg_offdiag(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(matrix[mask]))


def run_sanity_checks(
    manifest_path: str | Path = "outputs/manifests/matrix_index.csv",
    figure_path: str | Path = "outputs/figures/avg_corr.png",
) -> dict[str, float | int]:
    """Compute and plot simple sanity statistics from saved artifacts."""
    manifest_path_obj = Path(manifest_path)
    figure_path_obj = Path(figure_path)

    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

    manifest = pd.read_csv(manifest_path_obj, parse_dates=["date", "window_start", "window_end"])
    if manifest.empty:
        raise ValueError("Manifest is empty")

    avg_corr_series: list[float] = []
    min_dist_values: list[float] = []
    max_dist_values: list[float] = []

    for _, row in manifest.iterrows():
        corr = np.load(Path(row["corr_path"]))
        dist = np.load(Path(row["dist_path"]))

        avg_corr_series.append(_avg_offdiag(corr))
        min_dist_values.append(float(np.min(dist)))
        max_dist_values.append(float(np.max(dist)))

    ensure_dir(figure_path_obj.parent)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(manifest["date"], avg_corr_series, color="#1f77b4", linewidth=1.5)
    ax.set_title("Average Off-Diagonal Correlation Over Time")
    ax.set_xlabel("Snapshot Date")
    ax.set_ylabel("Average Correlation")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path_obj, dpi=150)
    plt.close(fig)

    stats: dict[str, float | int] = {
        "n_snapshots": int(len(manifest)),
        "min_n_tickers": int(manifest["n_tickers"].min()),
        "max_n_tickers": int(manifest["n_tickers"].max()),
        "mean_avg_corr": float(np.nanmean(avg_corr_series)),
        "std_avg_corr": float(np.nanstd(avg_corr_series)),
        "min_avg_corr": float(np.nanmin(avg_corr_series)),
        "max_avg_corr": float(np.nanmax(avg_corr_series)),
        "global_min_dist": float(np.min(min_dist_values)),
        "global_max_dist": float(np.max(max_dist_values)),
    }

    LOGGER.info("Saved sanity figure to %s", figure_path_obj)
    return stats

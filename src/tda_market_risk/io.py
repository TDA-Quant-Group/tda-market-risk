"""I/O helpers for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory path if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as CSV or Parquet based on file extension."""
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=True)
    elif suffix == ".parquet":
        df.to_parquet(path, index=True)
    else:
        raise ValueError(f"Unsupported dataframe output extension: {suffix}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV or Parquet based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe input extension: {suffix}")


def save_npy(array: np.ndarray, path: Path) -> None:
    """Save numpy array to .npy file."""
    ensure_dir(path.parent)
    np.save(path, array)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save dictionary to JSON file with stable key order."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    """Load dictionary from JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON file does not contain an object: {path}")
    return obj

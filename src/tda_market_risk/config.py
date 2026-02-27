"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the TDA market risk pipeline."""

    tickers: list[str]
    start_date: str
    end_date: str
    price_field: str
    rolling_window_days: int
    step_days: int
    min_non_nan_frac: float
    min_overlap_frac: float
    output_format: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "price_field": self.price_field,
            "rolling_window_days": self.rolling_window_days,
            "step_days": self.step_days,
            "min_non_nan_frac": self.min_non_nan_frac,
            "min_overlap_frac": self.min_overlap_frac,
            "output_format": self.output_format,
        }

    def fingerprint(self) -> str:
        """Return a stable hash for cache invalidation."""
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _parse_iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD, got: {value}") from exc


def _dedupe_tickers(raw_tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in raw_tickers:
        ticker_norm = ticker.strip().upper()
        if not ticker_norm:
            continue
        if ticker_norm not in seen:
            normalized.append(ticker_norm)
            seen.add(ticker_norm)
    return normalized


def validate_config(data: dict[str, Any]) -> PipelineConfig:
    """Validate config dictionary and return a typed config object."""
    required_fields = {
        "tickers",
        "start_date",
        "end_date",
        "price_field",
        "rolling_window_days",
        "step_days",
        "min_non_nan_frac",
        "min_overlap_frac",
        "output_format",
    }

    missing = required_fields.difference(data.keys())
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    tickers_obj = data["tickers"]
    if not isinstance(tickers_obj, list) or not tickers_obj:
        raise ValueError("tickers must be a non-empty list[str]")
    tickers = _dedupe_tickers([str(x) for x in tickers_obj])
    if not tickers:
        raise ValueError("tickers cannot be empty after normalization")

    start_date = str(data["start_date"])
    end_date = str(data["end_date"])
    start = _parse_iso_date(start_date, "start_date")
    end = _parse_iso_date(end_date, "end_date")
    if end <= start:
        raise ValueError("end_date must be strictly after start_date")

    price_field = str(data["price_field"])
    if not price_field:
        raise ValueError("price_field cannot be empty")

    rolling_window_days = int(data["rolling_window_days"])
    step_days = int(data["step_days"])
    if rolling_window_days < 2:
        raise ValueError("rolling_window_days must be >= 2")
    if step_days < 1:
        raise ValueError("step_days must be >= 1")

    min_non_nan_frac = float(data["min_non_nan_frac"])
    min_overlap_frac = float(data["min_overlap_frac"])
    if not 0.0 < min_non_nan_frac <= 1.0:
        raise ValueError("min_non_nan_frac must be in (0, 1]")
    if not 0.0 < min_overlap_frac <= 1.0:
        raise ValueError("min_overlap_frac must be in (0, 1]")

    output_format = str(data["output_format"]).lower()
    if output_format != "npy":
        raise ValueError("output_format must be 'npy'")

    return PipelineConfig(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        price_field=price_field,
        rolling_window_days=rolling_window_days,
        step_days=step_days,
        min_non_nan_frac=min_non_nan_frac,
        min_overlap_frac=min_overlap_frac,
        output_format=output_format,
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate pipeline config from YAML."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping at top level")

    return validate_config(raw)

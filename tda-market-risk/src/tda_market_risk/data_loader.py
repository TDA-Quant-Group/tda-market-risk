"""Raw price download and cache management."""

from __future__ import annotations

from datetime import date, timedelta
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PipelineConfig
from .io import ensure_dir, load_json, save_dataframe, save_json

LOGGER = logging.getLogger(__name__)


def _prepare_price_frame(raw: pd.DataFrame, tickers: list[str], price_field: str) -> pd.DataFrame:
    """Extract a ticker-columned price DataFrame from yfinance output."""
    if raw.empty:
        raise ValueError("No price data returned from yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = set(raw.columns.get_level_values(0))
        if price_field not in level_0:
            raise ValueError(f"Price field '{price_field}' not found in downloaded data")
        prices = raw[price_field].copy()
    else:
        if len(tickers) != 1:
            raise ValueError("Unexpected single-index yfinance output for multiple tickers")
        if price_field not in raw.columns:
            raise ValueError(f"Price field '{price_field}' not found in downloaded data")
        prices = raw[[price_field]].rename(columns={price_field: tickers[0]}).copy()

    prices.columns = [str(c).upper() for c in prices.columns]
    available = [ticker for ticker in tickers if ticker in prices.columns]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        LOGGER.warning("Tickers not returned by yfinance: %s", missing)
    if not available:
        raise ValueError("None of the requested tickers were returned by yfinance")

    prices = prices.loc[:, available]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    return prices


def fetch_prices(
    config: PipelineConfig,
    raw_dir: str | Path = "data/raw",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache adjusted close prices, reusing cache when config matches."""
    raw_dir_path = Path(raw_dir)
    ensure_dir(raw_dir_path)

    cache_path = raw_dir_path / "prices.parquet"
    meta_path = raw_dir_path / "prices_meta.json"

    fingerprint = config.fingerprint()

    if not force_refresh and cache_path.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("config_fingerprint") == fingerprint:
            LOGGER.info("Using cached raw prices at %s", cache_path)
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()
            return cached

    start = date.fromisoformat(config.start_date)
    end_inclusive = date.fromisoformat(config.end_date)
    end_exclusive = end_inclusive + timedelta(days=1)

    LOGGER.info(
        "Downloading prices for %d tickers from %s to %s",
        len(config.tickers),
        config.start_date,
        config.end_date,
    )

    raw = yf.download(
        tickers=config.tickers,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    prices = _prepare_price_frame(raw=raw, tickers=config.tickers, price_field=config.price_field)
    save_dataframe(prices, cache_path)
    save_json(
        {
            "config_fingerprint": fingerprint,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "price_field": config.price_field,
            "tickers": config.tickers,
            "n_rows": int(len(prices)),
            "n_cols": int(prices.shape[1]),
        },
        meta_path,
    )
    LOGGER.info("Saved raw price cache to %s", cache_path)
    return prices

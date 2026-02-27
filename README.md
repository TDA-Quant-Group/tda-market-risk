# tda-market-risk

Topological Data Analysis (TDA) data pipeline for equity market structure. This repository builds **time-indexed rolling correlation matrices** and corresponding **distance matrices** ready for persistent homology workflows.

## Run locally

```bash
cd tda-market-risk
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install '.[dev]'

python -m tda_market_risk.cli fetch
python -m tda_market_risk.cli build
python -m tda_market_risk.cli sanity
# or all-in-one:
python -m tda_market_risk.cli all
```

## What it does

1. Downloads daily adjusted close prices from Yahoo Finance (`yfinance`) for tickers in `configs/config.yaml`.
2. Caches raw prices in `data/raw/prices.parquet` with config-hash metadata for reproducible reruns.
3. Cleans data safely:
   - drops tickers below global missingness threshold,
   - aligns by **intersection of dates**,
   - computes log returns without forward-fill.
4. Computes rolling correlations with no look-ahead bias:
   - each snapshot at date `t` uses only the last `L` return rows up to and including `t`,
   - step size is `step_days` trading rows,
   - per-window ticker filtering via `min_overlap_frac`.
5. Converts correlations to distances:
   - `d_ij,t = sqrt(2 * (1 - rho_ij,t))`.
6. Saves artifacts and manifest for downstream topology code.

## Project structure

```text
tda-market-risk/
  configs/config.yaml
  data/raw/
  data/processed/
  outputs/correlation_matrices/
  outputs/distance_matrices/
  outputs/manifests/matrix_index.csv
  outputs/figures/avg_corr.png
  src/tda_market_risk/
  tests/
```

## Default config

`configs/config.yaml` includes a small liquid US universe (~20 tickers), start/end dates, rolling window settings, and missing-data thresholds.

## Outputs

- Correlation matrices: `outputs/correlation_matrices/corr_YYYYMMDD.npy`
- Distance matrices: `outputs/distance_matrices/dist_YYYYMMDD.npy`
- Manifest: `outputs/manifests/matrix_index.csv`

Manifest columns:
- `date`
- `corr_path`
- `dist_path`
- `tickers_json` (snapshot-specific ticker ordering)
- `n_tickers`
- `window_start`
- `window_end`

## For Topology Lead

Use `matrix_index.csv` to iterate snapshots and load distances:

```python
import json
import numpy as np
import pandas as pd

manifest = pd.read_csv("outputs/manifests/matrix_index.csv")
for _, row in manifest.iterrows():
    dist = np.load(row["dist_path"])
    tickers = json.loads(row["tickers_json"])
    # pass dist + tickers + row["date"] to persistent homology pipeline
```

## Tests

```bash
pytest
```

"""Command-line interface for the TDA market risk pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .data_loader import fetch_prices
from .preprocessing import preprocess_prices, save_processed_data
from .rolling import compute_rolling_snapshots, save_snapshots_and_manifest
from .sanity import run_sanity_checks

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_fetch(config_path: Path) -> None:
    """Run only the raw data download/cache step."""
    config = load_config(config_path)
    prices = fetch_prices(config=config)
    LOGGER.info("Fetched price matrix shape: %s", prices.shape)


def run_build(config_path: Path) -> None:
    """Run data download, preprocessing, rolling correlations, and matrix persistence."""
    config = load_config(config_path)

    prices = fetch_prices(config=config)
    aligned_prices, returns, _ = preprocess_prices(prices=prices, config=config)
    save_processed_data(aligned_prices=aligned_prices, returns=returns)

    snapshots = compute_rolling_snapshots(
        returns=returns,
        rolling_window_days=config.rolling_window_days,
        step_days=config.step_days,
        min_overlap_frac=config.min_overlap_frac,
    )

    manifest = save_snapshots_and_manifest(snapshots=snapshots)
    LOGGER.info("Build complete. Snapshots: %d", len(manifest))


def run_sanity() -> None:
    """Run sanity checks and save summary figure."""
    stats = run_sanity_checks()
    print("Sanity summary")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="TDA Market Risk pipeline CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch", help="Download and cache raw prices")
    subparsers.add_parser("build", help="Build rolling correlation and distance matrices")
    subparsers.add_parser("sanity", help="Run sanity checks on generated outputs")
    subparsers.add_parser("all", help="Run build and sanity checks")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if args.command == "fetch":
        run_fetch(config_path=args.config)
    elif args.command == "build":
        run_build(config_path=args.config)
    elif args.command == "sanity":
        run_sanity()
    elif args.command == "all":
        run_build(config_path=args.config)
        run_sanity()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

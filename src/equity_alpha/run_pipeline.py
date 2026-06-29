"""Command-line entry point for the cross-sectional equity research pipeline."""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CostConfig, run_backtest
from .data import download_yfinance_panel, generate_synthetic_panel, load_universe
from .features import build_features, feature_frame
from .metrics import (
    add_equity_columns,
    comparison_row,
    performance_summary,
    rank_ic_by_date,
    rank_ic_summary,
    regime_summary,
    slippage_sensitivity,
    split_summary,
)
from .portfolio import baseline_config, improved_config
from .reporting import ensure_output_dir, save_figures, write_csv, write_json
from .signals import BASELINE_SIGNAL, COMPOSITE_SIGNAL, add_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible cross-sectional equity alpha research backtest."
    )
    parser.add_argument("--data-source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--universe-file", default="data/universe/example_universe_300.csv")
    parser.add_argument("--max-tickers", type=int, default=300)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--synthetic-days", type=int, default=756)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--half-spread-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--save-feature-frame", action="store_true")
    parser.add_argument("--cache-dir", default=".cache/yfinance")
    parser.add_argument("--download-batch-size", type=int, default=50)
    parser.add_argument("--download-retries", type=int, default=3)
    return parser.parse_args()


def _clean_json_number(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _clean_mapping(mapping: dict[str, object]) -> dict[str, object]:
    return {key: _clean_json_number(value) for key, value in mapping.items()}


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    universe = load_universe(args.universe_file, max_tickers=args.max_tickers)

    if args.data_source == "synthetic":
        panel = generate_synthetic_panel(
            tickers=universe,
            days=args.synthetic_days,
            seed=args.seed,
            start=args.start,
        )
    else:
        panel = download_yfinance_panel(
            tickers=universe,
            start=args.start,
            end=args.end,
            min_history=args.min_history,
            cache_dir=args.cache_dir,
            batch_size=args.download_batch_size,
            retries=args.download_retries,
        )

    features = add_signals(build_features(panel))
    costs = CostConfig(
        commission_bps=args.commission_bps,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
    )

    baseline_daily, baseline_holdings = run_backtest(
        features, BASELINE_SIGNAL, baseline_config(), costs
    )
    improved_daily, improved_holdings = run_backtest(
        features, COMPOSITE_SIGNAL, improved_config(), costs
    )
    baseline_daily = add_equity_columns(baseline_daily)
    improved_daily = add_equity_columns(improved_daily)

    baseline_summary = performance_summary(baseline_daily)
    improved_summary = performance_summary(improved_daily)
    comparison = comparison_row(baseline_summary, improved_summary)

    baseline_ic = rank_ic_by_date(features, BASELINE_SIGNAL)
    improved_ic = rank_ic_by_date(features, COMPOSITE_SIGNAL)
    ic_summary = pd.DataFrame([rank_ic_summary(baseline_ic), rank_ic_summary(improved_ic)])

    baseline_regimes = regime_summary(baseline_daily, features)
    improved_regimes = regime_summary(improved_daily, features)
    baseline_splits = split_summary(baseline_daily)
    improved_splits = split_summary(improved_daily)
    baseline_slippage = slippage_sensitivity(baseline_daily, costs)
    improved_slippage = slippage_sensitivity(improved_daily, costs)

    write_csv(baseline_daily, output_dir, "baseline_daily_backtest.csv")
    write_csv(improved_daily, output_dir, "improved_daily_backtest.csv")
    write_csv(pd.DataFrame([baseline_summary]), output_dir, "baseline_summary.csv")
    write_csv(pd.DataFrame([improved_summary]), output_dir, "improved_summary.csv")
    write_csv(comparison, output_dir, "strategy_comparison.csv")
    write_csv(ic_summary, output_dir, "rank_ic_summary.csv")
    write_csv(baseline_ic, output_dir, "baseline_rank_ic_by_date.csv")
    write_csv(improved_ic, output_dir, "improved_rank_ic_by_date.csv")
    write_csv(baseline_regimes, output_dir, "baseline_regime_summary.csv")
    write_csv(improved_regimes, output_dir, "improved_regime_summary.csv")
    write_csv(baseline_splits, output_dir, "baseline_chronological_split_summary.csv")
    write_csv(improved_splits, output_dir, "improved_chronological_split_summary.csv")
    write_csv(baseline_slippage, output_dir, "baseline_slippage_sensitivity.csv")
    write_csv(improved_slippage, output_dir, "improved_slippage_sensitivity.csv")
    write_csv(baseline_holdings, output_dir, "baseline_holdings.csv")
    write_csv(improved_holdings, output_dir, "improved_holdings.csv")
    if args.save_feature_frame:
        write_csv(feature_frame(features), output_dir, "feature_frame.csv")
    save_figures(baseline_daily, improved_daily, output_dir)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": args.data_source,
        "requested_ticker_count": len(universe),
        "retained_ticker_count": int(features["ticker"].nunique()),
        "panel_start": str(features["date"].min().date()),
        "panel_end": str(features["date"].max().date()),
        "cost_assumptions_bps": {
            "commission": args.commission_bps,
            "half_spread": args.half_spread_bps,
            "slippage": args.slippage_bps,
        },
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    write_json(metadata, output_dir, "run_metadata.json")

    resume_metrics = {
        "warning": (
            "These are simulated historical research metrics. Verify them against the "
            "CSV reports and preserve the exact universe, dates, assumptions, and commit "
            "before using any number externally. Synthetic outputs are never resume evidence."
        ),
        "baseline": _clean_mapping(baseline_summary),
        "improved": _clean_mapping(improved_summary),
        "comparison": _clean_mapping(comparison.iloc[0].to_dict()),
    }
    write_json(resume_metrics, output_dir, "resume_metrics.json")

    print(f"Completed {args.data_source} run for {features['ticker'].nunique()} tickers.")
    print(pd.DataFrame([baseline_summary, improved_summary]).to_string(index=False))
    print(f"Outputs written to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()

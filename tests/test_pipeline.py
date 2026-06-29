from __future__ import annotations

from pathlib import Path

import numpy as np

from equity_alpha.backtest import CostConfig, run_backtest
from equity_alpha.data import generate_synthetic_panel
from equity_alpha.features import build_features
from equity_alpha.metrics import add_equity_columns, performance_summary, rank_ic_by_date, split_summary
from equity_alpha.portfolio import baseline_config, improved_config
from equity_alpha.signals import BASELINE_SIGNAL, COMPOSITE_SIGNAL, add_signals


def _features():
    tickers = [f"T{i:03d}" for i in range(40)]
    panel = generate_synthetic_panel(tickers=tickers, days=180, seed=17)
    return add_signals(build_features(panel))


def test_feature_timing_and_cross_sectional_columns():
    features = _features()
    assert features["next_ret_1d"].notna().sum() > 0
    assert features["feature_ready"].sum() > 0
    ready = features.loc[features["feature_ready"]]
    assert ready[BASELINE_SIGNAL].notna().all()
    assert ready[COMPOSITE_SIGNAL].notna().all()


def test_backtest_smoke_and_costs():
    features = _features()
    daily, holdings = run_backtest(
        features,
        COMPOSITE_SIGNAL,
        improved_config(),
        CostConfig(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=4.0),
    )
    daily = add_equity_columns(daily)
    assert not daily.empty
    assert not holdings.empty
    assert (daily["estimated_cost"] >= 0.0).all()
    assert np.allclose(daily["net_return"], daily["gross_return"] - daily["estimated_cost"])
    assert daily["one_way_turnover"].max() <= improved_config().max_daily_turnover + 1e-12


def test_summaries_and_rank_ic_exist():
    features = _features()
    daily, _ = run_backtest(features, BASELINE_SIGNAL, baseline_config(), CostConfig())
    summary = performance_summary(daily)
    assert summary["n_days"] == len(daily)
    assert "sharpe_zero_rf" in summary
    assert not rank_ic_by_date(features, BASELINE_SIGNAL).empty
    assert set(split_summary(daily)["evaluation_split"]) == {"in_sample", "validation", "out_of_sample"}

"""Performance, rank-IC, chronological split, regime, and slippage diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .backtest import CostConfig

TRADING_DAYS = 252


def add_equity_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative wealth and drawdown columns to one strategy's daily table."""
    df = daily.copy().sort_values("date").reset_index(drop=True)
    for prefix, return_column in [("net", "net_return"), ("gross", "gross_return")]:
        equity = (1.0 + df[return_column].fillna(0.0)).cumprod()
        df[f"{prefix}_equity"] = equity
        df[f"{prefix}_drawdown"] = equity / equity.cummax() - 1.0
    return df


def _annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    compound = float((1.0 + returns).prod())
    if compound <= 0.0:
        return float("nan")
    return compound ** (TRADING_DAYS / len(returns)) - 1.0


def _annualized_volatility(returns: pd.Series) -> float:
    volatility = float(returns.dropna().std(ddof=1))
    return volatility * math.sqrt(TRADING_DAYS) if np.isfinite(volatility) else float("nan")


def _zero_rate_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    standard_deviation = float(returns.std(ddof=1))
    if len(returns) < 2 or not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        return float("nan")
    return float(returns.mean() / standard_deviation * math.sqrt(TRADING_DAYS))


def performance_summary(daily: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize a daily backtest using a zero-risk-free-rate Sharpe convention."""
    df = add_equity_columns(daily)
    net_returns = df["net_return"]
    gross_returns = df["gross_return"]
    return {
        "strategy": str(df["strategy"].iloc[0]) if not df.empty and "strategy" in df else "unknown",
        "n_days": int(len(df)),
        "total_return": float(df["net_equity"].iloc[-1] - 1.0) if not df.empty else float("nan"),
        "annualized_return": _annualized_return(net_returns),
        "annualized_volatility": _annualized_volatility(net_returns),
        "sharpe_zero_rf": _zero_rate_sharpe(net_returns),
        "gross_sharpe_zero_rf": _zero_rate_sharpe(gross_returns),
        "maximum_drawdown": float(df["net_drawdown"].min()) if not df.empty else float("nan"),
        "average_daily_turnover": float(df["one_way_turnover"].mean()) if not df.empty else float("nan"),
        "annualized_turnover": float(df["one_way_turnover"].mean() * TRADING_DAYS) if not df.empty else float("nan"),
        "average_daily_cost_bps": float(df["estimated_cost"].mean() * 10_000.0) if not df.empty else float("nan"),
        "average_gross_exposure": float(df["gross_exposure"].mean()) if not df.empty else float("nan"),
        "average_abs_net_exposure": float(df["net_exposure"].abs().mean()) if not df.empty else float("nan"),
    }


def rank_ic_by_date(features: pd.DataFrame, signal_column: str, min_names: int = 20) -> pd.DataFrame:
    """Compute cross-sectional Spearman rank IC between a signal at t and return t+1."""
    rows: list[dict[str, object]] = []
    columns = ["date", signal_column, "next_ret_1d"]
    for date, day in features.loc[:, columns].groupby("date"):
        sample = day.dropna()
        if len(sample) < min_names:
            continue
        ic = sample[signal_column].rank().corr(sample["next_ret_1d"].rank())
        rows.append({"date": date, "signal": signal_column, "rank_ic": ic, "n_names": len(sample)})
    return pd.DataFrame(rows)


def rank_ic_summary(rank_ic: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize daily rank IC. The t-statistic assumes independent daily ICs."""
    if rank_ic.empty:
        return {"signal": "unknown", "n_days": 0, "mean_rank_ic": np.nan, "rank_ic_t_stat_iid": np.nan}
    values = rank_ic["rank_ic"].dropna()
    std = float(values.std(ddof=1))
    t_stat = float(values.mean() / std * math.sqrt(len(values))) if len(values) > 1 and std > 0.0 else np.nan
    return {
        "signal": str(rank_ic["signal"].iloc[0]),
        "n_days": int(len(values)),
        "mean_rank_ic": float(values.mean()),
        "median_rank_ic": float(values.median()),
        "rank_ic_std": std,
        "rank_ic_t_stat_iid": t_stat,
        "positive_ic_fraction": float((values > 0.0).mean()),
    }


def assign_chronological_split(daily: pd.DataFrame, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> pd.DataFrame:
    """Label predefined strategy results by chronological in/validation/OOS blocks."""
    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0 and train_fraction + validation_fraction < 1.0):
        raise ValueError("Split fractions must be positive and leave a non-empty out-of-sample segment.")
    df = daily.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    df["evaluation_split"] = "out_of_sample"
    df.loc[: train_end - 1, "evaluation_split"] = "in_sample"
    df.loc[train_end: validation_end - 1, "evaluation_split"] = "validation"
    return df


def split_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Return performance by chronological split for a fixed, pre-specified workflow."""
    labeled = assign_chronological_split(daily)
    rows = []
    for split, subset in labeled.groupby("evaluation_split", sort=False):
        row = performance_summary(subset)
        row["evaluation_split"] = split
        row["start_date"] = subset["date"].min()
        row["end_date"] = subset["date"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def _market_regime_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Create retrospective calm/volatile/stress labels for diagnostic reporting."""
    market = features.groupby("date")["ret_1d"].mean().fillna(0.0).sort_index().to_frame("market_return")
    market["market_rolling_vol"] = market["market_return"].rolling(21).std(ddof=1) * math.sqrt(TRADING_DAYS)
    market["market_equity"] = (1.0 + market["market_return"]).cumprod()
    market["market_drawdown"] = market["market_equity"] / market["market_equity"].cummax() - 1.0
    volatile_cutoff = market["market_rolling_vol"].dropna().quantile(0.67)
    market["regime"] = "calm"
    market.loc[market["market_rolling_vol"] >= volatile_cutoff, "regime"] = "volatile"
    market.loc[market["market_drawdown"] <= -0.10, "regime"] = "stress"
    return market.reset_index()


def regime_summary(daily: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance in retrospective volatility/drawdown regimes."""
    labels = _market_regime_labels(features)
    merged = daily.merge(labels[["date", "regime", "market_rolling_vol", "market_drawdown"]], on="date", how="left")
    rows = []
    for regime, subset in merged.groupby("regime", sort=False):
        row = performance_summary(subset)
        row["regime"] = regime
        row["mean_market_rolling_vol"] = float(subset["market_rolling_vol"].mean())
        row["mean_market_drawdown"] = float(subset["market_drawdown"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def slippage_sensitivity(
    daily: pd.DataFrame,
    costs: CostConfig,
    slippage_grid_bps: Iterable[float] = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0),
) -> pd.DataFrame:
    """Revalue executed trades under alternative slippage assumptions."""
    rows = []
    for slippage_bps in slippage_grid_bps:
        scenario = daily.copy()
        scenario["estimated_cost"] = (
            scenario["traded_notional"] * (costs.commission_bps + costs.half_spread_bps) / 10_000.0
            + scenario["liquidity_weighted_trade"] * float(slippage_bps) / 10_000.0
        )
        scenario["net_return"] = scenario["gross_return"] - scenario["estimated_cost"]
        row = performance_summary(scenario)
        row["slippage_bps"] = float(slippage_bps)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_row(
    baseline_summary: dict[str, float | int | str],
    improved_summary: dict[str, float | int | str],
) -> pd.DataFrame:
    """Create derived comparison statistics without asserting any target result."""
    baseline_turnover = float(baseline_summary["average_daily_turnover"])
    improved_turnover = float(improved_summary["average_daily_turnover"])
    turnover_reduction = np.nan
    if np.isfinite(baseline_turnover) and baseline_turnover != 0.0:
        turnover_reduction = 1.0 - improved_turnover / baseline_turnover
    return pd.DataFrame(
        [
            {
                "baseline_sharpe_zero_rf": baseline_summary["sharpe_zero_rf"],
                "improved_sharpe_zero_rf": improved_summary["sharpe_zero_rf"],
                "sharpe_change": float(improved_summary["sharpe_zero_rf"]) - float(baseline_summary["sharpe_zero_rf"]),
                "baseline_maximum_drawdown": baseline_summary["maximum_drawdown"],
                "improved_maximum_drawdown": improved_summary["maximum_drawdown"],
                "turnover_reduction_pct": turnover_reduction * 100.0 if np.isfinite(turnover_reduction) else np.nan,
            }
        ]
    )

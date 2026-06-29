"""Transaction-cost-aware long/short backtest runner."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio import PortfolioConfig, execute_portfolio


@dataclass(frozen=True)
class CostConfig:
    """Simple daily transaction-cost assumptions, all expressed in basis points."""

    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 4.0


def apply_costs(daily: pd.DataFrame, costs: CostConfig) -> pd.DataFrame:
    """Apply fixed and liquidity-scaled slippage costs to executed trade notional."""
    result = daily.copy()
    result["commission_cost"] = result["traded_notional"] * costs.commission_bps / 10_000.0
    result["spread_cost"] = result["traded_notional"] * costs.half_spread_bps / 10_000.0
    result["slippage_cost"] = result["liquidity_weighted_trade"] * costs.slippage_bps / 10_000.0
    result["estimated_cost"] = result[["commission_cost", "spread_cost", "slippage_cost"]].sum(axis=1)
    result["net_return"] = result["gross_return"] - result["estimated_cost"]
    return result


def run_backtest(
    features: pd.DataFrame,
    signal_column: str,
    portfolio_config: PortfolioConfig,
    cost_config: CostConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct, execute, and cost a portfolio under close(t)->close(t+1) timing."""
    daily, holdings = execute_portfolio(
        features=features,
        signal_column=signal_column,
        config=portfolio_config,
    )
    return apply_costs(daily, cost_config), holdings

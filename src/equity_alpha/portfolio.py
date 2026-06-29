"""Portfolio construction and execution-aware weight controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio and trading-control parameters for one strategy."""

    name: str
    selection_fraction: float
    liquidity_percentile: float | None = None
    inverse_volatility: bool = False
    trade_threshold: float = 0.0
    max_daily_turnover: float | None = None
    drawdown_trigger: float | None = None
    drawdown_gross_multiplier: float = 1.0
    gross_exposure: float = 1.0


def baseline_config() -> PortfolioConfig:
    return PortfolioConfig(
        name="baseline",
        selection_fraction=0.10,
        liquidity_percentile=None,
        inverse_volatility=False,
        trade_threshold=0.0,
        max_daily_turnover=None,
        drawdown_trigger=None,
        drawdown_gross_multiplier=1.0,
    )


def improved_config() -> PortfolioConfig:
    return PortfolioConfig(
        name="improved",
        selection_fraction=0.18,
        liquidity_percentile=0.30,
        inverse_volatility=True,
        trade_threshold=0.0015,
        max_daily_turnover=0.35,
        drawdown_trigger=-0.10,
        drawdown_gross_multiplier=0.60,
    )


def market_drawdown_series(features: pd.DataFrame) -> pd.Series:
    """Build a market proxy from average known same-day stock returns.

    `ret_1d` at date t is available at close(t), so this drawdown state is usable
    when forming the position for t -> t+1. It is not based on future returns.
    """
    market_return = features.groupby("date")["ret_1d"].mean().fillna(0.0).sort_index()
    wealth = (1.0 + market_return).cumprod()
    return wealth / wealth.cummax() - 1.0


def _target_weights_for_day(
    day: pd.DataFrame,
    signal_column: str,
    config: PortfolioConfig,
    drawdown: float,
) -> pd.Series:
    """Create desired weights before trade threshold and turnover constraints."""
    eligible = day.loc[
        day["feature_ready"] & day[signal_column].notna() & day["volatility_21"].notna(),
        ["ticker", signal_column, "volatility_21", "liquidity_rank"],
    ].copy()
    if config.liquidity_percentile is not None:
        eligible = eligible.loc[eligible["liquidity_rank"] >= config.liquidity_percentile]

    if len(eligible) < 10:
        return pd.Series(dtype=float)

    side_count = max(1, int(np.floor(len(eligible) * config.selection_fraction)))
    side_count = min(side_count, len(eligible) // 2)
    if side_count < 1:
        return pd.Series(dtype=float)

    long_book = eligible.nlargest(side_count, signal_column).copy()
    short_book = eligible.nsmallest(side_count, signal_column).copy()
    overlap = set(long_book["ticker"]).intersection(short_book["ticker"])
    if overlap:
        short_book = short_book.loc[~short_book["ticker"].isin(overlap)].copy()
    if short_book.empty:
        return pd.Series(dtype=float)

    gross_multiplier = config.gross_exposure
    if config.drawdown_trigger is not None and drawdown <= config.drawdown_trigger:
        gross_multiplier *= config.drawdown_gross_multiplier

    def side_weights(book: pd.DataFrame) -> np.ndarray:
        if config.inverse_volatility:
            raw = 1.0 / book["volatility_21"].clip(lower=1.0e-4).to_numpy()
        else:
            raw = np.ones(len(book))
        return raw / raw.sum()

    long_weights = side_weights(long_book) * (gross_multiplier / 2.0)
    short_weights = -side_weights(short_book) * (gross_multiplier / 2.0)
    target = pd.concat(
        [
            pd.Series(long_weights, index=long_book["ticker"].to_numpy()),
            pd.Series(short_weights, index=short_book["ticker"].to_numpy()),
        ]
    )
    return target.groupby(level=0).sum()


def execute_portfolio(
    features: pd.DataFrame,
    signal_column: str,
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply target weights sequentially with realistic simple execution controls.

    Trade threshold: suppress tiny desired changes.
    Turnover cap: proportionally scale the remaining trades when the one-way cap
    would be exceeded. Holdings are carried forward instead of re-normalized,
    which preserves the intended turnover reduction.
    """
    df = features.sort_values(["date", "ticker"]).copy()
    dates = pd.Index(sorted(df["date"].unique()))
    tickers = pd.Index(sorted(df["ticker"].unique()))
    drawdowns = market_drawdown_series(df)

    previous = pd.Series(0.0, index=tickers, dtype=float)
    daily_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    holdings_rows: list[pd.DataFrame] = []

    for date in dates:
        day = df.loc[df["date"] == date].copy().set_index("ticker", drop=False)
        target_sparse = _target_weights_for_day(
            day.reset_index(drop=True),
            signal_column=signal_column,
            config=config,
            drawdown=float(drawdowns.get(date, 0.0)),
        )
        target = target_sparse.reindex(tickers, fill_value=0.0)
        delta = target - previous

        if config.trade_threshold > 0.0:
            delta = delta.where(delta.abs() >= config.trade_threshold, 0.0)

        one_way_turnover = 0.5 * float(delta.abs().sum())
        if config.max_daily_turnover is not None and one_way_turnover > config.max_daily_turnover:
            delta = delta * (config.max_daily_turnover / one_way_turnover)
            one_way_turnover = config.max_daily_turnover

        current = previous + delta
        traded_notional = float(delta.abs().sum())
        next_return = day["next_ret_1d"].reindex(tickers).fillna(0.0)
        gross_return = float((current * next_return).sum())

        adv = day["adv20"].reindex(tickers)
        valid_adv = adv[adv > 0]
        median_adv = float(valid_adv.median()) if not valid_adv.empty else 1.0
        liquidity_multiplier = (median_adv / adv.replace(0.0, np.nan)).clip(0.5, 3.0).fillna(1.0)
        liquidity_weighted_trade = float((delta.abs() * liquidity_multiplier).sum())

        day_holdings = pd.DataFrame(
            {
                "date": date,
                "ticker": tickers,
                "weight": current.to_numpy(),
                "trade_weight": delta.to_numpy(),
                "signal": day[signal_column].reindex(tickers).to_numpy(),
                "liquidity_rank": day["liquidity_rank"].reindex(tickers).to_numpy(),
                "volatility_21": day["volatility_21"].reindex(tickers).to_numpy(),
            }
        )
        holdings_rows.append(day_holdings.loc[(day_holdings["weight"] != 0.0) | (day_holdings["trade_weight"] != 0.0)])

        daily_rows.append(
            {
                "date": date,
                "strategy": config.name,
                "gross_return": gross_return,
                "traded_notional": traded_notional,
                "liquidity_weighted_trade": liquidity_weighted_trade,
                "one_way_turnover": one_way_turnover,
                "gross_exposure": float(current.abs().sum()),
                "net_exposure": float(current.sum()),
                "long_exposure": float(current.clip(lower=0.0).sum()),
                "short_exposure": float(-current.clip(upper=0.0).sum()),
                "n_long": int((current > 0.0).sum()),
                "n_short": int((current < 0.0).sum()),
                "market_drawdown_at_trade": float(drawdowns.get(date, 0.0)),
            }
        )
        previous = current

    daily = pd.DataFrame(daily_rows)
    holdings = pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame()
    return daily, holdings

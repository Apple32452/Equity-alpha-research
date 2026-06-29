"""Feature construction using only information available at the close of date t."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "momentum_21",
    "momentum_63",
    "reversal_5",
    "volatility_21",
    "volatility_63",
    "adv20",
    "liquidity_rank",
    "rank_momentum_21",
    "rank_momentum_63",
    "rank_reversal_5",
    "rank_low_volatility_21",
]


def _winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize one cross-section while preserving missing observations."""
    valid = series.dropna()
    if len(valid) < 8:
        return series
    return series.clip(lower=valid.quantile(lower), upper=valid.quantile(upper))


def _cross_section_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    """Percentile rank within each date; rank 1.0 means highest feature value."""
    return frame.groupby("date")[column].transform(lambda series: series.rank(pct=True, method="average"))


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build momentum, reversal, volatility, and liquidity features.

    Timing convention:
    * features are calculated from adjusted closes through close(t);
    * `next_ret_1d` is close(t) to close(t+1), used only as the subsequent target;
    * no feature is computed from the next-day return.
    """
    df = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = grouped["adj_close"].transform(lambda series: series / series.shift(1) - 1.0)
    df["next_ret_1d"] = grouped["ret_1d"].shift(-1)
    df["dollar_volume"] = df["close"] * df["volume"]

    df["momentum_21"] = grouped["adj_close"].transform(lambda series: series / series.shift(21) - 1.0)
    df["momentum_63"] = grouped["adj_close"].transform(lambda series: series / series.shift(63) - 1.0)
    df["reversal_5"] = -grouped["adj_close"].transform(lambda series: series / series.shift(5) - 1.0)
    df["volatility_21"] = grouped["ret_1d"].transform(lambda series: series.rolling(21).std(ddof=1))
    df["volatility_63"] = grouped["ret_1d"].transform(lambda series: series.rolling(63).std(ddof=1))
    df["adv20"] = grouped["dollar_volume"].transform(lambda series: series.rolling(20).mean())

    for column in ["momentum_21", "momentum_63", "reversal_5", "volatility_21", "volatility_63", "adv20"]:
        df[column] = df.groupby("date")[column].transform(_winsorize_series)

    df["liquidity_rank"] = _cross_section_rank(df, "adv20")
    df["rank_momentum_21"] = _cross_section_rank(df, "momentum_21")
    df["rank_momentum_63"] = _cross_section_rank(df, "momentum_63")
    df["rank_reversal_5"] = _cross_section_rank(df, "reversal_5")
    df["rank_low_volatility_21"] = 1.0 - _cross_section_rank(df, "volatility_21")

    df["feature_ready"] = df[
        ["momentum_21", "momentum_63", "reversal_5", "volatility_21", "adv20"]
    ].notna().all(axis=1)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready feature table with the most useful fields first."""
    first = ["date", "ticker", "adj_close", "close", "volume", "next_ret_1d", "feature_ready"]
    available = [column for column in first + FEATURE_COLUMNS if column in features.columns]
    return features.loc[:, available].copy()

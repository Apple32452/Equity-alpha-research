"""Pre-specified cross-sectional signal definitions."""

from __future__ import annotations

import pandas as pd

BASELINE_SIGNAL = "baseline_signal"
COMPOSITE_SIGNAL = "composite_signal"


def add_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Add a momentum baseline and a diversified composite signal.

    The definitions are intentionally fixed before evaluation. The composite is a
    combined signal-and-portfolio-construction experiment, not a causal estimate
    of any one feature's contribution.
    """
    df = features.copy()
    df[BASELINE_SIGNAL] = 0.70 * df["rank_momentum_63"] + 0.30 * df["rank_momentum_21"]
    df[COMPOSITE_SIGNAL] = (
        0.66 * df["rank_momentum_63"]
        + 0.27 * df["rank_momentum_21"]
        + 0.05 * df["rank_reversal_5"]
        + 0.02 * df["rank_low_volatility_21"]
    )
    return df

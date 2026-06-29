"""Data access, normalization, and deterministic synthetic OHLCV generation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "ticker", "close", "adj_close", "volume"}


def load_universe(path: str | Path, max_tickers: int | None = None) -> list[str]:
    """Load a ticker universe from a CSV with a `ticker` column or first column."""
    universe = pd.read_csv(path)
    if universe.empty:
        raise ValueError(f"Universe file is empty: {path}")

    ticker_column = "ticker" if "ticker" in universe.columns else universe.columns[0]
    tickers = (
        universe[ticker_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)  # Yahoo convention: BRK-B, BF-B
    )
    tickers = [ticker for ticker in tickers if ticker and ticker != "NAN"]
    tickers = list(dict.fromkeys(tickers))
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    if not tickers:
        raise ValueError("No valid tickers were found in the universe file.")
    return tickers


def _standardize_columns(columns: Iterable[object]) -> dict[object, str]:
    """Map common vendor naming conventions to this project's standard names."""
    mapping: dict[object, str] = {}
    aliases = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "adjusted close": "adj_close",
        "volume": "volume",
    }
    for column in columns:
        normalized = str(column).strip().lower().replace("_", " ")
        mapping[column] = aliases.get(normalized, normalized.replace(" ", "_"))
    return mapping


def prepare_panel(panel: pd.DataFrame, min_history: int = 0) -> pd.DataFrame:
    """Validate, de-duplicate, and normalize a long OHLCV panel.

    The canonical format is one row per (`date`, `ticker`) with daily OHLCV fields.
    Adjusted close is used for returns; raw close times volume is used for dollar volume.
    """
    if panel.empty:
        raise ValueError("OHLCV panel is empty.")

    df = panel.copy()
    df = df.rename(columns=_standardize_columns(df.columns))
    missing_core = {"date", "ticker", "close", "volume"} - set(df.columns)
    if missing_core:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_core)}")
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["ticker"] = (
        df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close", "adj_close", "volume"])
    df = df[(df["close"] > 0) & (df["adj_close"] > 0) & (df["volume"] >= 0)]
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if min_history > 0:
        counts = df.groupby("ticker")["date"].transform("size")
        df = df.loc[counts >= min_history].copy()

    if df.empty:
        raise ValueError("No tickers remain after panel cleaning and history filters.")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, batch_size: int) -> pd.DataFrame:
    """Extract one ticker from a yfinance response without using DataFrame.stack.

    Avoiding stack is intentional: pandas 3 no longer permits `stack(dropna=False)`
    under the new stack implementation.
    """
    if raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy() if batch_size == 1 else pd.DataFrame()

    for level in range(raw.columns.nlevels):
        values = raw.columns.get_level_values(level).astype(str)
        if ticker in set(values):
            extracted = raw.xs(ticker, axis=1, level=level, drop_level=True)
            if isinstance(extracted, pd.Series):
                extracted = extracted.to_frame()
            return extracted.copy()
    return pd.DataFrame()


def _wide_yfinance_to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert yfinance wide output into the canonical long panel."""
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        part = _extract_ticker_frame(raw, ticker=ticker, batch_size=len(tickers))
        if part.empty:
            continue
        part = part.copy()
        part.index = pd.to_datetime(part.index, errors="coerce")
        if getattr(part.index, "tz", None) is not None:
            part.index = part.index.tz_localize(None)
        part = part.reset_index().rename(columns={part.index.name or "index": "date"})
        # yfinance may name the date column "Date" or "Datetime" after reset_index.
        if "date" not in part.columns:
            date_column = part.columns[0]
            part = part.rename(columns={date_column: "date"})
        part = part.rename(columns=_standardize_columns(part.columns))
        if "close" not in part.columns:
            continue
        if "adj_close" not in part.columns:
            part["adj_close"] = part["close"]
        if "volume" not in part.columns:
            continue
        part["ticker"] = ticker
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def download_yfinance_panel(
    tickers: list[str],
    start: str,
    end: str | None,
    min_history: int,
    cache_dir: str | Path = ".cache/yfinance",
    batch_size: int = 50,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance in resilient batches.

    This function keeps a cache inside the repository rather than relying on the
    default user cache. That directly avoids the macOS cache-path error seen in
    some local installations. Failed batches are retried and incomplete tickers
    are excluded by `min_history` rather than silently forward-filled.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "yfinance is required for --data-source yfinance. "
            "Run: pip install -r requirements.txt"
        ) from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_path))
    except Exception:
        # Downloading remains possible on yfinance versions that do not expose it.
        pass

    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for start_idx in range(0, len(tickers), batch_size):
        batch = tickers[start_idx : start_idx + batch_size]
        raw = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    group_by="column",
                    progress=False,
                    threads=True,
                )
                if not raw.empty:
                    break
            except Exception as exc:  # network/vendor failures are transient in practice
                last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds * attempt)

        if raw.empty:
            message = f"Unable to download batch beginning with {batch[0]}"
            if last_error is not None:
                message += f": {last_error}"
            print(f"Warning: {message}")
            failed.extend(batch)
            continue

        long_batch = _wide_yfinance_to_long(raw, batch)
        retrieved = set(long_batch.get("ticker", pd.Series(dtype=str)).astype(str))
        failed.extend([ticker for ticker in batch if ticker not in retrieved])
        if not long_batch.empty:
            all_frames.append(long_batch)

    if not all_frames:
        raise RuntimeError("yfinance returned no usable OHLCV data for the requested universe.")

    panel = prepare_panel(pd.concat(all_frames, ignore_index=True), min_history=min_history)
    retained = set(panel["ticker"])
    print(
        f"Downloaded {len(retained)}/{len(tickers)} tickers meeting min_history={min_history}. "
        f"Batch failures or missing data: {len(set(failed) - retained)}."
    )
    return panel


def generate_synthetic_panel(
    tickers: list[str],
    days: int = 756,
    seed: int = 7,
    start: str = "2023-01-03",
) -> pd.DataFrame:
    """Generate a deterministic synthetic daily OHLCV panel for software tests.

    The generator includes persistent asset-level states, heterogeneous volatility,
    liquidity, and a common market component. It exists only to verify the code
    path. It is not a source of investment evidence or resume metrics.
    """
    if len(tickers) < 10:
        raise ValueError("Synthetic mode needs at least 10 tickers for cross-sectional portfolios.")
    if days < 120:
        raise ValueError("Synthetic mode needs at least 120 days for the default features.")

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    dates = pd.bdate_range(start=start, periods=days)

    market = rng.normal(loc=0.00015, scale=0.0080, size=days)
    idio_vol = rng.uniform(0.008, 0.025, size=n_assets)
    liquidity_scale = np.exp(rng.normal(loc=np.log(4.0e8), scale=0.9, size=n_assets))
    prices = rng.uniform(25.0, 250.0, size=n_assets)
    states = rng.normal(0.0, 1.0, size=n_assets)

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for date_idx, date in enumerate(dates):
        state_noise = rng.normal(size=n_assets)
        states = 0.965 * states + np.sqrt(1.0 - 0.965**2) * state_noise
        idio = rng.normal(scale=idio_vol, size=n_assets)
        # A small persistent component creates a usable synthetic test bed while
        # leaving substantial noise. It does not represent a real alpha claim.
        returns = market[date_idx] + 0.00055 * states + idio
        open_prices = prices * np.exp(rng.normal(0.0, 0.002, size=n_assets))
        close_prices = np.maximum(open_prices * (1.0 + returns), 1.0)
        high_prices = np.maximum(open_prices, close_prices) * (1.0 + rng.uniform(0.0, 0.012, n_assets))
        low_prices = np.minimum(open_prices, close_prices) * (1.0 - rng.uniform(0.0, 0.012, n_assets))
        dollar_volume = liquidity_scale * np.exp(rng.normal(0.0, 0.45, size=n_assets))
        volumes = np.maximum((dollar_volume / close_prices).round(), 1.0)

        for idx, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": float(open_prices[idx]),
                    "high": float(high_prices[idx]),
                    "low": float(low_prices[idx]),
                    "close": float(close_prices[idx]),
                    "adj_close": float(close_prices[idx]),
                    "volume": float(volumes[idx]),
                }
            )
        prices = close_prices

    return prepare_panel(pd.DataFrame(rows), min_history=days)

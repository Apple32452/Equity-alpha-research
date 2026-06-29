# Cross-Sectional Equity Alpha Research Pipeline

A reproducible daily equity-research pipeline for evaluating cross-sectional momentum, reversal, volatility, and liquidity features under next-day-return, transaction-cost-aware, liquidity-aware, chronological split, and regime-diagnostic settings.

> **Research prototype, not a live trading system.** Results are simulated historical backtests. They are not live PnL, investment advice, a production execution system, or an institutional-grade market-impact model.

## What it implements

- A configurable U.S.-equity universe (the included `example_universe_300.csv` contains 300+ liquid U.S. ticker symbols).
- Two data modes:
  - `synthetic`: deterministic OHLCV data for software validation only.
  - `yfinance`: public daily OHLCV download for a supplied universe.
- Data cleaning: canonical long panel, duplicate removal, price/volume validation, ticker-history filters, and cross-sectional winsorization.
- Features formed using data through the close of day `t`:
  - 21-day and 63-day momentum
  - 5-day short-term reversal
  - 21-day and 63-day realized volatility
  - 20-day average dollar volume and cross-sectional liquidity rank
- A fixed momentum baseline and a fixed composite signal.
- Long/short portfolio construction with a liquidity screen, inverse-volatility sizing, trade/no-trade threshold, daily turnover cap, and drawdown-triggered gross-exposure reduction.
- Close(`t`) to close(`t+1`) portfolio evaluation, preventing same-day signal/return look-ahead.
- Simulated transaction costs: commissions, half-spread, and liquidity-scaled slippage.
- Diagnostics: zero-risk-free-rate Sharpe, drawdown, turnover, rank IC, chronological in/validation/OOS partitions, calm/volatile/stress summaries, and slippage sensitivity.

## Important interpretation limits

1. The baseline-versus-improved comparison changes both the signal and portfolio-construction choices. It is **not** an attribution study proving that one individual feature caused the performance difference.
2. The rank-IC t-statistic uses an IID approximation. It is not HAC/Newey-West adjusted.
3. Regime labels are **retrospective diagnostics** based on full-sample market volatility quantiles and drawdown states. They are not a real-time regime model.
4. The cost model is intended for sensitivity analysis. It excludes borrow fees, participation limits, calibrated square-root market impact, exchange fees/rebates, intraday order-book depth, and production execution logic.
5. Public yfinance data can change through corrections, ticker changes, survivorship issues, and vendor adjustments. Preserve raw data and all exported reports for any external claim.
6. Synthetic performance output is only a code-path test and must never be used on a resume, portfolio, interview, or investment discussion.

## Repository structure

```text
equity-alpha-research-pipeline/
├── data/universe/
│   ├── example_universe_50.csv
│   └── example_universe_300.csv
├── src/equity_alpha/
│   ├── backtest.py
│   ├── data.py
│   ├── features.py
│   ├── metrics.py
│   ├── portfolio.py
│   ├── reporting.py
│   ├── run_pipeline.py
│   └── signals.py
├── tests/test_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
cd equity-alpha-research-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Run the deterministic software-validation demo

```bash
python -m equity_alpha.run_pipeline \
  --data-source synthetic \
  --universe-file data/universe/example_universe_300.csv \
  --max-tickers 300 \
  --synthetic-days 756 \
  --seed 7 \
  --commission-bps 1.0 \
  --half-spread-bps 2.0 \
  --slippage-bps 4.0 \
  --output-dir outputs/demo \
  --save-feature-frame
```

## Run historical public-data research

```bash
python -m equity_alpha.run_pipeline \
  --data-source yfinance \
  --universe-file data/universe/example_universe_300.csv \
  --max-tickers 300 \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --min-history 500 \
  --commission-bps 1.0 \
  --half-spread-bps 2.0 \
  --slippage-bps 4.0 \
  --cache-dir .cache/yfinance \
  --output-dir outputs/real_run \
  --save-feature-frame
```

`data.py` uses a manual wide-to-long conversion rather than `DataFrame.stack(dropna=False)`, so it runs under pandas 2.x and pandas 3.x. It also assigns yfinance a project-local timezone cache directory, which avoids the common macOS default-cache-path failure.

## Main outputs

- `baseline_daily_backtest.csv`, `improved_daily_backtest.csv`
- `baseline_summary.csv`, `improved_summary.csv`, `strategy_comparison.csv`
- `rank_ic_summary.csv` and per-date rank-IC files
- chronological split and regime summary CSVs
- slippage-sensitivity CSVs
- holdings, feature frame (optional), three PNG figures, run metadata, and `resume_metrics.json`

The `resume_metrics.json` file is an auditable convenience export, not proof that a value should be used externally. Verify every value against the CSV reports, exact universe, date range, execution assumptions, source data, and repository commit.

## Resume-safe wording after a verified historical run

> Built a reproducible cross-sectional equity research pipeline over a documented U.S. equity universe, generating momentum, reversal, volatility, and liquidity features at close `t` and evaluating long/short portfolios on close-to-close next-day returns with simulated transaction costs, turnover controls, and regime diagnostics.

Only add numerical Sharpe, turnover, slippage, or drawdown claims after reproducing them from a preserved historical-data run and its exported reports.

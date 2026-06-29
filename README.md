# Cross-Sectional Equity Research & Backtesting Pipeline

A reproducible Python research pipeline for testing cross-sectional U.S. equity signals with daily historical data, long–short portfolio construction, execution-aware trading controls, and transaction-cost modeling.

> **Research prototype only.** This repository is designed for historical simulation and research diagnostics. It is not a live trading system, investment recommendation, or institutional-grade execution platform.

## Overview

The pipeline evaluates whether predefined cross-sectional predictors retain information about next-day returns after accounting for liquidity, volatility, turnover, drawdowns, and modeled trading costs.

It supports two modes:

* `synthetic`: deterministic OHLCV generation for unit tests and software validation.
* `yfinance`: public daily historical OHLCV data for empirical research.

Synthetic results are used only to validate that the research and backtesting workflow executes correctly. Historical-data results are the appropriate basis for interpreting the strategy diagnostics.

## Research workflow

The pipeline performs the following steps:

1. Downloads or generates daily OHLCV data.
2. Cleans the equity panel by removing duplicates, invalid prices, invalid volumes, and tickers with insufficient history.
3. Constructs cross-sectional momentum, reversal, volatility, and liquidity features using information available through the close of day (t).
4. Forms signals at close (t) and evaluates realized close-to-close returns from (t) to (t+1).
5. Builds market-neutral long–short portfolios.
6. Applies transaction costs, liquidity-aware slippage, turnover constraints, and risk controls.
7. Exports performance, rank-IC, chronological-split, regime, turnover, drawdown, holdings, and slippage-sensitivity diagnostics.

## Features

The feature set includes:

* 21-day momentum
* 63-day momentum
* 5-day short-term reversal
* 21-day realized volatility
* 63-day realized volatility
* 20-day average dollar volume
* Cross-sectional liquidity rank
* Cross-sectional percentile ranks for each predictor

All features are winsorized cross-sectionally before ranking. The target variable is the next trading day's return, which avoids using future returns in signal construction.

## Strategy comparison

The repository compares two fixed research configurations.

### Baseline

* Momentum-only signal using 21-day and 63-day momentum ranks
* Equal-weighted long–short construction
* Long the top 10% and short the bottom 10% of eligible names
* No liquidity screen, trade threshold, turnover cap, or drawdown exposure reduction

### Improved execution-aware configuration

* Composite signal combining momentum, reversal, and low-volatility ranks
* Liquidity filter retaining names above the 30th percentile of average dollar volume
* Inverse-volatility position sizing
* Trade/no-trade threshold to avoid very small rebalances
* Maximum one-way daily turnover cap of 35%
* Gross-exposure reduction when the market proxy drawdown reaches -10%

The baseline-versus-improved comparison is an end-to-end research comparison. It does not isolate the causal contribution of any single feature or execution control.

## Verified historical-data run

A historical `yfinance` run requested a 300-symbol U.S. equity universe. After data-availability and minimum-history filtering, 76 equities were retained across 771 trading days.

| Metric                                  | Baseline | Improved |
| --------------------------------------- | -------: | -------: |
| Total return                            |   -7.95% |   -6.15% |
| Annualized return                       |   -2.67% |   -2.05% |
| Annualized volatility                   |   15.34% |   11.92% |
| Net Sharpe ratio, zero risk-free rate   |   -0.100 |   -0.115 |
| Gross Sharpe ratio, zero risk-free rate |    0.327 |    0.326 |
| Maximum drawdown                        |  -33.32% |  -23.19% |
| Average daily turnover                  |   16.26% |   16.10% |
| Average modeled daily cost              | 2.60 bps | 2.08 bps |

In this run, the execution-aware configuration:

* Reduced annualized volatility by approximately 22%.
* Reduced maximum drawdown magnitude by 10.1 percentage points.
* Reduced modeled average daily trading costs by approximately 20%.
* Kept gross Sharpe nearly unchanged at 0.326 versus 0.327.
* Did not produce positive net simulated returns under the stated assumptions.

These results should be interpreted as risk-control and execution diagnostics rather than evidence of deployable alpha.

## Repository structure

```text
equity-alpha-research-pipeline/
├── data/
│   └── universe/
│       ├── example_universe_50.csv
│       └── example_universe_300.csv
├── src/
│   └── equity_alpha/
│       ├── backtest.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── portfolio.py
│       ├── reporting.py
│       ├── run_pipeline.py
│       └── signals.py
├── tests/
│   └── test_pipeline.py
├── outputs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone <YOUR-REPOSITORY-URL>
cd equity-alpha-research-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

pytest -q
```

## Run the synthetic software-validation demo

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

## Run historical-data research

```bash
python -m equity_alpha.run_pipeline \
  --data-source yfinance \
  --universe-file data/universe/example_universe_300.csv \
  --max-tickers 300 \
  --start 2023-06-01 \
  --min-history 252 \
  --commission-bps 1.0 \
  --half-spread-bps 2.0 \
  --slippage-bps 4.0 \
  --cache-dir .cache/yfinance \
  --output-dir outputs/real_run \
  --save-feature-frame
```

The final number of retained equities may be lower than the requested universe size because tickers can fail downloads, have insufficient history, be delisted, or have incomplete price and volume data.

## Main outputs

Each run writes reports to the chosen output directory:

```text
baseline_daily_backtest.csv
improved_daily_backtest.csv
baseline_summary.csv
improved_summary.csv
strategy_comparison.csv
rank_ic_summary.csv
baseline_rank_ic_by_date.csv
improved_rank_ic_by_date.csv
baseline_chronological_split_summary.csv
improved_chronological_split_summary.csv
baseline_regime_summary.csv
improved_regime_summary.csv
baseline_slippage_sensitivity.csv
improved_slippage_sensitivity.csv
baseline_holdings.csv
improved_holdings.csv
feature_frame.csv
equity_curve.png
drawdown.png
turnover.png
run_metadata.json
resume_metrics.json
```

## Diagnostics

The pipeline reports:

* Net and gross zero-risk-free-rate Sharpe ratios
* Annualized return and volatility
* Maximum drawdown
* Daily and annualized turnover
* Modeled trading costs
* Spearman rank information coefficient between each signal and next-day returns
* Chronological in-sample, validation, and out-of-sample summaries
* Retrospective calm, volatile, and stress regime summaries
* Slippage sensitivity from 0 to 12 basis points

## Reproducibility notes

* Signals are generated using data available through close (t).
* Portfolio returns use the next-day close-to-close return from (t) to (t+1).
* Transaction costs include commissions, half-spread, and liquidity-scaled slippage.
* The code uses a project-local `yfinance` cache directory to avoid common macOS cache-path issues.
* The downloader uses a manual wide-to-long conversion compatible with pandas 2.x and pandas 3.x.
* Public market data can change due to ticker changes, corrections, corporate actions, vendor adjustments, or missing history.

For any external use of metrics, preserve the output folder, repository commit, command-line parameters, data-download date, retained universe, and cost assumptions.

## Limitations

* Historical backtests do not guarantee future performance.
* The cost model is a simplified sensitivity model and does not include borrow fees, market impact calibration, order-book depth, participation limits, exchange rebates, or intraday execution.
* Regime labels are retrospective diagnostics, not a real-time regime classifier.
* The rank-IC t-statistic uses an IID approximation and is not Newey-West adjusted.
* The strategy comparison changes both signal construction and portfolio controls, so it should not be interpreted as a single-factor attribution analysis.
* Synthetic-data performance is not used as evidence of real-world predictive performance.

## Author

Taewoon Choi

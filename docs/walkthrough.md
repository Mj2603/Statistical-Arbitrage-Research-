# Research Walkthrough

This guide runs the repository's NSE banking-pairs research entry point. It uses daily prices retrieved by yfinance; results can vary with the data vendor, retrieval date, and corporate-action adjustments.

## 1. Set up the environment

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Run the walk-forward example

From the repository root:

```bash
python scripts/run_banking_pairs.py \
  --start 2019-01-01 \
  --leg-y HDFCBANK \
  --leg-x ICICIBANK
```

The script prints candidate cointegrated pairs, then reports out-of-sample Sharpe, Sortino, maximum drawdown, turnover, and hit rate for the selected pair. It also prints linear and XGBoost spread-prediction baseline scores.

## 3. Run the test suite

```bash
pytest -q
```

The tests include synthetic-data checks. A passing test suite verifies code behavior on those fixtures; it does not validate a profitable strategy.

## Reproducibility checklist

For a research comparison, record the price source, retrieval date, date range, selected pair, train/test window lengths, signal thresholds, cost assumptions, and package versions. Preserve the input data snapshot where licensing permits.

## Interpretation

The backtester uses a fixed OLS hedge ratio for each training window and a simplified cost model. The optional ML output is a baseline diagnostic; it is not used to generate the backtest's trades. Read the [README](../README.md) and [architecture notes](architecture.md) before interpreting performance.

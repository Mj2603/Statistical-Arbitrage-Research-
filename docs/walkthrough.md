# Research Walkthrough

This guide runs the repository's NSE banking-pairs research entry point. It uses daily prices retrieved by yfinance; results can vary with the data vendor, retrieval date, and corporate-action adjustments.

## 1. Set up the environment

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 2. Run the walk-forward example

From the repository root:

```bash
python scripts/run_banking_pairs.py \
  --start 2019-01-01 \
  --leg-y HDFCBANK \
  --leg-x ICICIBANK
```

The script needs network access to Yahoo Finance. It prints summary statistics for the pre-specified pair and for per-window pair selection, where the pair is chosen from training data only. It then writes `results/nse_banking_walkforward.md`, which has a cost-sensitivity table (0/5/10/20 bps), the pair selected in each window and the ML baseline scores. Use `--csv path.csv` to run on a saved price snapshot instead.

For a deterministic run that needs no network, use `python scripts/run_synthetic_experiment.py`, which writes `results/synthetic_walkforward.md`. That run uses simulated data.

## 3. Run the test suite

```bash
pytest -q
```

The tests include synthetic-data checks. A passing test suite verifies code behavior on those fixtures; it does not validate a profitable strategy.

## Reproducibility checklist

For a research comparison, record the price source, retrieval date, date range, selected pair, train/test window lengths, signal thresholds, cost assumptions, and package versions. Preserve the input data snapshot where licensing permits.

## Interpretation

The backtester uses a fixed OLS hedge ratio for each training window, a trailing z-score warmed up from the training window, flat 0% days for windows where no pair passes the test, and a simplified cost model. The optional ML output is a baseline diagnostic; it is not used to generate the backtest's trades. Read the [README](../README.md) and [architecture notes](architecture.md) before interpreting performance.

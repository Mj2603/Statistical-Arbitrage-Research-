# Statistical Arbitrage Research

A Python research prototype for cointegration-based pairs research on daily NSE equity data. It includes Engle–Granger and Johansen selection, rolling walk-forward evaluation, a z-score signal, a simple transaction-cost model, performance summaries, and optional linear/XGBoost spread prediction.

The implementation is intended for reproducible research and education. It is not a live trading system and does not establish profitability.

## Implemented

- Pair screening with Engle–Granger or Johansen methods.
- OLS hedge-ratio estimation using the training window in walk-forward runs.
- Threshold-based spread signals and periodic out-of-sample windows.
- Basic brokerage, half-spread, and slippage cost assumptions.
- Performance summaries for return, risk, drawdown, turnover, and hit rate.
- Optional spread-prediction experiments.

The current walk-forward implementation uses one fixed OLS hedge ratio per training window. It does **not** currently implement a Kalman filter or rolling Johansen hedge-ratio updates. Johansen screening reports a coarse 5% critical-value decision as a proxy rather than a calibrated p-value. Treat its output accordingly.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the research script

```bash
python scripts/run_banking_pairs.py \
  --start 2019-01-01 \
  --leg-y HDFCBANK \
  --leg-x ICICIBANK
```

The data loader downloads daily price data through `yfinance`; results therefore depend on the data vendor, retrieval date, and adjustments. For reproducible analysis, preserve the input dataset, retrieval date, and configuration with each result.

## Backtest return convention

The backtester expresses net returns as decimal fractions of the configured gross pair notional. Spread-price changes are converted using the prior observation's gross pair value; transaction costs are retained in currency units in `BacktestResult.costs` and deducted from returns after normalization by notional. This is a simplified constant-notional approximation, not a broker-grade fill or financing model.

## Repository layout

- `sar/features/` — cointegration methods and hedge-ratio utilities
- `sar/signals/` — spread z-score strategy
- `sar/backtest/` — backtest and cost model
- `sar/metrics/` — performance summaries
- `sar/ml/` — optional spread-prediction experiments
- `scripts/` — runnable research entry point
- `tests/` — synthetic-data regression tests
- `docs/` — architecture and research notes

## Research limitations

- Current data and cost assumptions are simplified; no order-book execution, borrow, funding, taxes, market impact, or corporate-action handling is modelled.
- Strategy sizing and the two-leg return approximation require independent validation before financial interpretation.
- Cointegration selection across many pairs creates multiple-testing risk.
- The optional ML experiment is a research baseline, not evidence of predictive edge.
- Walk-forward evaluation alone does not remove all selection bias or guarantee robustness.

## License

MIT. See [LICENSE](LICENSE).

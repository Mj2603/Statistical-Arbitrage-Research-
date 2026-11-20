# Statistical Arbitrage Research

A Python research prototype for cointegration-based pairs trading on daily equity prices, with an NSE banking example (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK). It covers Engle–Granger and Johansen screening, walk-forward evaluation with training-only pair selection and hedge ratios, a z-score signal, a simple transaction-cost model, performance metrics, and an optional linear/XGBoost spread-prediction baseline.

It is a research and teaching codebase. It is not a live trading system, and nothing here shows that the strategy is profitable on real markets.

## What is implemented

- **Cointegration screening**: Engle–Granger (`statsmodels.coint`, p-value) and Johansen trace test (statistic vs. 10%/5%/1% critical value; no p-value is reported). The Johansen eigenproblem is solved as a symmetric generalised eigenproblem, so it returns real eigenvalues and raises no `ComplexWarning`. Tests check it against `statsmodels.coint_johansen`.
- **Walk-forward evaluation** (`sar/backtest/engine.py`), default 252-day train / 63-day test / 63-day step:
  - `WalkForwardEngine.run`: a pre-specified pair; the cointegration test and OLS hedge ratio use the training window only.
  - `WalkForwardEngine.run_universe`: in each window, tests every pair on the training slice and trades the best-ranked significant pair in the next test slice.
  - The trailing 20-day z-score is warmed up from the end of the training slice, so it does not reset at window boundaries and the first test day can trade.
  - Positions are closed on the last day of each test window, and the exit cost is charged.
  - Windows where no pair passes the test count as flat 0% days in all statistics. They are not dropped.
- **Costs**: by default 10 bps of gross pair notional per unit change in position (3 bps brokerage + 5 bps half-spread + 2 bps slippage). `sar.reporting.run_cost_grid` reruns the walk-forward at several cost levels.
- **Metrics**: Sharpe, Sortino, Calmar, CAGR, max drawdown, annualised volatility, turnover (computed from the actual OOS positions), exposure and hit rate.
- **Tests**: 46 tests run on synthetic data. They include look-ahead tests: randomly perturbing all prices after a cutoff date leaves every earlier OOS return, position and pair selection unchanged.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## Results

### Synthetic (committed)

```bash
python scripts/run_synthetic_experiment.py
```

This writes [`results/synthetic_walkforward.md`](results/synthetic_walkforward.md) and [`results/synthetic_walkforward.csv`](results/synthetic_walkforward.csv). The run is deterministic (fixed seed) and uses **simulated** cointegrated pairs, a regime-break pair and random-walk controls. Gross of costs, the cointegrated pairs average an OOS Sharpe of about 1.1 and the controls about 0. At 10 bps the edge is gone with this calibration. The run checks that the pipeline behaves correctly; it is not evidence about real markets.

### Real NSE data (not yet committed)

Real-data results: run the command below locally (it needs network access to Yahoo Finance). They are not yet committed.

```bash
python scripts/run_banking_pairs.py --start 2019-01-01 --leg-y HDFCBANK --leg-x ICICIBANK
```

The script downloads prices with `yfinance` and saves a snapshot to `data/cache/nse_banking_<date>.csv`. It then writes `results/nse_banking_walkforward.md`, which holds the fixed-pair and per-window-selection walk-forward results at 0/5/10/20 bps, the pair chosen in each window, and the ML baseline. To rerun on a saved snapshot, or on your own wide CSV (a `date` column plus one close-price column per symbol):

```bash
python scripts/run_banking_pairs.py --csv data/cache/nse_banking_YYYYMMDD.csv --start 2019-01-01
```

If the download fails or returns no data, the script stops with a clear error and exits with status 2.

## Backtest return convention

Net returns are decimal fractions of gross pair notional (|y| + |β|·|x| at the prior close). On day t, the position held at the end of day t−1 earns the spread change from t−1 to t. Costs are charged in currency units and deducted after dividing by notional. This is a constant-notional approximation, not a broker-grade fill, financing or borrow model.

## Repository layout

- `sar/features/`: cointegration tests, hedge ratios, spread construction
- `sar/signals/`: trailing z-score entry/exit rules
- `sar/backtest/`: pair backtester, cost model, walk-forward engine
- `sar/metrics/`: performance metrics
- `sar/reporting.py`: cost grids and Markdown tables used by the scripts
- `sar/ml/`: optional spread-change prediction baseline
- `scripts/`: NSE banking run and synthetic experiment
- `results/`: committed experiment outputs (synthetic only so far)
- `tests/`: tests on synthetic data

## Limitations

- **No real-data results are committed yet.** Any performance claim about NSE stocks should wait for `results/nse_banking_walkforward.md` and should be read with the caveats in that file.
- The hedge ratio is a fixed OLS estimate for each training window. There is no Kalman filter and no rolling Johansen re-estimation.
- The cost model leaves out borrow, funding, taxes, market impact, lot sizes and corporate actions beyond the vendor's adjustments. Overnight short selling of NSE cash equities is restricted, so a practical version would need single-stock futures.
- The universe is small (4 stocks, 6 pairs). Selecting pairs by p-value across windows still carries multiple-testing risk: in the synthetic run, about 7% of random-walk control windows pass the 5% screen.
- Signal parameters (entry 2.0, exit 0.0, lookback 20) are fixed defaults and were not optimised. No parameter search is nested inside the walk-forward.
- The ML module is a diagnostic baseline. Its hedge ratio is fitted on the ML training split only, and its predictions do not drive any trades.
- Sharpe ratios estimated from a few years of daily data have wide confidence intervals.

## License

MIT. See [LICENSE](LICENSE).

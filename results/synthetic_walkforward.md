# SYNTHETIC walk-forward experiment

> **All numbers on this page come from simulated prices, not market data.** They
> test whether the pipeline behaves sensibly on data whose properties are known.
> They are not evidence of a profitable strategy.

## Summary

- Gross of costs (0 bps), the cointegrated pairs average an OOS Sharpe of 1.06 versus -0.01 for the random-walk controls.
- At 10 bps per unit position change the cointegrated average is -0.29 (controls -0.18). With this calibration the simulated spread's standard deviation is only 0.23% of gross pair notional, so the expected capture of a 2-sigma entry is of the same order as the round-trip cost: the net result is cost-dominated.
- The 5% Engle-Granger screen admitted 7% of control windows (false positives) and 52% of truly cointegrated windows; power falls sharply as the spread half-life lengthens (see per-pair table).

Reproduce: `python scripts/run_synthetic_experiment.py --seed 20240601`
(Python 3.13.18, numpy 2.5.5, pandas 3.1.1).
Raw rows for every pair and cost level: [`synthetic_walkforward.csv`](synthetic_walkforward.csv).

## Setup

- 2016 business days per series (about 8 years), seed 20240601.
- Price levels follow arithmetic random walks starting at 400 with daily sd 1.0;
  cointegrated spreads have stationary sd 2.0 price units. These parameters were
  chosen once before the first run and were not tuned to the results.
- Walk-forward: train 252 / test 63 / step 63 days; Engle-Granger at 5% on the
  training window; OLS hedge ratio from the training window only.
- Signal: trailing 20-day z-score (warmed up from the training window, no reset),
  enter at |z| >= 2, exit at z = 0, flatten on the last day of each test window.
- Costs: `cost_bps` of gross notional per unit change in position. Windows that
  fail the cointegration test are kept as flat 0% days in all statistics.

| pair | type | data-generating process |
|---|---|---|
| COINT_1 | cointegrated | AR(1) spread, phi=0.8, half-life 3.1d, beta=1.0 |
| COINT_2 | cointegrated | AR(1) spread, phi=0.9, half-life 6.6d, beta=1.2 |
| COINT_3 | cointegrated | AR(1) spread, phi=0.95, half-life 13.5d, beta=0.8 |
| COINT_4 | cointegrated | AR(1) spread, phi=0.98, half-life 34.3d, beta=1.5 |
| BREAK_1 | regime break | AR(1) phi=0.90 for first half, random-walk spread after |
| CONTROL_1 | not cointegrated | two independent random walks |
| CONTROL_2 | not cointegrated | two independent random walks |
| CONTROL_3 | not cointegrated | two independent random walks |
| CONTROL_4 | not cointegrated | two independent random walks |

## Per-pair results at 10 bps

| pair | type | oos_days | windows_traded | windows | sharpe | cagr | max_drawdown | turnover_per_year | exposure | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| COINT_1 | cointegrated | 1764 | 26 | 28 | 1.05 | 1.06% | -1.50% | 17.1 | 21.2% | 53.2% |
| COINT_2 | cointegrated | 1764 | 21 | 28 | -0.32 | -0.23% | -2.47% | 15.4 | 22.7% | 48.8% |
| COINT_3 | cointegrated | 1764 | 9 | 28 | -0.85 | -0.48% | -3.46% | 5.4 | 12.3% | 43.2% |
| COINT_4 | cointegrated | 1764 | 2 | 28 | -1.02 | -0.16% | -1.15% | 1.1 | 5.0% | 41.9% |
| BREAK_1 | regime break | 1764 | 17 | 28 | -0.61 | -0.50% | -4.03% | 14.3 | 27.3% | 45.7% |
| CONTROL_1 | not cointegrated | 1764 | 1 | 28 | -0.39 | -0.15% | -1.33% | 0.6 | 2.3% | 45.2% |
| CONTROL_2 | not cointegrated | 1764 | 6 | 28 | -0.34 | -0.38% | -4.50% | 4.6 | 11.4% | 45.6% |
| CONTROL_3 | not cointegrated | 1764 | 1 | 28 | 0.00 | 0.00% | -1.27% | 0.6 | 0.9% | 55.6% |
| CONTROL_4 | not cointegrated | 1764 | 0 | 28 | 0.00 | 0.00% | 0.00% | 0.0 | 0.0% | 0.0% |

## Cost sensitivity by group

| type | cost_bps | mean_sharpe | mean_cagr | worst_max_drawdown | windows_traded_pct |
|---|---|---|---|---|---|
| cointegrated | 0 | 1.06 | 1.04% | -2.26% | 52% |
| cointegrated | 5 | 0.40 | 0.54% | -2.36% | 52% |
| cointegrated | 10 | -0.29 | 0.05% | -3.46% | 52% |
| cointegrated | 20 | -1.36 | -0.93% | -12.01% | 52% |
| regime break | 0 | 1.13 | 0.93% | -1.56% | 61% |
| regime break | 5 | 0.27 | 0.21% | -2.13% | 61% |
| regime break | 10 | -0.61 | -0.50% | -4.03% | 61% |
| regime break | 20 | -2.00 | -1.91% | -12.65% | 61% |
| not cointegrated | 0 | -0.01 | 0.01% | -3.00% | 7% |
| not cointegrated | 5 | -0.09 | -0.06% | -3.56% | 7% |
| not cointegrated | 10 | -0.18 | -0.13% | -4.50% | 7% |
| not cointegrated | 20 | -0.35 | -0.27% | -6.83% | 7% |

## Reading the results

- `windows_traded` for the controls is the empirical false-positive rate of the
  5% Engle-Granger screen on 252-day windows; trades taken in those windows are
  what multiple testing costs in practice.
- Returns are a fraction of gross pair notional, so CAGR figures are small.
- Turnover per year is the sum of absolute position changes (1 = open or close
  a full-size spread position once).

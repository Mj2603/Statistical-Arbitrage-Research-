Author notes — Research project (stat-arb)
=========================================

Purpose
-------

This document records the author's intent, design choices, and experiment notes so
that a reviewer or recruiter can quickly understand what I implemented and why.

Design highlights
-----------------

- Cointegration and hedge-ratio estimation: Engle-Granger and Johansen are
  implemented in `sar/features/cointegration.py`. These are used to find
  candidate pairs and to construct the tradable spread.
- Signals and backtest: z-score entry/exit signals implemented in `sar/signals`;
  a walk-forward engine (`sar/backtest/engine.py`) freezes hedge-ratios on
  training windows and evaluates OOS performance.
- Costs & metrics: `sar/backtest/costs.py` implements a simple round-trip and
  per-leg cost model; `sar/metrics` contains Sharpe/Sortino/Calmar/drawdown
  utilities and a small attribution helper.
- ML experiments: `sar/ml/predictor.py` is a deliberately minimal, transparent
  feature-based predictor used to explore whether short-horizon spread deltas
  are predictable. It includes a linear baseline and an XGBoost baseline.

Why this structure
-------------------

I intentionally kept the ML piece small and separated from execution logic so
that (a) experiments are easy to reproduce and (b) the research code remains
auditable. The feature set (lag features, 20-day z-score, 20-day vol) focuses
on interpretability rather than raw predictive power.

Reproducibility notes
---------------------

1. Create a virtual environment and install dependencies from `requirements.txt`.

2. Run the main research script (example):

```bash
python scripts/run_banking_pairs.py --start 2019-01-01 --leg-y HDFCBANK --leg-x ICICIBANK
```

3. Run unit tests:

```bash
pytest -q
```

ML/hyperparameters used in experiments
--------------------------------------

- Linear baseline: scikit-learn `LinearRegression()` (default settings).
- XGBoost baseline: `XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8)`.

Notes for reviewers
-------------------

- The ML module is experimental: OOS R^2 is reported for comparison, but I do
  not use ML predictions in live execution — the backtest uses z-score signals.
- Tests include a small synthetic-data test for ML; `tests/test_ml.py` skips the
  XGBoost test if `xgboost` is not installed so the test suite remains light.

Contact / next steps
--------------------

If you want a short notebook visualizing the spread, features and predictions
I can add `notebooks/exploration.ipynb` showing the exploratory plots and
intermediate outputs used during development.

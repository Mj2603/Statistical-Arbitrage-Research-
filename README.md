# Stat-Arb Research Platform

Python research framework for pairs trading and cointegration on NSE banking names. Covers signal generation, walk-forward backtesting, transaction costs, performance metrics, and optional ML spread prediction.

## Requirements

Python 3.10+, see `requirements.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run

```bash
python scripts/run_banking_pairs.py --start 2019-01-01 --leg-y HDFCBANK --leg-x ICICIBANK
```

## Layout

- `sar/` — library
- `tests/` — pytest (synthetic data, no network)
- `scripts/` — research runs
- `docs/` — notes

## Pipeline

Data -> cointegration scan -> z-score signals -> walk-forward backtest -> metrics / ML

Default universe: HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK (.NS via yfinance).

## Scope

Implemented: Engle-Granger, Johansen, walk-forward OOS, cost model, Sharpe/Sortino/Calmar/drawdown/turnover/hit rate, linear + XGBoost spread prediction.

## License

MIT

## Author contributions

This project and its code were developed as an individual research project. Key contributions and decisions:

- Implemented cointegration scanning and hedge-ratio estimation (`sar/features/cointegration.py`) using Engle-Granger and Johansen tests.
- Built a transparent signal/backtest pipeline (`sar/signals`, `sar/backtest`) with a simple transaction cost model (`sar/backtest/costs.py`).
- Implemented spread prediction experiments in `sar/ml/predictor.py` (feature engineering + two baselines: LinearRegression and XGBoost). Feature choices (lag features, 20-day z-score, rolling vol) are documented in the file.
- Added unit tests (see `tests/test_ml.py`) and simple synthetic-data tests for core functionality.

Reproducibility / how I ran experiments:

1. Create and activate a venv, then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full research run (downloads data via `yfinance` by default):

```bash
python scripts/run_banking_pairs.py --start 2019-01-01 --leg-y HDFCBANK --leg-x ICICIBANK
```

3. Run ML experiments independently:

```py
from sar.data.loader import DataLoader
from sar.features.cointegration import compute_spread, ols_hedge_ratio
from sar.ml.predictor import SpreadPredictor

panel = DataLoader().load_banking_universe(start="2019-01-01")
prices = panel.close
y, x = prices['HDFCBANK'], prices['ICICIBANK']
beta = ols_hedge_ratio(y, x)
spread = compute_spread(y, x, beta)
ml = SpreadPredictor(lookback=5)
res = ml.fit(spread, method='linear')
print('OOS R2:', res.r2_oos)
```



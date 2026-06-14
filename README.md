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

Not done: live execution, multi-asset portfolio optimizer, futures roll, intraday data.

## License

MIT

Walkthrough — running experiments and ML quick-start
===================================================

This short walkthrough shows the commands and minimal code snippets I used to
run experiments and validate the ML baselines.

Quick commands
--------------

Create a venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main script (example):

```bash
python scripts/run_banking_pairs.py --start 2019-01-01 --leg-y HDFCBANK --leg-x ICICIBANK
```

Run only the ML unit tests:

```bash
pytest -q tests/test_ml.py
```

Example: run the predictor interactively
---------------------------------------

Save this snippet as `examples/run_predictor.py` and run it to reproduce a
single ML baseline run.

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
print('linear OOS R2 =', res.r2_oos)

try:
    res_xgb = ml.fit(spread, method='xgboost')
    print('xgboost OOS R2 =', res_xgb.r2_oos)
except Exception as exc:
    print('xgboost unavailable or failed:', exc)
```

Commit message examples (used for this project)
-----------------------------------------------

I prefer clear, action-oriented commit messages. Examples used for the
changes in this branch:

- `docs: add author notes describing design and reproducibility steps`
- `docs: add walkthrough and example script for ML experiments`
- `feat: annotate SpreadPredictor with design rationale and feature notes`

These messages are short, deterministic, and explain what changed.

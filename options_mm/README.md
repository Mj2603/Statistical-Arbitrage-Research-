Options Market-Making Simulator (phase 1)
======================================

This is a professional-grade options market-making simulator scaffold. It focuses
on reproducibility, modularity, and progressive complexity: data -> pricing ->
quoting -> execution -> risk -> backtest.

Phase 1 implemented here:

- Market Data Engine: CSV loader and sample tick data

- Pricing Engine: Black–Scholes pricing + Greeks

Quick start
-----------

Create a venv and install the small set of dependencies used for Phase 1:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r options_mm/requirements.txt
```

Run the Phase 1 demo which loads sample ticks and computes theoretical prices:

```bash
python options_mm/scripts/run_phase1.py
```

Project layout
--------------

- `options_mm/src/options_mm/data` — market data loader and helpers
- `options_mm/src/options_mm/pricing` — Black–Scholes and Greeks
- `options_mm/scripts` — runnable examples and demos
- `options_mm/data` — sample CSV ticks for phase 1
- `options_mm/tests` — unit tests for core pricing functions

Next steps
----------

Phase 2 will expand the pricing engine, then we'll add a Quote Engine and an
Execution Engine with inventory tracking and delta-hedging.

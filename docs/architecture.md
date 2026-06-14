# Notes

Research pipeline for pairs stat-arb. Not a live strategy.

```
DataLoader -> CointegrationEngine -> SignalGenerator -> PairsBacktester
                                              |
                                    WalkForwardEngine (train/test/roll)
                                              |
                              PerformanceAttribution + RiskAttribution
                                              |
                                    SpreadPredictor (linear / xgboost)
```

## Modules

- `sar/data` — NSE price loading via yfinance or CSV
- `sar/features` — Engle-Granger, Johansen, hedge ratio, spread
- `sar/signals` — z = (spread - mu) / sigma, entry +/-2, exit 0
- `sar/backtest` — pairs PnL, transaction costs, walk-forward
- `sar/metrics` — Sharpe, Sortino, Calmar, drawdown, turnover, hit rate
- `sar/risk` — rolling factor beta
- `sar/ml` — spread change prediction

## Walk-forward

Each window: estimate cointegration on train, trade only OOS segment with frozen hedge ratio from train. Windows roll forward by `step_days`.

## Costs

Per-leg: brokerage + half-spread + slippage in bps. Applied on position changes.

## Scope

NSE banking universe hardcoded (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK). Single pair backtest. ML is feature-based spread delta prediction, not integrated into execution.

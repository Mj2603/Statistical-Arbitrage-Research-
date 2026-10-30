# Research Architecture

This repository is a reproducible research prototype for cointegration-based pairs strategies on daily NSE equity data. It separates data loading, pair diagnostics, signal generation, backtesting, and evaluation so that each stage can be inspected independently.

## Research flow

```text
Daily price panel
    ├── Pair diagnostics (Engle–Granger or Johansen)
    └── Walk-forward evaluation
          ├── Fit cointegration and a fixed OLS hedge ratio on training data
          ├── Generate z-score signals for the selected pair
          ├── Apply the configured transaction-cost approximation
          └── Report out-of-sample returns and performance metrics
```

The optional linear and XGBoost experiments predict spread changes. They are research baselines and are not connected to the backtest's trading decisions.

## Modules

| Path | Responsibility |
|---|---|
| `sar/data/` | Load daily prices from yfinance or CSV |
| `sar/features/` | Engle–Granger/Johansen diagnostics, OLS hedge ratios, and spread construction |
| `sar/signals/` | Z-score entry and exit rules |
| `sar/backtest/` | Pair backtest, costs, and rolling train/test evaluation |
| `sar/metrics/` | Sharpe, Sortino, Calmar, drawdown, turnover, and hit rate |
| `sar/risk/` | Rolling factor-beta analysis |
| `sar/ml/` | Optional linear and XGBoost spread-change prediction |

## Walk-forward and return conventions

Each walk-forward window estimates the pair relationship on its training interval and trades only the following out-of-sample interval with the training-window OLS hedge ratio held fixed. The implementation does not currently use a Kalman filter or re-estimate Johansen hedge ratios continuously.

Net returns are expressed as decimal fractions of the configured gross pair notional. Spread-price changes are converted using the prior observation's gross pair value. Costs are stored in currency units and deducted after normalization by notional. This is a simplified constant-notional approximation, not a broker-grade two-leg fill, financing, or borrow model.

## Scope and limitations

- The example universe is a small set of NSE banking equities; the backtest evaluates a selected pair.
- Prices retrieved through yfinance depend on the vendor, retrieval date, and adjustment policy.
- The cost model does not include borrow, funding, taxes, market impact, or corporate actions.
- Pair selection across many candidates creates multiple-testing risk.
- Walk-forward evaluation reduces temporal leakage but does not remove selection bias or establish profitability.

See the [README](../README.md) for installation, the runnable command, and the full research limitations.

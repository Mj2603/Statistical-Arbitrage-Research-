#!/usr/bin/env python3
"""Walk-forward pairs research on NSE banking stocks.

Downloads daily closes via yfinance (or reads ``--csv``), runs

1. a walk-forward backtest of the pre-specified pair ``--leg-y``/``--leg-x``;
2. a walk-forward backtest that re-selects the best cointegrated pair from the
   universe inside every training window (selection uses training data only);
3. an optional spread-prediction baseline whose hedge ratio is fitted on the
   ML training split only;

and writes a Markdown report (default ``results/nse_banking_walkforward.md``).
"""

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sar.backtest.engine import WalkForwardEngine
from sar.data.loader import NSE_BANKING, DataLoader, DataUnavailableError
from sar.features.cointegration import compute_spread, ols_hedge_ratio
from sar.ml.predictor import SpreadPredictor
from sar.reporting import (
    DEFAULT_COST_GRID_BPS,
    cost_model_from_bps,
    markdown_table,
    run_cost_grid,
    summarize,
)
from sar.signals.zscore import SignalParams

METRIC_COLUMNS = [
    "cost_bps",
    "oos_days",
    "windows_traded",
    "windows",
    "sharpe",
    "cagr",
    "max_drawdown",
    "ann_vol",
    "turnover_per_year",
    "exposure",
    "hit_rate",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NSE banking pairs walk-forward research run")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--leg-y", default="HDFCBANK")
    p.add_argument("--leg-x", default="ICICIBANK")
    p.add_argument(
        "--csv",
        default=None,
        help="Read prices from a wide CSV (date column + one close column per symbol) "
        "instead of downloading from yfinance.",
    )
    p.add_argument(
        "--save-csv",
        default=None,
        help="Save the downloaded price panel to this CSV for reproducibility "
        "(default: data/cache/nse_banking_<today>.csv when downloading).",
    )
    p.add_argument("--output", default="results/nse_banking_walkforward.md")
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--test-days", type=int, default=63)
    p.add_argument("--step-days", type=int, default=63)
    p.add_argument("--cost-bps", type=float, nargs="+", default=list(DEFAULT_COST_GRID_BPS))
    p.add_argument("--no-ml", action="store_true", help="Skip the spread-prediction baseline.")
    return p.parse_args(argv)


def load_prices(args):
    loader = DataLoader()
    if args.csv:
        panel = loader.load_csv(args.csv)
        source = f"CSV file `{args.csv}`"
    else:
        panel = loader.load_banking_universe(start=args.start, end=args.end)
        save_path = args.save_csv or f"data/cache/nse_banking_{dt.date.today():%Y%m%d}.csv"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        loader.save_csv(panel, save_path)
        source = (
            "yfinance (auto-adjusted closes) for "
            + ", ".join(f"{k}={v}" for k, v in NSE_BANKING.items())
            + f", retrieved {dt.date.today()}; snapshot saved to `{save_path}`"
        )
    prices = panel.close
    prices = prices.loc[args.start : args.end] if args.end else prices.loc[args.start :]
    if prices.empty:
        raise DataUnavailableError("no price rows in the requested date range")
    return prices, source


def ml_baseline(prices, leg_y, leg_x, train_frac=0.7, use_xgboost=True):
    """Spread-prediction baseline with the hedge ratio fitted on the ML train split only."""
    pair = prices[[leg_y, leg_x]].dropna()
    y, x = pair[leg_y], pair[leg_x]
    ml = SpreadPredictor()
    # The feature/target row index does not depend on the hedge ratio, so a
    # placeholder spread identifies the train/test boundary date.
    rows = ml._build_xy(compute_spread(y, x, 1.0))[0].index
    train_end = rows[int(len(rows) * train_frac) - 1]
    beta = ols_hedge_ratio(y.loc[:train_end], x.loc[:train_end])
    spread = compute_spread(y, x, beta)
    out = {"beta": beta, "train_end": train_end, "linear": ml.fit_linear(spread, train_frac).r2_oos}
    out["xgboost"] = float("nan")
    if use_xgboost:
        try:
            out["xgboost"] = ml.fit_xgboost(spread, train_frac).r2_oos
        except ImportError:
            pass
    return out


def main(argv=None):
    args = parse_args(argv)
    try:
        prices, source = load_prices(args)
    except DataUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for leg in (args.leg_y, args.leg_x):
        if leg not in prices:
            print(f"error: {leg} not in price columns {list(prices.columns)}", file=sys.stderr)
            return 2

    windows = dict(train_days=args.train_days, test_days=args.test_days, step_days=args.step_days)
    params = SignalParams()

    pair_rows = run_cost_grid(prices, args.cost_bps, pair=(args.leg_y, args.leg_x), **windows)
    universe_rows = run_cost_grid(prices, args.cost_bps, pair=None, **windows)

    base_engine = WalkForwardEngine(**windows, costs=cost_model_from_bps(10.0))
    pair_res = base_engine.run(prices, args.leg_y, args.leg_x)
    universe_res = base_engine.run_universe(prices)

    print(f"data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} rows)")
    for name, res in (("fixed pair", pair_res), ("per-window selection", universe_res)):
        s = summarize(res)
        print(
            f"{name} @10bps: OOS sharpe={s['sharpe']:.2f} cagr={s['cagr']:.2%} "
            f"maxdd={s['max_drawdown']:.2%} turnover/yr={s['turnover_per_year']:.1f} "
            f"windows traded={s['windows_traded']}/{s['windows']}"
        )

    ml = None
    if not args.no_ml:
        ml = ml_baseline(prices, args.leg_y, args.leg_x)
        print(f"ML oos R2 linear={ml['linear']:.3f} xgboost={ml['xgboost']:.3f}")

    log_rows = [
        {
            "test_start": str(pd.Timestamp(r.test_start).date()),
            "selected_pair": "/".join(r.pair) if r.pair else "none (flat)",
            "hedge_ratio": r.hedge_ratio,
        }
        for r in universe_res.window_log.itertuples()
    ]

    lines = [
        "# NSE banking pairs: walk-forward results "
        + ("(user-supplied CSV; verify its provenance)" if args.csv else "(market data via yfinance)"),
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/run_banking_pairs.py`.",
        "",
        "## Configuration",
        "",
        f"- Data: {source}",
        f"- Sample: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} rows, "
        f"columns {list(prices.columns)}",
        f"- Walk-forward: train {args.train_days} / test {args.test_days} / step "
        f"{args.step_days} trading days; Engle-Granger at 5%; OLS hedge ratio from the "
        "training window only",
        f"- Signal: entry |z| >= {params.entry_z}, exit at z = {params.exit_z}, trailing "
        f"{params.lookback}-day z-score warmed up from the training window; positions "
        "closed on the last day of each test window",
        "- Costs: `cost_bps` of gross notional per unit change in position (entry and "
        "exit are each charged). Windows where no pair passes the test count as flat "
        "0% days.",
        f"- Environment: Python {platform.python_version()}, pandas {pd.__version__}, "
        f"numpy {np.__version__}",
        "",
        f"## Pre-specified pair {args.leg_y}/{args.leg_x}",
        "",
        markdown_table(pair_rows, METRIC_COLUMNS),
        "",
        "## Per-window pair selection across the universe (training data only)",
        "",
        markdown_table(universe_rows, METRIC_COLUMNS),
        "",
        "Pairs selected per test window:",
        "",
        markdown_table(log_rows, ["test_start", "selected_pair", "hedge_ratio"]),
        "",
    ]
    if ml is not None:
        lines += [
            "## Spread-prediction baseline (diagnostic only, not used for trading)",
            "",
            f"Hedge ratio {ml['beta']:.3f} fitted on data up to {ml['train_end'].date()} "
            f"(ML train split). OOS R^2: linear {ml['linear']:.3f}, XGBoost {ml['xgboost']:.3f}.",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- Small universe (4 stocks, 6 pairs); results are sensitive to the sample period, "
        "vendor adjustments and retrieval date.",
        "- The cost model ignores borrow, funding, taxes, market impact and lot sizes; "
        "overnight short selling of NSE cash equities is restricted, so a real "
        "implementation would need single-stock futures.",
        "- Sharpe estimates over a few years of daily data have wide confidence intervals.",
        "",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

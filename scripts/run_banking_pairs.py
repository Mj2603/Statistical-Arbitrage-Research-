#!/usr/bin/env python3

import argparse

from sar.backtest.engine import WalkForwardEngine
from sar.data.loader import DataLoader
from sar.features.cointegration import CointegrationEngine
from sar.metrics.performance import PerformanceAttribution
from sar.ml.predictor import SpreadPredictor
from sar.features.cointegration import compute_spread


def main():
    p = argparse.ArgumentParser(description="NSE banking pairs stat-arb research run")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--leg-y", default="HDFCBANK")
    p.add_argument("--leg-x", default="ICICIBANK")
    args = p.parse_args()

    panel = DataLoader().load_banking_universe(start=args.start, end=args.end)
    prices = panel.close

    coint = CointegrationEngine()
    pairs = coint.scan_pairs(prices, method="engle_granger")
    print(f"cointegrated pairs: {len(pairs)}")
    for r in pairs[:5]:
        print(f"  {r.pair} p={r.pvalue:.4f} beta={r.hedge_ratio:.3f}")

    wf = WalkForwardEngine(train_days=252, test_days=63, step_days=63)
    wf_res = wf.run(prices, args.leg_y, args.leg_x)
    perf = PerformanceAttribution().compute(wf_res.oos_returns)
    print(f"OOS sharpe={perf.sharpe:.2f} sortino={perf.sortino:.2f} maxdd={perf.max_drawdown:.2%}")
    print(f"turnover={perf.turnover:.2f} hit_rate={perf.hit_rate:.2%}")

    if args.leg_y in prices and args.leg_x in prices:
        y, x = prices[args.leg_y], prices[args.leg_x]
        beta = pairs[0].hedge_ratio if pairs else 1.0
        spread = compute_spread(y, x, beta)
        ml = SpreadPredictor()
        lin = ml.fit_linear(spread)
        xgb = ml.fit_xgboost(spread)
        print(f"ML oos R2 linear={lin.r2_oos:.3f} xgboost={xgb.r2_oos:.3f}")


if __name__ == "__main__":
    main()

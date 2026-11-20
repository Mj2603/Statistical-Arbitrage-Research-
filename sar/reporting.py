"""Helpers that turn walk-forward results into metric rows and Markdown tables."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sar.backtest.costs import TransactionCostModel
from sar.backtest.engine import WalkForwardEngine, WalkForwardResult
from sar.metrics.performance import PerformanceAttribution
from sar.signals.zscore import SignalParams

DEFAULT_COST_GRID_BPS = (0.0, 5.0, 10.0, 20.0)


def cost_model_from_bps(bps: float) -> TransactionCostModel:
    """Cost model charging ``bps`` of gross notional per unit position change."""
    return TransactionCostModel(brokerage_bps=bps, half_spread_bps=0.0, slippage_bps=0.0)


def summarize(result: WalkForwardResult, periods: int = 252) -> Dict[str, float]:
    perf = PerformanceAttribution(periods).compute(result.oos_returns, result.oos_positions)
    log = result.window_log
    n_windows = len(log)
    traded = int(log["traded"].sum()) if n_windows else 0
    return {
        "oos_days": perf.n_days,
        "windows": n_windows,
        "windows_traded": traded,
        "sharpe": perf.sharpe,
        "cagr": perf.cagr,
        "max_drawdown": perf.max_drawdown if perf.n_days else 0.0,
        "ann_vol": perf.ann_vol,
        "total_return": perf.total_return,
        "turnover_per_day": perf.turnover,
        "turnover_per_year": perf.turnover * periods,
        "exposure": perf.exposure,
        "hit_rate": perf.hit_rate,
    }


def run_cost_grid(
    prices: pd.DataFrame,
    cost_grid_bps: Iterable[float] = DEFAULT_COST_GRID_BPS,
    pair: Optional[tuple] = None,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    coint_method: str = "engle_granger",
    pvalue_threshold: float = 0.05,
    signal_params: Optional[SignalParams] = None,
) -> List[Dict[str, float]]:
    """Run the same walk-forward once per cost level and return metric rows."""
    rows = []
    for bps in cost_grid_bps:
        engine = WalkForwardEngine(
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            coint_method=coint_method,
            pvalue_threshold=pvalue_threshold,
            signal_params=signal_params,
            costs=cost_model_from_bps(bps),
        )
        if pair is None:
            res = engine.run_universe(prices)
        else:
            res = engine.run(prices, pair[0], pair[1])
        row = {"cost_bps": float(bps)}
        row.update(summarize(res))
        rows.append(row)
    return rows


_FORMATS = {
    "sharpe": "{:.2f}",
    "cagr": "{:.2%}",
    "max_drawdown": "{:.2%}",
    "ann_vol": "{:.2%}",
    "total_return": "{:.2%}",
    "turnover_per_day": "{:.4f}",
    "turnover_per_year": "{:.1f}",
    "exposure": "{:.1%}",
    "hit_rate": "{:.1%}",
    "cost_bps": "{:g}",
}


def _fmt(key: str, value) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return _FORMATS.get(key, "{:.4g}").format(value)
    return str(value)


def markdown_table(rows: List[Dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(c, r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    turnover: float
    hit_rate: float
    total_return: float
    ann_vol: float
    cagr: float = 0.0
    # Fraction of days with a non-zero position.
    exposure: float = 0.0
    n_days: int = 0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods) * returns.mean() / returns.std())


def sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * returns.mean() / downside.std())


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods: int = 252) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    ann_ret = (1 + returns.mean()) ** periods - 1
    return float(ann_ret / dd)


def turnover(positions: pd.Series) -> float:
    """Average absolute position change per day.

    The first observation counts as a change from flat, so an entry on day one
    is not missed. Multiply by 252 for annual turnover.
    """
    if len(positions) == 0:
        return 0.0
    changes = positions.diff().abs().fillna(positions.abs())
    return float(changes.sum() / len(positions))


def cagr(equity: pd.Series, periods: int = 252) -> float:
    if len(equity) == 0 or equity.iloc[-1] <= 0:
        return 0.0
    years = len(equity) / periods
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


class PerformanceAttribution:
    def __init__(self, periods: int = 252):
        self.periods = periods

    def compute(self, returns: pd.Series, positions: Optional[pd.Series] = None) -> PerformanceReport:
        equity = (1 + returns.fillna(0)).cumprod()
        if positions is None:
            # Turnover cannot be computed without positions; report NaN rather
            # than a misleading 0.0.
            turnover_value = float("nan")
            exposure = float("nan")
        else:
            pos = positions.reindex(returns.index).fillna(0.0)
            turnover_value = turnover(pos)
            exposure = float((pos != 0).mean()) if len(pos) else 0.0
        return PerformanceReport(
            sharpe=sharpe_ratio(returns, self.periods),
            sortino=sortino_ratio(returns, self.periods),
            calmar=calmar_ratio(returns, equity, self.periods),
            max_drawdown=max_drawdown(equity),
            turnover=turnover_value,
            hit_rate=hit_rate(returns),
            total_return=float(equity.iloc[-1] - 1) if len(equity) else 0.0,
            ann_vol=float(returns.std() * np.sqrt(self.periods)) if len(returns) > 1 else 0.0,
            cagr=cagr(equity, self.periods),
            exposure=exposure,
            n_days=int(len(returns)),
        )

    def regime_breakdown(
        self,
        returns: pd.Series,
        benchmark_vol: pd.Series,
        quantile: float = 0.5,
    ) -> Dict[str, PerformanceReport]:
        aligned = pd.concat([returns, benchmark_vol], axis=1).dropna()
        if aligned.empty:
            return {}
        threshold = aligned.iloc[:, 1].quantile(quantile)
        low = aligned[aligned.iloc[:, 1] <= threshold].iloc[:, 0]
        high = aligned[aligned.iloc[:, 1] > threshold].iloc[:, 0]
        return {
            "low_vol": self.compute(low),
            "high_vol": self.compute(high),
        }

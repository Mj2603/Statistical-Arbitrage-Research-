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
    return float(positions.diff().abs().sum() / max(len(positions), 1))


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
        pos = positions if positions is not None else pd.Series(0, index=returns.index)
        return PerformanceReport(
            sharpe=sharpe_ratio(returns, self.periods),
            sortino=sortino_ratio(returns, self.periods),
            calmar=calmar_ratio(returns, equity, self.periods),
            max_drawdown=max_drawdown(equity),
            turnover=turnover(pos),
            hit_rate=hit_rate(returns),
            total_return=float(equity.iloc[-1] - 1) if len(equity) else 0.0,
            ann_vol=float(returns.std() * np.sqrt(self.periods)),
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

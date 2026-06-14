from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sar.backtest.costs import TransactionCostModel
from sar.features.cointegration import CointegrationEngine, compute_spread, ols_hedge_ratio
from sar.signals.zscore import SignalGenerator, SignalParams


@dataclass
class BacktestResult:
    returns: pd.Series
    positions: pd.Series
    spread: pd.Series
    zscore: pd.Series
    costs: pd.Series
    equity: pd.Series


class PairsBacktester:
    def __init__(
        self,
        signal_params: Optional[SignalParams] = None,
        costs: Optional[TransactionCostModel] = None,
        notional: float = 1_000_000,
    ):
        self.signal = SignalGenerator(signal_params)
        self.costs = costs or TransactionCostModel()
        self.notional = notional

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        hedge_ratio: float,
    ) -> BacktestResult:
        spread = compute_spread(y, x, hedge_ratio)
        sig = self.signal.generate(spread)
        pos = sig.position.astype(float)

        spread_ret = spread.diff()
        gross = pos.shift(1).fillna(0) * spread_ret

        cost_series = pd.Series(0.0, index=spread.index)
        turnover = pos.diff().abs().fillna(0)
        for i, dt in enumerate(spread.index):
            if turnover.iloc[i] > 0:
                cost_series.iloc[i] = self.costs.single_leg_cost(self.notional) * turnover.iloc[i]

        net = gross - cost_series
        equity = (1 + net.fillna(0)).cumprod()

        return BacktestResult(
            returns=net.fillna(0),
            positions=pos,
            spread=spread,
            zscore=sig.zscore,
            costs=cost_series,
            equity=equity,
        )


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    windows: List[WalkForwardWindow]
    oos_returns: pd.Series
    hedge_ratios: pd.Series


class WalkForwardEngine:
    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
        coint_method: str = "engle_granger",
        pvalue_threshold: float = 0.05,
        signal_params: Optional[SignalParams] = None,
        costs: Optional[TransactionCostModel] = None,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.coint_method = coint_method
        self.pvalue_threshold = pvalue_threshold
        self.backtester = PairsBacktester(signal_params, costs)
        self.coint = CointegrationEngine()

    def _windows(self, index: pd.DatetimeIndex) -> List[WalkForwardWindow]:
        windows = []
        i = 0
        while i + self.train_days + self.test_days <= len(index):
            train_idx = index[i : i + self.train_days]
            test_idx = index[i + self.train_days : i + self.train_days + self.test_days]
            windows.append(
                WalkForwardWindow(
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                )
            )
            i += self.step_days
        return windows

    def run(
        self,
        prices: pd.DataFrame,
        leg_y: str,
        leg_x: str,
    ) -> WalkForwardResult:
        windows = self._windows(prices.index)
        oos_parts = []
        hedge_log = {}

        for w in windows:
            train = prices.loc[w.train_start:w.train_end, [leg_y, leg_x]]
            test = prices.loc[w.test_start:w.test_end, [leg_y, leg_x]]

            if self.coint_method == "johansen":
                cres = self.coint.johansen(train, (leg_y, leg_x))
            else:
                cres = self.coint.engle_granger(train[leg_y], train[leg_x], (leg_y, leg_x))

            if cres.pvalue > self.pvalue_threshold:
                continue

            hedge = cres.hedge_ratio
            hedge_log[w.test_start] = hedge

            bt = self.backtester.run(test[leg_y], test[leg_x], hedge)
            oos_parts.append(bt.returns)

        if not oos_parts:
            empty = pd.Series(dtype=float)
            return WalkForwardResult(windows=windows, oos_returns=empty, hedge_ratios=pd.Series(dtype=float))

        oos = pd.concat(oos_parts).sort_index()
        oos = oos[~oos.index.duplicated(keep="first")]
        return WalkForwardResult(
            windows=windows,
            oos_returns=oos,
            hedge_ratios=pd.Series(hedge_log),
        )

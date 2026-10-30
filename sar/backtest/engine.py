from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sar.backtest.costs import TransactionCostModel
from sar.features.cointegration import CointegrationEngine, compute_spread
from sar.signals.zscore import SignalGenerator, SignalParams


@dataclass
class BacktestResult:
    # Net portfolio returns as decimal fractions of gross notional.
    returns: pd.Series
    positions: pd.Series
    spread: pd.Series
    zscore: pd.Series
    # Transaction costs in currency units, for transparency.
    costs: pd.Series
    # Growth of one unit of starting capital.
    equity: pd.Series


class PairsBacktester:
    def __init__(
        self,
        signal_params: Optional[SignalParams] = None,
        costs: Optional[TransactionCostModel] = None,
        notional: float = 1_000_000,
    ):
        if not np.isfinite(notional) or notional <= 0:
            raise ValueError("notional must be finite and positive")
        self.signal = SignalGenerator(signal_params)
        self.costs = costs or TransactionCostModel()
        self.notional = float(notional)

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        hedge_ratio: float,
    ) -> BacktestResult:
        if not np.isfinite(hedge_ratio):
            raise ValueError("hedge_ratio must be finite")
        aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if aligned.empty:
            raise ValueError("price series have no aligned observations")
        if (aligned <= 0).any().any():
            raise ValueError("price observations must be positive")

        y_aligned = aligned["y"]
        x_aligned = aligned["x"]
        spread = compute_spread(y_aligned, x_aligned, hedge_ratio)
        sig = self.signal.generate(spread)
        pos = sig.position.astype(float)

        # Translate spread price changes into return on gross pair notional.
        # The denominator uses the prior close because the prior position earns
        # the move from t-1 to t.
        prior_gross_value = y_aligned.shift(1).abs() + abs(hedge_ratio) * x_aligned.shift(1).abs()
        spread_change = y_aligned.diff() - hedge_ratio * x_aligned.diff()
        gross_returns = (
            pos.shift(1).fillna(0) * spread_change.div(prior_gross_value)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        turnover = pos.diff().abs().fillna(pos.abs())
        costs = turnover * self.costs.single_leg_cost(self.notional)
        cost_returns = costs / self.notional
        net_returns = gross_returns - cost_returns
        equity = (1.0 + net_returns).cumprod()

        return BacktestResult(
            returns=net_returns,
            positions=pos,
            spread=spread,
            zscore=sig.zscore,
            costs=costs,
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
        if train_days < 2 or test_days < 1 or step_days < 1:
            raise ValueError("train_days >= 2 and test_days/step_days >= 1 are required")
        if coint_method not in {"engle_granger", "johansen"}:
            raise ValueError("coint_method must be 'engle_granger' or 'johansen'")
        if not 0.0 <= pvalue_threshold <= 1.0:
            raise ValueError("pvalue_threshold must be between 0 and 1")
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
        if leg_y == leg_x:
            raise ValueError("leg_y and leg_x must be different columns")
        if leg_y not in prices or leg_x not in prices:
            raise ValueError("both pair columns must exist in prices")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must use a DatetimeIndex")
        if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
            raise ValueError("prices index must be sorted and unique")

        pair_prices = prices[[leg_y, leg_x]].dropna()
        windows = self._windows(pair_prices.index)
        oos_parts = []
        hedge_log = {}

        for w in windows:
            train = pair_prices.loc[w.train_start:w.train_end]
            test = pair_prices.loc[w.test_start:w.test_end]

            if self.coint_method == "johansen":
                cres = self.coint.johansen(train, (leg_y, leg_x))
            else:
                cres = self.coint.engle_granger(train[leg_y], train[leg_x], (leg_y, leg_x))

            if not cres.is_significant:
                continue

            hedge = cres.hedge_ratio
            hedge_log[w.test_start] = hedge
            bt = self.backtester.run(test[leg_y], test[leg_x], hedge)
            oos_parts.append(bt.returns)

        if not oos_parts:
            return WalkForwardResult(
                windows=windows,
                oos_returns=pd.Series(dtype=float),
                hedge_ratios=pd.Series(dtype=float),
            )

        oos = pd.concat(oos_parts).sort_index()
        oos = oos[~oos.index.duplicated(keep="first")]
        return WalkForwardResult(
            windows=windows,
            oos_returns=oos,
            hedge_ratios=pd.Series(hedge_log, dtype=float),
        )

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    # Absolute change in spread position per day (units of gross notional).
    turnover: pd.Series


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
        trade_start: Optional[pd.Timestamp] = None,
        flat_at_end: bool = False,
    ) -> BacktestResult:
        """Backtest one pair with a fixed hedge ratio.

        Observations before ``trade_start`` are used only to warm up the
        trailing z-score; no positions are taken and no returns are reported
        for them. ``flat_at_end`` closes the final position (charging exit
        cost) so that a window does not carry an uncosted open position.
        """
        if not np.isfinite(hedge_ratio):
            raise ValueError("hedge_ratio must be finite")
        aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if aligned.empty:
            raise ValueError("price series have no aligned observations")
        if (aligned <= 0).any().any():
            raise ValueError("price observations must be positive")

        spread_full = compute_spread(aligned["y"], aligned["x"], hedge_ratio)
        sig = self.signal.generate(spread_full, trade_start=trade_start, flat_at_end=flat_at_end)
        idx = sig.position.index
        if len(idx) == 0:
            raise ValueError("no observations on or after trade_start")
        y_aligned = aligned["y"].loc[idx]
        x_aligned = aligned["x"].loc[idx]
        pos = sig.position.astype(float)

        # Translate spread price changes into return on gross pair notional.
        # The denominator uses the prior close because the prior position earns
        # the move from t-1 to t. The first trading day starts flat.
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
            spread=sig.spread,
            zscore=sig.zscore,
            costs=costs,
            equity=equity,
            turnover=turnover,
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
    # Net returns for every out-of-sample day, including 0.0 on days where
    # the training window rejected the pair (flat, not dropped).
    oos_returns: pd.Series
    hedge_ratios: pd.Series
    oos_positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # One row per window: selected pair (or None), hedge ratio, traded flag.
    window_log: pd.DataFrame = field(default_factory=pd.DataFrame)


class WalkForwardEngine:
    """Rolling train/test evaluation.

    For every window, cointegration testing, pair selection and the hedge
    ratio use only the training slice. The z-score on the first test day is
    computed from the trailing ``lookback`` spread values, which come from the
    end of the training slice (no reset at the window boundary). Positions are
    opened only inside the test slice and closed on its last day. Windows in
    which no pair passes the test contribute flat 0.0 returns.
    """

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
        if coint_method == "johansen":
            CointegrationEngine._validate_significance(pvalue_threshold)
        elif not 0.0 < pvalue_threshold < 1.0:
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

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> None:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must use a DatetimeIndex")
        if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
            raise ValueError("prices index must be sorted and unique")

    def _test_pair(self, train: pd.DataFrame, pair: Tuple[str, str]):
        if self.coint_method == "johansen":
            return self.coint.johansen(train, pair, self.pvalue_threshold)
        return self.coint.engle_granger(
            train[pair[0]], train[pair[1]], pair, significance_level=self.pvalue_threshold
        )

    def _assemble(self, windows, parts, hedge_log, log_rows) -> WalkForwardResult:
        if parts:
            frame = pd.concat(parts).sort_index()
            frame = frame[~frame.index.duplicated(keep="first")]
        else:
            frame = pd.DataFrame(columns=["returns", "positions", "turnover"], dtype=float)
        return WalkForwardResult(
            windows=windows,
            oos_returns=frame["returns"].astype(float),
            hedge_ratios=pd.Series(hedge_log, dtype=float),
            oos_positions=frame["positions"].astype(float),
            oos_turnover=frame["turnover"].astype(float),
            window_log=pd.DataFrame(log_rows),
        )

    def _window_frame(self, test_index, bt: Optional[BacktestResult]) -> pd.DataFrame:
        if bt is None:
            zeros = np.zeros(len(test_index))
            return pd.DataFrame(
                {"returns": zeros, "positions": zeros, "turnover": zeros}, index=test_index
            )
        return pd.DataFrame(
            {"returns": bt.returns, "positions": bt.positions, "turnover": bt.turnover}
        ).reindex(test_index)

    def run(
        self,
        prices: pd.DataFrame,
        leg_y: str,
        leg_x: str,
    ) -> WalkForwardResult:
        """Walk-forward evaluation of one pre-specified pair."""
        if leg_y == leg_x:
            raise ValueError("leg_y and leg_x must be different columns")
        if leg_y not in prices or leg_x not in prices:
            raise ValueError("both pair columns must exist in prices")
        self._validate_prices(prices)

        pair_prices = prices[[leg_y, leg_x]].dropna()
        windows = self._windows(pair_prices.index)
        parts, hedge_log, log_rows = [], {}, []

        for w in windows:
            train = pair_prices.loc[w.train_start:w.train_end]
            history = pair_prices.loc[w.train_start:w.test_end]
            test_index = pair_prices.loc[w.test_start:w.test_end].index
            cres = self._test_pair(train, (leg_y, leg_x))
            bt = None
            if cres.is_significant:
                hedge_log[w.test_start] = cres.hedge_ratio
                bt = self.backtester.run(
                    history[leg_y], history[leg_x], cres.hedge_ratio,
                    trade_start=w.test_start, flat_at_end=True,
                )
            parts.append(self._window_frame(test_index, bt))
            log_rows.append(
                {
                    "test_start": w.test_start,
                    "pair": (leg_y, leg_x) if bt is not None else None,
                    "hedge_ratio": cres.hedge_ratio if bt is not None else np.nan,
                    "traded": bt is not None,
                }
            )
        return self._assemble(windows, parts, hedge_log, log_rows)

    def run_universe(self, prices: pd.DataFrame) -> WalkForwardResult:
        """Walk-forward evaluation with per-window pair selection.

        In each window every pair in ``prices`` is tested on the training
        slice only; the best-ranked significant pair (lowest Engle-Granger
        p-value, or highest Johansen trace statistic) is traded in the
        following test slice with its training hedge ratio. No information
        from the test slice or later is used for selection.
        """
        self._validate_prices(prices)
        if prices.shape[1] < 2:
            raise ValueError("prices must contain at least two columns")
        panel = prices.dropna()
        windows = self._windows(panel.index)
        parts, hedge_log, log_rows = [], {}, []

        for w in windows:
            train = panel.loc[w.train_start:w.train_end]
            history = panel.loc[w.train_start:w.test_end]
            test_index = panel.loc[w.test_start:w.test_end].index
            ranked = self.coint.scan_pairs(
                train, method=self.coint_method, pvalue_threshold=self.pvalue_threshold
            )
            bt = None
            best = ranked[0] if ranked else None
            if best is not None:
                a, b = best.pair
                hedge_log[w.test_start] = best.hedge_ratio
                bt = self.backtester.run(
                    history[a], history[b], best.hedge_ratio,
                    trade_start=w.test_start, flat_at_end=True,
                )
            parts.append(self._window_frame(test_index, bt))
            log_rows.append(
                {
                    "test_start": w.test_start,
                    "pair": best.pair if best is not None else None,
                    "hedge_ratio": best.hedge_ratio if best is not None else np.nan,
                    "traded": bt is not None,
                }
            )
        return self._assemble(windows, parts, hedge_log, log_rows)

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Position(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class SignalParams:
    entry_z: float = 2.0
    exit_z: float = 0.0
    lookback: int = 20


@dataclass
class SignalFrame:
    spread: pd.Series
    zscore: pd.Series
    position: pd.Series


class SignalGenerator:
    def __init__(self, params: Optional[SignalParams] = None):
        self.params = params or SignalParams()

    def zscore(self, spread: pd.Series) -> pd.Series:
        """Trailing z-score: day t uses only spread values up to and including t."""
        mu = spread.rolling(self.params.lookback).mean()
        sigma = spread.rolling(self.params.lookback).std()
        return (spread - mu) / sigma

    def generate(
        self,
        spread: pd.Series,
        trade_start: Optional[pd.Timestamp] = None,
        flat_at_end: bool = False,
    ) -> SignalFrame:
        """Generate positions from a trailing z-score.

        ``spread`` may include history before ``trade_start`` (for example the
        tail of a training window). That history is used only to warm up the
        rolling z-score, so the first trading day already has a full lookback;
        positions are forced flat before ``trade_start``. The returned frame
        covers only ``trade_start`` onward. With ``flat_at_end`` the final
        position is closed so that exit costs are charged inside the window.
        """
        z_full = self.zscore(spread)
        if trade_start is not None:
            spread = spread.loc[trade_start:]
            z = z_full.loc[trade_start:]
        else:
            z = z_full
        pos = pd.Series(0, index=spread.index, dtype=int)
        state = Position.FLAT

        for i, dt in enumerate(spread.index):
            if flat_at_end and i == len(spread.index) - 1:
                pos.iloc[i] = 0
                continue
            if pd.isna(z.iloc[i]):
                state = Position.FLAT
                pos.iloc[i] = 0
                continue
            zi = z.iloc[i]
            if state == Position.FLAT:
                if zi >= self.params.entry_z:
                    state = Position.SHORT
                elif zi <= -self.params.entry_z:
                    state = Position.LONG
            elif state == Position.LONG:
                if zi >= self.params.exit_z:
                    state = Position.FLAT
            elif state == Position.SHORT:
                if zi <= self.params.exit_z:
                    state = Position.FLAT
            pos.iloc[i] = state.value

        return SignalFrame(spread=spread, zscore=z, position=pos)

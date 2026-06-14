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
        mu = spread.rolling(self.params.lookback).mean()
        sigma = spread.rolling(self.params.lookback).std()
        return (spread - mu) / sigma

    def generate(self, spread: pd.Series) -> SignalFrame:
        z = self.zscore(spread)
        pos = pd.Series(0, index=spread.index, dtype=int)
        state = Position.FLAT

        for i, dt in enumerate(spread.index):
            if pd.isna(z.iloc[i]):
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

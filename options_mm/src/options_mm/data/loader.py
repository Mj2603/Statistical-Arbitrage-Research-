"""Simple CSV loader for options tick data (phase 1).

Expected CSV columns:
- timestamp: ISO-8601 or pandas-parsable
- underlying: underlying price (float)
- strike: strike price (float)
- expiry: expiry date (YYYY-MM-DD)
- bid: best bid
- ask: best ask
- iv: implied volatility (decimal, e.g. 0.25)

This loader returns a cleaned pandas DataFrame with computed mid price and time-to-expiry in years.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OptionTick:
    timestamp: pd.Timestamp
    underlying: float
    strike: float
    expiry: pd.Timestamp
    bid: float
    ask: float
    mid: float
    iv: Optional[float]
    time_to_expiry: float


def load_option_ticks_csv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # parse timestamps and expiry
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()

    if tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")

    # drop rows with missing price data
    df = df.dropna(subset=["underlying", "strike", "bid", "ask"])

    # compute mid price
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # compute time to expiry in years
    now = df["timestamp"].max()
    df["time_to_expiry"] = (df["expiry"] - now).dt.total_seconds() / (365.25 * 24 * 3600)
    df["time_to_expiry"] = df["time_to_expiry"].clip(lower=0.0)

    # ensure implied vol is numeric and non-negative
    if "iv" in df.columns:
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        df.loc[df["iv"] < 0, "iv"] = np.nan

    return df

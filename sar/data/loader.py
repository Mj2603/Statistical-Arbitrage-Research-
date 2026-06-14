from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


NSE_BANKING = {
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
}


@dataclass
class PricePanel:
    close: pd.DataFrame
    symbols: List[str]


class DataLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load_csv(self, path: str, date_col: str = "date") -> PricePanel:
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.set_index(date_col).sort_index()
        symbols = list(df.columns)
        return PricePanel(close=df, symbols=symbols)

    def fetch_yfinance(
        self,
        tickers: Dict[str, str],
        start: str,
        end: Optional[str] = None,
    ) -> PricePanel:
        import yfinance as yf

        raw = yf.download(
            list(tickers.values()),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = list(tickers.values())

        inv = {v: k for k, v in tickers.items()}
        close = close.rename(columns=inv)
        close = close.dropna(how="all").ffill().dropna()
        return PricePanel(close=close, symbols=list(close.columns))

    def load_banking_universe(self, start: str, end: Optional[str] = None) -> PricePanel:
        return self.fetch_yfinance(NSE_BANKING, start=start, end=end)

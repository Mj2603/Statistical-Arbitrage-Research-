from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


NSE_BANKING = {
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
}


class DataUnavailableError(RuntimeError):
    """Raised when price data cannot be loaded or is empty."""


@dataclass
class PricePanel:
    close: pd.DataFrame
    symbols: List[str]


class DataLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load_csv(self, path: str, date_col: str = "date") -> PricePanel:
        """Load a wide CSV: one date column plus one close-price column per symbol."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"price CSV not found: {path}") from exc
        if date_col not in df.columns:
            raise DataUnavailableError(
                f"price CSV {path} has no '{date_col}' column; columns are {list(df.columns)}"
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if df.empty or df.shape[1] == 0:
            raise DataUnavailableError(f"price CSV {path} contains no usable price rows")
        return PricePanel(close=df, symbols=list(df.columns))

    @staticmethod
    def save_csv(panel: PricePanel, path: str, date_col: str = "date") -> None:
        """Persist a price panel in the format ``load_csv`` reads."""
        panel.close.rename_axis(date_col).to_csv(path)

    def fetch_yfinance(
        self,
        tickers: Dict[str, str],
        start: str,
        end: Optional[str] = None,
    ) -> PricePanel:
        import yfinance as yf

        try:
            raw = yf.download(
                list(tickers.values()),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # network, DNS, rate limiting, vendor errors
            raise DataUnavailableError(f"yfinance download failed: {exc}") from exc
        if raw is None or raw.empty:
            raise DataUnavailableError(
                "yfinance returned no data for "
                f"{list(tickers.values())} from {start} to {end or 'today'}. "
                "Check network access to Yahoo Finance, the tickers and the date "
                "range, or pass a local file with --csv."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = list(tickers.values())

        inv = {v: k for k, v in tickers.items()}
        close = close.rename(columns=inv)
        missing = [k for k in tickers if k not in close or close[k].isna().all()]
        if missing:
            raise DataUnavailableError(
                f"yfinance returned no prices for {missing}; refusing to continue "
                "with a partial universe"
            )
        close = close.dropna(how="all").ffill().dropna()
        if close.empty:
            raise DataUnavailableError("no dates with prices for every symbol after alignment")
        return PricePanel(close=close, symbols=list(close.columns))

    def load_banking_universe(self, start: str, end: Optional[str] = None) -> PricePanel:
        return self.fetch_yfinance(NSE_BANKING, start=start, end=end)

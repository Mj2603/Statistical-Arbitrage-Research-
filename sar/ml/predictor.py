from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass
class MLPredictionResult:
    model: str
    r2_oos: float
    predictions: pd.Series
    actual: pd.Series


class SpreadPredictor:
    def __init__(self, lookback: int = 5, horizon: int = 1):
        self.lookback = lookback
        self.horizon = horizon

    def _build_features(self, spread: pd.Series) -> pd.DataFrame:
        feats = {}
        for lag in range(1, self.lookback + 1):
            feats[f"lag_{lag}"] = spread.shift(lag)
        feats["zscore"] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        feats["vol"] = spread.diff().rolling(20).std()
        return pd.DataFrame(feats)

    def _build_xy(self, spread: pd.Series):
        X = self._build_features(spread)
        y = spread.shift(-self.horizon) - spread
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        return data[X.columns], data["target"]

    def fit_linear(self, spread: pd.Series, train_frac: float = 0.7) -> MLPredictionResult:
        X, y = self._build_xy(spread)
        split = int(len(X) * train_frac)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_test), index=X_test.index)
        r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0
        return MLPredictionResult(model="linear", r2_oos=r2, predictions=pred, actual=y_test)

    def fit_xgboost(self, spread: pd.Series, train_frac: float = 0.7) -> MLPredictionResult:
        import xgboost as xgb

        X, y = self._build_xy(spread)
        split = int(len(X) * train_frac)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_test), index=X_test.index)
        r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0
        return MLPredictionResult(model="xgboost", r2_oos=r2, predictions=pred, actual=y_test)

    def fit(
        self,
        spread: pd.Series,
        method: Literal["linear", "xgboost"] = "linear",
        train_frac: float = 0.7,
    ) -> MLPredictionResult:
        if method == "xgboost":
            return self.fit_xgboost(spread, train_frac)
        return self.fit_linear(spread, train_frac)

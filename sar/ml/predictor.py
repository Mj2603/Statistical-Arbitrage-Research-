"""ML helpers for spread prediction used in the research pipeline.

This module contains a small, transparent feature-based regressor used for
experimental spread-delta prediction. The implementation is intentionally
lightweight to make it easy to inspect, test, and reproduce.

Design notes (author):
- Features: lagged spread values, a 20-day rolling z-score, and a 20-day
  rolling volatility of spread changes. These capture short-term momentum,
  mean-reversion (z-score) and recent volatility.
- Models: simple linear regression (baseline) and an XGBoost regressor
  (non-linear baseline). Hyperparameters are conservative to avoid overfit
  on small windows (n_estimators=100, max_depth=3).
- Evaluation: out-of-sample R^2 computed on the test split; production use
  would require more robust CV and risk-aware metrics.

The comments below explain why choices were made; they also serve as a
readable log of the author's thought process for recruiters or reviewers.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass
class MLPredictionResult:
    """Simple container for ML prediction outputs.

    Fields:
    - model: a short string identifying the model used (e.g., "linear").
    - r2_oos: out-of-sample R^2 computed on the test partition.
    - predictions: pd.Series of model predictions (aligned to the test index).
    - actual: pd.Series of actual target values used for evaluation.
    """


class SpreadPredictor:
    """Feature-based predictor for short-horizon spread changes.

    Arguments:
    - lookback: number of lag features to create (lag_1 .. lag_N).
    - horizon: prediction horizon in periods (default 1).

    The implementation favors clarity and reproducibility over performance.
    """

    def __init__(self, lookback: int = 5, horizon: int = 1):
        self.lookback = lookback
        self.horizon = horizon

    def _build_features(self, spread: pd.Series) -> pd.DataFrame:
        """Create a small set of interpretable features from the spread.

        Features:
        - lag_k: spread shifted by k periods (captures short-term autocorrelation)
        - zscore: (spread - 20d mean) / 20d std (captures mean-reversion)
        - vol: rolling std of spread diff over 20 days (recent volatility)
        """
        feats = {}
        for lag in range(1, self.lookback + 1):
            feats[f"lag_{lag}"] = spread.shift(lag)
        feats["zscore"] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        feats["vol"] = spread.diff().rolling(20).std()
        return pd.DataFrame(feats)

    def _build_xy(self, spread: pd.Series):
        """Return (X, y) for supervised learning.

        The target is the spread change at `horizon` periods ahead (spread_{t+h} - spread_t).
        Rows with NaNs due to shifting/rolling are dropped so resulting X and y align.
        """
        X = self._build_features(spread)
        y = spread.shift(-self.horizon) - spread
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        return data[X.columns], data["target"]

    def fit_linear(self, spread: pd.Series, train_frac: float = 0.7) -> MLPredictionResult:
        """Fit a linear regression baseline and return predictions + OOS R^2."""
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
        """Fit an XGBoost regressor as a non-linear baseline.

        XGBoost is imported locally so the dependency is optional for users who
        only want to run the linear baseline or the tests that skip XGBoost.
        """
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

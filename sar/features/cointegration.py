from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class CointegrationResult:
    pair: Tuple[str, str]
    hedge_ratio: float
    method: str
    is_significant: bool
    pvalue: Optional[float] = None
    test_statistic: Optional[float] = None
    critical_value: Optional[float] = None
    significance_level: Optional[float] = None


def _aligned_pair(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        raise TypeError("y and x must be pandas Series")
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("at least three aligned observations are required")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("price series must contain only finite values")
    if np.ptp(values[:, 1]) == 0:
        raise ValueError("x price series must have non-zero variance")
    return aligned["y"], aligned["x"]


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    y_aligned, x_aligned = _aligned_pair(y, x)
    x_arr = np.column_stack([np.ones(len(x_aligned)), x_aligned.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(x_arr, y_aligned.to_numpy(dtype=float), rcond=None)[0]
    if not np.isfinite(beta[1]):
        raise ValueError("hedge ratio estimation produced a non-finite value")
    return float(beta[1])


def compute_spread(y: pd.Series, x: pd.Series, hedge_ratio: float) -> pd.Series:
    if not np.isfinite(hedge_ratio):
        raise ValueError("hedge_ratio must be finite")
    y_aligned, x_aligned = _aligned_pair(y, x)
    return y_aligned - hedge_ratio * x_aligned


class CointegrationEngine:
    @staticmethod
    def _validate_significance(significance_level: float) -> None:
        if significance_level not in {0.01, 0.05, 0.10}:
            raise ValueError("significance_level must be one of 0.01, 0.05, or 0.10")

    def engle_granger(
        self,
        y: pd.Series,
        x: pd.Series,
        pair: Tuple[str, str],
        significance_level: float = 0.05,
    ) -> CointegrationResult:
        if not 0.0 < significance_level < 1.0:
            raise ValueError("significance_level must be between 0 and 1")
        y_aligned, x_aligned = _aligned_pair(y, x)
        hedge = ols_hedge_ratio(y_aligned, x_aligned)
        statistic, pvalue, _ = coint(y_aligned, x_aligned)
        if not np.isfinite(statistic) or not np.isfinite(pvalue):
            raise ValueError("Engle-Granger test returned a non-finite result")
        return CointegrationResult(
            pair=pair,
            hedge_ratio=hedge,
            method="engle_granger",
            is_significant=bool(pvalue <= significance_level),
            pvalue=float(pvalue),
            test_statistic=float(statistic),
            significance_level=significance_level,
        )

    def johansen(
        self,
        prices: pd.DataFrame,
        pair: Tuple[str, str],
        significance_level: float = 0.05,
    ) -> CointegrationResult:
        self._validate_significance(significance_level)
        if pair[0] == pair[1] or pair[0] not in prices or pair[1] not in prices:
            raise ValueError("pair must name two distinct columns in prices")
        sub = prices.loc[:, list(pair)].dropna()
        values = sub.to_numpy(dtype=float)
        if len(sub) < 3 or not np.isfinite(values).all():
            raise ValueError("Johansen test requires at least three finite aligned observations")

        result = coint_johansen(sub, det_order=0, k_ar_diff=1)
        eigenvector = result.evec[:, 0]
        if not np.isfinite(eigenvector).all() or abs(eigenvector[0]) <= np.finfo(float).eps:
            raise ValueError("Johansen eigenvector cannot produce a finite hedge ratio")
        hedge = -eigenvector[1] / eigenvector[0]

        critical_column = {0.10: 0, 0.05: 1, 0.01: 2}[significance_level]
        statistic = float(result.lr1[0])
        critical_value = float(result.cvt[0, critical_column])
        if not np.isfinite(statistic) or not np.isfinite(critical_value):
            raise ValueError("Johansen test returned a non-finite statistic")

        return CointegrationResult(
            pair=pair,
            hedge_ratio=float(hedge),
            method="johansen",
            is_significant=bool(statistic > critical_value),
            pvalue=None,
            test_statistic=statistic,
            critical_value=critical_value,
            significance_level=significance_level,
        )

    def scan_pairs(
        self,
        prices: pd.DataFrame,
        method: str = "engle_granger",
        pvalue_threshold: float = 0.05,
    ) -> List[CointegrationResult]:
        if method not in {"engle_granger", "johansen"}:
            raise ValueError("method must be 'engle_granger' or 'johansen'")
        if method == "johansen":
            self._validate_significance(pvalue_threshold)
        elif not 0.0 < pvalue_threshold < 1.0:
            raise ValueError("pvalue_threshold must be between 0 and 1")

        results = []
        for a, b in combinations(prices.columns, 2):
            sub = prices[[a, b]].dropna()
            if len(sub) < 60:
                continue
            if method == "johansen":
                result = self.johansen(sub, (a, b), pvalue_threshold)
            else:
                result = self.engle_granger(
                    sub[a], sub[b], (a, b), significance_level=pvalue_threshold
                )
            if result.is_significant:
                results.append(result)

        if method == "johansen":
            return sorted(results, key=lambda item: item.test_statistic or float("-inf"), reverse=True)
        return sorted(results, key=lambda item: item.pvalue if item.pvalue is not None else float("inf"))

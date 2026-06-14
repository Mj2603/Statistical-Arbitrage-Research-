from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class CointegrationResult:
    pair: Tuple[str, str]
    hedge_ratio: float
    pvalue: float
    method: str


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    x_arr = np.column_stack([np.ones(len(x)), x.values])
    beta = np.linalg.lstsq(x_arr, y.values, rcond=None)[0]
    return float(beta[1])


def compute_spread(y: pd.Series, x: pd.Series, hedge_ratio: float) -> pd.Series:
    return y - hedge_ratio * x


class CointegrationEngine:
    def engle_granger(self, y: pd.Series, x: pd.Series, pair: Tuple[str, str]) -> CointegrationResult:
        hedge = ols_hedge_ratio(y, x)
        spread = compute_spread(y, x, hedge)
        _, pvalue, _ = coint(y, x)
        return CointegrationResult(pair=pair, hedge_ratio=hedge, pvalue=float(pvalue), method="engle_granger")

    def johansen(self, prices: pd.DataFrame, pair: Tuple[str, str]) -> CointegrationResult:
        sub = prices[list(pair)].dropna()
        result = coint_johansen(sub, det_order=0, k_ar_diff=1)
        eigvec = result.evec[:, 0]
        hedge = -eigvec[1] / eigvec[0]
        spread = compute_spread(sub[pair[0]], sub[pair[1]], hedge)
        # trace stat vs critical; use rank-0 rejection as cointegration proxy
        trace = result.lr1[0]
        crit = result.cvt[0, 1]  # 95%
        pvalue = 0.01 if trace > crit else 0.10
        return CointegrationResult(pair=pair, hedge_ratio=float(hedge), pvalue=pvalue, method="johansen")

    def scan_pairs(
        self,
        prices: pd.DataFrame,
        method: str = "engle_granger",
        pvalue_threshold: float = 0.05,
    ) -> List[CointegrationResult]:
        results = []
        for a, b in combinations(prices.columns, 2):
            sub = prices[[a, b]].dropna()
            if len(sub) < 60:
                continue
            if method == "johansen":
                res = self.johansen(sub, (a, b))
            else:
                res = self.engle_granger(sub[a], sub[b], (a, b))
            if res.pvalue <= pvalue_threshold:
                results.append(res)
        return sorted(results, key=lambda r: r.pvalue)

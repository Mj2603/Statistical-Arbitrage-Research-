from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class FactorExposure:
    beta: float
    r_squared: float


class RiskAttribution:
    def rolling_beta(
        self,
        returns: pd.Series,
        factor: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        aligned = pd.concat([returns, factor], axis=1).dropna()
        aligned.columns = ["ret", "fac"]
        betas = []
        idx = []
        for i in range(window, len(aligned)):
            chunk = aligned.iloc[i - window : i]
            x = chunk["fac"].values
            y = chunk["ret"].values
            if chunk["fac"].std() == 0:
                betas.append(0.0)
            else:
                betas.append(float(np.cov(y, x)[0, 1] / np.var(x)))
            idx.append(aligned.index[i])
        return pd.Series(betas, index=idx)

    def factor_exposure(self, returns: pd.Series, factor: pd.Series) -> FactorExposure:
        aligned = pd.concat([returns, factor], axis=1).dropna()
        if len(aligned) < 10:
            return FactorExposure(beta=0.0, r_squared=0.0)
        x = aligned.iloc[:, 1].values
        y = aligned.iloc[:, 0].values
        if np.var(x) == 0:
            return FactorExposure(beta=0.0, r_squared=0.0)
        beta = float(np.cov(y, x)[0, 1] / np.var(x))
        corr = np.corrcoef(y, x)[0, 1]
        return FactorExposure(beta=beta, r_squared=float(corr ** 2))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cointegrated_prices():
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = 100 + np.cumsum(rng.normal(0, 0.5, n))
    spread = rng.normal(0, 1, n)
    y = 1.2 * x + spread
    return pd.DataFrame({"LEG_A": y, "LEG_B": x}, index=dates)


@pytest.fixture
def random_walk_prices():
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.bdate_range("2021-01-01", periods=n)
    a = 100 + np.cumsum(rng.normal(0, 1, n))
    b = 50 + np.cumsum(rng.normal(0, 1.2, n))
    return pd.DataFrame({"LEG_A": a, "LEG_B": b}, index=dates)

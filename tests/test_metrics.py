import numpy as np
import pandas as pd

from sar.metrics.performance import PerformanceAttribution


def test_sharpe_sign():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.001, 0.01, 252))
    rep = PerformanceAttribution().compute(ret)
    assert rep.sharpe > 0


def test_drawdown_negative():
    ret = pd.Series([0.01, -0.05, 0.02, -0.03])
    rep = PerformanceAttribution(periods=252).compute(ret)
    assert rep.max_drawdown <= 0


def test_hit_rate_bounds():
    ret = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    rep = PerformanceAttribution().compute(ret)
    assert 0 <= rep.hit_rate <= 1

import pytest

from sar.features.cointegration import compute_spread, ols_hedge_ratio
from sar.ml.predictor import SpreadPredictor


def test_linear_runs(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    spread = compute_spread(y, x, ols_hedge_ratio(y, x))
    res = SpreadPredictor(lookback=3).fit_linear(spread, train_frac=0.7)
    assert len(res.predictions) > 0


def test_xgboost_runs(cointegrated_prices):
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        pytest.skip(f"xgboost unavailable: {exc}")

    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    spread = compute_spread(y, x, ols_hedge_ratio(y, x))
    res = SpreadPredictor(lookback=3).fit_xgboost(spread, train_frac=0.7)
    assert res.model == "xgboost"

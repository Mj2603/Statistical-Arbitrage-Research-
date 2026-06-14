from sar.features.cointegration import CointegrationEngine, compute_spread, ols_hedge_ratio


def test_hedge_ratio(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    beta = ols_hedge_ratio(y, x)
    assert 1.0 < beta < 1.4


def test_engle_granger(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    res = CointegrationEngine().engle_granger(y, x, ("LEG_A", "LEG_B"))
    assert res.pvalue < 0.05


def test_johansen(cointegrated_prices):
    res = CointegrationEngine().johansen(cointegrated_prices, ("LEG_A", "LEG_B"))
    assert res.hedge_ratio != 0


def test_spread_narrower_than_legs(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    beta = ols_hedge_ratio(y, x)
    spread = compute_spread(y, x, beta)
    assert spread.std() < y.std()

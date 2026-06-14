from sar.backtest.costs import TransactionCostModel
from sar.backtest.engine import PairsBacktester
from sar.features.cointegration import ols_hedge_ratio


def test_costs_positive():
    m = TransactionCostModel()
    assert m.round_trip_cost(1_000_000) > 0


def test_backtest_runs(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    beta = ols_hedge_ratio(y, x)
    bt = PairsBacktester().run(y, x, beta)
    assert len(bt.returns) == len(y)
    assert bt.costs.sum() >= 0


def test_costs_reduce_pnl(cointegrated_prices):
    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    beta = ols_hedge_ratio(y, x)
    cheap = PairsBacktester(costs=TransactionCostModel(brokerage_bps=0, half_spread_bps=0, slippage_bps=0))
    costly = PairsBacktester(costs=TransactionCostModel(brokerage_bps=10, half_spread_bps=10, slippage_bps=10))
    eq_cheap = cheap.run(y, x, beta).equity.iloc[-1]
    eq_costly = costly.run(y, x, beta).equity.iloc[-1]
    assert eq_costly <= eq_cheap

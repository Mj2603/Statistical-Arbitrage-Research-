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


def test_returns_are_scaled_to_notional_and_costs_keep_currency_units():
    import pandas as pd
    from sar.signals.zscore import SignalParams

    index = pd.date_range("2025-01-01", periods=6, freq="D")
    y = pd.Series([100, 100, 102, 100, 97, 100], index=index, dtype=float)
    x = pd.Series([100, 100, 100, 100, 100, 100], index=index, dtype=float)
    result = PairsBacktester(
        signal_params=SignalParams(entry_z=0.5, exit_z=0.0, lookback=2),
        notional=10_000,
    ).run(y, x, hedge_ratio=1.0)

    assert result.returns.abs().max() < 0.1
    assert result.costs.max() > 0.0
    assert result.equity.iloc[0] == 1.0


def test_backtester_rejects_invalid_notional_and_prices(cointegrated_prices):
    import numpy as np
    import pandas as pd
    import pytest

    with pytest.raises(ValueError, match="notional"):
        PairsBacktester(notional=0.0)
    y = cointegrated_prices["LEG_A"].copy()
    x = cointegrated_prices["LEG_B"].copy()
    y.iloc[0] = np.nan
    x.iloc[0] = np.nan
    with pytest.raises(ValueError, match="aligned"):
        PairsBacktester().run(y.iloc[:1], x.iloc[:1], 1.0)

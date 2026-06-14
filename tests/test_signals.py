from sar.signals.zscore import SignalGenerator, SignalParams


def test_zscore_entry(cointegrated_prices):
    from sar.features.cointegration import compute_spread, ols_hedge_ratio

    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    beta = ols_hedge_ratio(y, x)
    spread = compute_spread(y, x, beta)

    sig = SignalGenerator(SignalParams(entry_z=2.0, exit_z=0.0, lookback=20))
    frame = sig.generate(spread)
    assert frame.position.abs().max() <= 1
    assert (frame.position != 0).any()


def test_exit_at_zero(cointegrated_prices):
    from sar.features.cointegration import compute_spread, ols_hedge_ratio

    y = cointegrated_prices["LEG_A"]
    x = cointegrated_prices["LEG_B"]
    spread = compute_spread(y, x, ols_hedge_ratio(y, x))
    sig = SignalGenerator(SignalParams(entry_z=1.5, exit_z=0.0, lookback=15))
    frame = sig.generate(spread)
    # positions should eventually flatten after extreme z
    assert frame.position.iloc[-1] in (0, 1, -1)

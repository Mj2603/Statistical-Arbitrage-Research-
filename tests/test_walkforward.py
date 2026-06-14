from sar.backtest.engine import WalkForwardEngine


def test_walk_forward_oos(cointegrated_prices):
    wf = WalkForwardEngine(train_days=120, test_days=40, step_days=40)
    res = wf.run(cointegrated_prices, "LEG_A", "LEG_B")
    assert len(res.windows) > 0
    assert len(res.oos_returns) > 0


def test_no_lookahead(cointegrated_prices):
    wf = WalkForwardEngine(train_days=100, test_days=30, step_days=30)
    res = wf.run(cointegrated_prices, "LEG_A", "LEG_B")
    for w in res.windows:
        assert w.train_end < w.test_start

"""Regression tests for walk-forward correctness fixes."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar.backtest.costs import TransactionCostModel
from sar.backtest.engine import PairsBacktester, WalkForwardEngine
from sar.data.loader import DataLoader, DataUnavailableError, PricePanel
from sar.features.cointegration import CointegrationEngine, johansen_trace, ols_hedge_ratio
from sar.metrics.performance import PerformanceAttribution, turnover
from sar.reporting import summarize
from sar.signals.zscore import SignalGenerator, SignalParams

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _universe(seed=3, n=700):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    x = 200 + np.cumsum(rng.normal(0, 1, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.85 * s[t - 1] + rng.normal(0, 1.5)
    return pd.DataFrame(
        {
            "A": 1.3 * x + 20 + s,
            "B": x,
            "C": 300 + np.cumsum(rng.normal(0, 1, n)),
            "D": 250 + np.cumsum(rng.normal(0, 1, n)),
        },
        index=dates,
    )


def _perturb_after(prices, cutoff, seed=99):
    rng = np.random.default_rng(seed)
    out = prices.copy()
    future = out.index > cutoff
    out.loc[future] = out.loc[future] * rng.uniform(0.7, 1.3, size=out.loc[future].shape)
    return out


# --- look-ahead -----------------------------------------------------------


@pytest.mark.parametrize("cut_pos", [300, 420, 555])
def test_fixed_pair_past_results_ignore_future_prices(cut_pos):
    prices = _universe()
    cutoff = prices.index[cut_pos]
    engine = WalkForwardEngine(train_days=200, test_days=50, step_days=50)
    base = engine.run(prices, "A", "B")
    pert = engine.run(_perturb_after(prices, cutoff), "A", "B")
    past = base.oos_returns.index <= cutoff
    assert past.sum() > 0
    pd.testing.assert_series_equal(base.oos_returns[past], pert.oos_returns[past])
    pd.testing.assert_series_equal(base.oos_positions[past], pert.oos_positions[past])
    # The perturbation must actually matter later, or the test proves nothing.
    assert not base.oos_returns[~past].equals(pert.oos_returns[~past])


@pytest.mark.parametrize("cut_pos", [300, 480])
def test_universe_selection_ignores_future_prices(cut_pos):
    prices = _universe()
    cutoff = prices.index[cut_pos]
    engine = WalkForwardEngine(train_days=200, test_days=50, step_days=50)
    base = engine.run_universe(prices)
    pert = engine.run_universe(_perturb_after(prices, cutoff))
    past = base.oos_returns.index <= cutoff
    pd.testing.assert_series_equal(base.oos_returns[past], pert.oos_returns[past])
    pd.testing.assert_series_equal(base.oos_positions[past], pert.oos_positions[past])
    started = base.window_log["test_start"] <= cutoff
    assert list(base.window_log.loc[started, "pair"]) == list(pert.window_log.loc[started, "pair"])


def test_universe_selects_true_pair_from_training_data():
    res = WalkForwardEngine(train_days=200, test_days=50, step_days=50).run_universe(_universe())
    chosen = [p for p in res.window_log["pair"] if p is not None]
    assert chosen and all(set(p) == {"A", "B"} for p in chosen)


def test_zscore_is_trailing():
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.normal(size=200), index=pd.bdate_range("2020-01-01", periods=200))
    gen = SignalGenerator(SignalParams(lookback=20))
    base = gen.generate(spread)
    changed = spread.copy()
    changed.iloc[120:] += 50.0
    pert = gen.generate(changed)
    pd.testing.assert_series_equal(base.zscore.iloc[:120], pert.zscore.iloc[:120])
    pd.testing.assert_series_equal(base.position.iloc[:120], pert.position.iloc[:120])


def test_ml_hedge_ratio_uses_only_ml_training_split():
    script = _load_script("run_banking_pairs")
    prices = _universe()
    base = script.ml_baseline(prices, "A", "B", use_xgboost=False)
    pert = script.ml_baseline(_perturb_after(prices, base["train_end"]), "A", "B", use_xgboost=False)
    assert base["beta"] == pert["beta"]
    assert base["train_end"] == pert["train_end"]


# --- z-score warm-up across window boundaries -------------------------------


def test_zscore_defined_on_first_test_day(cointegrated_prices):
    y, x = cointegrated_prices["LEG_A"], cointegrated_prices["LEG_B"]
    trade_start = cointegrated_prices.index[150]
    bt = PairsBacktester(SignalParams(lookback=20)).run(
        y, x, ols_hedge_ratio(y.iloc[:150], x.iloc[:150]), trade_start=trade_start
    )
    assert bt.returns.index[0] == trade_start
    assert np.isfinite(bt.zscore.iloc[0])
    assert bt.zscore.notna().all()


def test_walk_forward_can_trade_in_first_lookback_days():
    prices = _universe()
    res = WalkForwardEngine(train_days=200, test_days=50, step_days=50).run(prices, "A", "B")
    early = []
    for w in res.windows:
        test_pos = res.oos_positions.loc[w.test_start:w.test_end]
        early.append((test_pos.iloc[:19] != 0).any())
    assert any(early)


def test_positions_flat_on_last_test_day():
    prices = _universe()
    res = WalkForwardEngine(train_days=200, test_days=50, step_days=50).run(prices, "A", "B")
    for w in res.windows:
        assert res.oos_positions.loc[w.test_end] == 0


# --- skipped windows count as flat days ------------------------------------


def test_skipped_windows_are_flat_days(random_walk_prices):
    engine = WalkForwardEngine(train_days=100, test_days=40, step_days=40)
    res = engine.run(random_walk_prices, "LEG_A", "LEG_B")
    assert len(res.oos_returns) == len(res.windows) * 40
    skipped = res.window_log[~res.window_log["traded"]]
    assert len(skipped) > 0
    for start in skipped["test_start"]:
        w = next(w for w in res.windows if w.test_start == start)
        assert (res.oos_returns.loc[w.test_start:w.test_end] == 0).all()


def test_sharpe_includes_flat_days():
    prices = _universe()
    engine = WalkForwardEngine(train_days=200, test_days=50, step_days=50)
    res = engine.run(prices, "A", "C")  # mostly not cointegrated
    assert len(res.oos_returns) == len(res.windows) * 50
    perf = PerformanceAttribution().compute(res.oos_returns, res.oos_positions)
    assert perf.n_days == len(res.windows) * 50


# --- turnover --------------------------------------------------------------


def test_turnover_counts_initial_entry():
    assert turnover(pd.Series([1.0, 1.0, 0.0, -1.0])) == pytest.approx(3 / 4)


def test_turnover_reported_from_walk_forward_positions():
    prices = _universe()
    res = WalkForwardEngine(train_days=200, test_days=50, step_days=50).run(prices, "A", "B")
    stats = summarize(res)
    assert stats["turnover_per_day"] > 0
    assert stats["turnover_per_day"] == pytest.approx(res.oos_turnover.mean())


def test_turnover_is_nan_without_positions():
    perf = PerformanceAttribution().compute(pd.Series([0.01, -0.01, 0.0]))
    assert np.isnan(perf.turnover)


def test_costs_scale_with_turnover():
    prices = _universe()
    kw = dict(train_days=200, test_days=50, step_days=50)
    free = WalkForwardEngine(**kw, costs=TransactionCostModel(0, 0, 0)).run(prices, "A", "B")
    paid = WalkForwardEngine(**kw, costs=TransactionCostModel(10, 0, 0)).run(prices, "A", "B")
    drag = (free.oos_returns - paid.oos_returns).sum()
    assert drag == pytest.approx(free.oos_turnover.sum() * 10 / 10_000)


# --- Johansen ---------------------------------------------------------------


def test_johansen_emits_no_complex_warning(cointegrated_prices):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = CointegrationEngine().johansen(cointegrated_prices, ("LEG_A", "LEG_B"))
    assert isinstance(res.hedge_ratio, float)
    assert res.is_significant


def test_johansen_matches_statsmodels(cointegrated_prices, random_walk_prices):
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    for frame in (cointegrated_prices, random_walk_prices):
        values = frame.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = coint_johansen(values, det_order=0, k_ar_diff=1)
        ours = johansen_trace(values, k_ar_diff=1)
        np.testing.assert_allclose(ours.trace_stats, np.real(ref.lr1), rtol=1e-8)
        np.testing.assert_allclose(ours.critical_values, ref.cvt)
        ref_hedge = np.real(-ref.evec[1, 0] / ref.evec[0, 0])
        assert -ours.eigenvectors[1, 0] / ours.eigenvectors[0, 0] == pytest.approx(ref_hedge)


# --- data loading -----------------------------------------------------------


def test_empty_download_raises(monkeypatch):
    yf = pytest.importorskip("yfinance")
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataUnavailableError, match="no data"):
        DataLoader().load_banking_universe(start="2019-01-01")


def test_failed_download_raises(monkeypatch):
    yf = pytest.importorskip("yfinance")

    def boom(*a, **k):
        raise ConnectionError("blocked")

    monkeypatch.setattr(yf, "download", boom)
    with pytest.raises(DataUnavailableError, match="blocked"):
        DataLoader().load_banking_universe(start="2019-01-01")


def test_csv_round_trip(tmp_path, cointegrated_prices):
    path = tmp_path / "prices.csv"
    DataLoader.save_csv(PricePanel(cointegrated_prices, list(cointegrated_prices)), path)
    loaded = DataLoader().load_csv(path)
    pd.testing.assert_frame_equal(
        loaded.close, cointegrated_prices.rename_axis("date"), check_freq=False
    )


def test_missing_csv_raises(tmp_path):
    with pytest.raises(DataUnavailableError, match="not found"):
        DataLoader().load_csv(tmp_path / "nope.csv")


# --- scripts ---------------------------------------------------------------


def test_banking_script_runs_from_csv(tmp_path):
    prices = _universe(n=600).rename(
        columns={"A": "HDFCBANK", "B": "ICICIBANK", "C": "AXISBANK", "D": "KOTAKBANK"}
    )
    csv = tmp_path / "prices.csv"
    DataLoader.save_csv(PricePanel(prices, list(prices)), csv)
    out = tmp_path / "report.md"
    script = _load_script("run_banking_pairs")
    code = script.main(
        ["--csv", str(csv), "--start", "2018-01-01", "--output", str(out), "--cost-bps", "0", "10",
         "--no-ml"]
    )
    assert code == 0
    text = out.read_text()
    assert "turnover_per_year" in text and "HDFCBANK/ICICIBANK" in text


def test_banking_script_exits_cleanly_without_data(tmp_path, capsys):
    script = _load_script("run_banking_pairs")
    code = script.main(["--csv", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "r.md")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_synthetic_experiment_is_deterministic():
    script = _load_script("run_synthetic_experiment")
    a, _ = script.simulate(seed=5, n=300)
    b, _ = script.simulate(seed=5, n=300)
    pd.testing.assert_frame_equal(a, b)

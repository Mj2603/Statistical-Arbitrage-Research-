#!/usr/bin/env python3
"""Deterministic SYNTHETIC walk-forward experiment.

Simulates seeded price paths for

- cointegrated pairs: y = beta * x + s, with x a random walk and s a stationary
  AR(1) spread (several mean-reversion speeds);
- a regime-break pair: cointegrated for the first half, then the spread turns
  into a random walk;
- non-cointegrated controls: two independent random walks;

runs the same walk-forward engine used for real data (252/63/63, Engle-Granger
at 5%, training-window OLS hedge ratio, trailing 20-day z-score) at several cost
levels and writes ``results/synthetic_walkforward.md`` and
``results/synthetic_walkforward.csv``.

The data are simulated. The output checks that the pipeline behaves as
expected (makes money on mean-reverting spreads, does not on random walks,
and reacts to costs); it says nothing about real markets.
"""

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from sar.reporting import DEFAULT_COST_GRID_BPS, markdown_table, run_cost_grid

SEED = 20240601
N_DAYS = 2016  # 8 years of business days


def _random_walk(rng, n, start, vol):
    return start + np.cumsum(rng.normal(0.0, vol, n))


def _ar1(rng, n, phi, sd):
    # sd is the stationary standard deviation of the AR(1) process.
    eps_sd = sd * np.sqrt(1.0 - phi**2)
    s = np.empty(n)
    s[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, eps_sd)
    return s


def simulate(seed: int = SEED, n: int = N_DAYS):
    """Return (prices, specs) for a deterministic synthetic universe."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    cols, specs = {}, []

    for i, (phi, beta) in enumerate([(0.80, 1.0), (0.90, 1.2), (0.95, 0.8), (0.98, 1.5)]):
        x = _random_walk(rng, n, 400.0, 1.0)
        y = beta * x + 50.0 + _ar1(rng, n, phi, 2.0)
        name = f"COINT_{i + 1}"
        cols[f"{name}_Y"], cols[f"{name}_X"] = y, x
        half_life = np.log(0.5) / np.log(phi)
        specs.append((name, "cointegrated", f"AR(1) spread, phi={phi}, half-life {half_life:.1f}d, beta={beta}"))

    x = _random_walk(rng, n, 400.0, 1.0)
    half = n // 2
    s = np.concatenate([_ar1(rng, half, 0.90, 2.0), np.zeros(n - half)])
    s[half:] = s[half - 1] + np.cumsum(rng.normal(0.0, 0.9, n - half))
    cols["BREAK_1_Y"], cols["BREAK_1_X"] = 1.1 * x + 50.0 + s, x
    specs.append(("BREAK_1", "regime break", "AR(1) phi=0.90 for first half, random-walk spread after"))

    for i in range(4):
        cols[f"CONTROL_{i + 1}_Y"] = _random_walk(rng, n, 400.0, 1.0)
        cols[f"CONTROL_{i + 1}_X"] = _random_walk(rng, n, 400.0, 1.0)
        specs.append((f"CONTROL_{i + 1}", "not cointegrated", "two independent random walks"))

    prices = pd.DataFrame(cols, index=dates)
    if (prices <= 0).any().any():
        raise RuntimeError("simulation produced non-positive prices; change the seed")
    return prices, specs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--cost-bps", type=float, nargs="+", default=list(DEFAULT_COST_GRID_BPS))
    args = p.parse_args(argv)

    prices, specs = simulate(args.seed)
    ratios = []
    for name, kind, _ in specs:
        if kind == "cointegrated":
            y, x = prices[f"{name}_Y"], prices[f"{name}_X"]
            beta = np.polyfit(x, y, 1)[0]
            ratios.append((y - beta * x).std() / (y.abs() + abs(beta) * x.abs()).mean())
    spread_to_notional = float(np.mean(ratios))
    rows = []
    for name, kind, desc in specs:
        pair_rows = run_cost_grid(prices, args.cost_bps, pair=(f"{name}_Y", f"{name}_X"))
        for r in pair_rows:
            rows.append({"pair": name, "type": kind, **r})
    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "synthetic_walkforward.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    cols = [
        "pair", "type", "oos_days", "windows_traded", "windows", "sharpe", "cagr",
        "max_drawdown", "turnover_per_year", "exposure", "hit_rate",
    ]
    base_bps = 10.0 if 10.0 in args.cost_bps else args.cost_bps[0]
    at_base = df[df["cost_bps"] == base_bps].to_dict("records")

    groups = []
    for kind in ["cointegrated", "regime break", "not cointegrated"]:
        for bps in args.cost_bps:
            sub = df[(df["type"] == kind) & (df["cost_bps"] == bps)]
            groups.append(
                {
                    "type": kind,
                    "cost_bps": bps,
                    "mean_sharpe": sub["sharpe"].mean(),
                    "mean_cagr": sub["cagr"].mean(),
                    "worst_max_drawdown": sub["max_drawdown"].min(),
                    "windows_traded_pct": sub["windows_traded"].sum() / sub["windows"].sum(),
                }
            )
    group_fmt = [
        {
            **g,
            "mean_sharpe": f"{g['mean_sharpe']:.2f}",
            "mean_cagr": f"{g['mean_cagr']:.2%}",
            "worst_max_drawdown": f"{g['worst_max_drawdown']:.2%}",
            "windows_traded_pct": f"{g['windows_traded_pct']:.0%}",
        }
        for g in groups
    ]

    def g(kind, bps):
        return next(x for x in groups if x["type"] == kind and x["cost_bps"] == bps)

    zero_bps = min(args.cost_bps)
    summary = [
        f"- Gross of costs ({zero_bps:g} bps), the cointegrated pairs average an OOS Sharpe of "
        f"{g('cointegrated', zero_bps)['mean_sharpe']:.2f} versus "
        f"{g('not cointegrated', zero_bps)['mean_sharpe']:.2f} for the random-walk controls.",
        f"- At {base_bps:g} bps per unit position change the cointegrated average is "
        f"{g('cointegrated', base_bps)['mean_sharpe']:.2f} (controls "
        f"{g('not cointegrated', base_bps)['mean_sharpe']:.2f}). With this calibration the "
        f"simulated spread's standard deviation is only {spread_to_notional:.2%} of gross pair "
        "notional, so the expected capture of a 2-sigma entry is of the same order as the "
        "round-trip cost: the net result is cost-dominated.",
        f"- The 5% Engle-Granger screen admitted {g('not cointegrated', base_bps)['windows_traded_pct']:.0%} "
        "of control windows (false positives) and "
        f"{g('cointegrated', base_bps)['windows_traded_pct']:.0%} of truly cointegrated windows; power "
        "falls sharply as the spread half-life lengthens (see per-pair table).",
    ]

    lines = [
        "# SYNTHETIC walk-forward experiment",
        "",
        "> **All numbers on this page come from simulated prices, not market data.** They",
        "> test whether the pipeline behaves sensibly on data whose properties are known.",
        "> They are not evidence of a profitable strategy.",
        "",
        "## Summary",
        "",
        *summary,
        "",
        f"Reproduce: `python scripts/run_synthetic_experiment.py --seed {args.seed}`",
        f"(Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}).",
        "Raw rows for every pair and cost level: "
        "[`synthetic_walkforward.csv`](synthetic_walkforward.csv).",
        "",
        "## Setup",
        "",
        f"- {len(prices)} business days per series (about 8 years), seed {args.seed}.",
        "- Price levels follow arithmetic random walks starting at 400 with daily sd 1.0;",
        "  cointegrated spreads have stationary sd 2.0 price units. These parameters were",
        "  chosen once before the first run and were not tuned to the results.",
        "- Walk-forward: train 252 / test 63 / step 63 days; Engle-Granger at 5% on the",
        "  training window; OLS hedge ratio from the training window only.",
        "- Signal: trailing 20-day z-score (warmed up from the training window, no reset),",
        "  enter at |z| >= 2, exit at z = 0, flatten on the last day of each test window.",
        "- Costs: `cost_bps` of gross notional per unit change in position. Windows that",
        "  fail the cointegration test are kept as flat 0% days in all statistics.",
        "",
        "| pair | type | data-generating process |",
        "|---|---|---|",
        *[f"| {n} | {k} | {d} |" for n, k, d in specs],
        "",
        f"## Per-pair results at {base_bps:g} bps",
        "",
        markdown_table(at_base, cols),
        "",
        "## Cost sensitivity by group",
        "",
        markdown_table(
            group_fmt,
            ["type", "cost_bps", "mean_sharpe", "mean_cagr", "worst_max_drawdown", "windows_traded_pct"],
        ),
        "",
        "## Reading the results",
        "",
        "- `windows_traded` for the controls is the empirical false-positive rate of the",
        "  5% Engle-Granger screen on 252-day windows; trades taken in those windows are",
        "  what multiple testing costs in practice.",
        "- Returns are a fraction of gross pair notional, so CAGR figures are small.",
        "- Turnover per year is the sum of absolute position changes (1 = open or close",
        "  a full-size spread position once).",
        "",
    ]
    md_path = out_dir / "synthetic_walkforward.md"
    md_path.write_text("\n".join(lines))
    print(df[df["cost_bps"] == base_bps][["pair", "sharpe", "cagr", "max_drawdown", "turnover_per_year"]].to_string(index=False))
    print(f"wrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()

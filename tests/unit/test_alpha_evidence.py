"""Live-evidence rules (pre-registered 2026-10-09): overlap-aware sample size, verdicts, checkpoints."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from trading_system.alpha import evidence as E

from test_alpha import _bdays


def _ledger(n_dates: int, h: int, slope: float, n_names: int = 120, matured: bool = True, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in _bdays(n_dates, start=date(2026, 9, 16)):
        s = rng.normal(size=n_names)
        r = slope * s + rng.normal(size=n_names)
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i}", "horizon": h, "mode": "live", "score": float(s[i]),
                         "realized_ret": float(r[i]) if matured else None})
    return pl.DataFrame(rows, schema_overrides={"horizon": pl.Int32, "realized_ret": pl.Float64})


def test_effective_sample_size_counts_overlapping_windows_once():
    assert E.n_eff(0, 21) == 0.0
    assert E.n_eff(1, 21) == pytest.approx(1.0)
    assert E.n_eff(50, 1) == pytest.approx(50.0)                 # daily horizon: no overlap
    assert E.n_eff(21, 21) < 2.0                                  # a month of daily 21d forecasts ≈ one observation
    assert E.n_eff(2100, 21) == pytest.approx(2100 / 21, rel=0.02)


def test_strongly_negative_live_ic_is_red_and_noise_is_green():
    bad = _ledger(30, 5, slope=-1.0)
    rows = [E.horizon_evidence(bad, 5)]
    assert rows[0]["n_eff"] >= E.RED_MIN_NEFF and rows[0]["t"] < -2
    assert E.verdict(rows)[0] == "RED"
    fine = _ledger(30, 5, slope=0.02, seed=1)
    v, why = E.verdict([E.horizon_evidence(fine, 5)])
    assert v == "GREEN" and "not yet informative" in why


def test_nothing_matured_is_green_and_reports_no_ic():
    led = _ledger(5, 21, slope=0.0, matured=False)
    r = E.horizon_evidence(led, 21)
    assert r["matured_dates"] == 0 and "ic" not in r
    assert E.verdict([r]) == ("GREEN", "no live forecast has matured yet")


def test_first_maturity_checkpoint_fires_once():
    led = pl.concat([_ledger(25, 21, slope=0.1), _ledger(25, 63, slope=0.1, matured=False)])
    led = pl.concat([led, _ledger(25, 5, slope=0.1)])
    rep = E.report(led, date(2026, 10, 20))
    due = [k for k, _ in E.due_checkpoints(rep, date(2026, 10, 20), set())]
    assert due == ["21d-first"]                                   # 63d has not matured, calendar ones are in the future
    assert E.due_checkpoints(rep, date(2026, 10, 20), {"21d-first"}) == []
    later = [k for k, _ in E.due_checkpoints(rep, date(2027, 4, 1), {"21d-first"})]
    assert later == ["2026-11-16", "2027-03-15"]


def test_book_is_compared_with_spy_and_rsp_over_its_own_dates(tmp_path):
    log = [{"date": "2026-09-18", "equity": 10000.0}, {"date": "2026-09-21", "equity": 10100.0},
           {"date": "2026-10-08", "equity": 9990.0}]
    (tmp_path / "alpha_v2.json").write_text(json.dumps({"equity_log": log}))
    px = pl.DataFrame({"date": [date(2026, 9, 1), date(2026, 9, 18), date(2026, 10, 8)] * 2,
                       "ticker": ["SPY"] * 3 + ["RSP"] * 3, "adj_close": [90.0, 100.0, 101.6, 50.0, 50.0, 50.1]})
    b = E.book_vs_benchmarks(tmp_path, px)
    assert b["ret"] == pytest.approx(-0.001) and b["SPY"] == pytest.approx(0.016) and b["RSP"] == pytest.approx(0.002)
    text = E.render({"as_of": date(2026, 10, 9), "live_since": date(2026, 9, 16), "live_dates": 3, "verdict": "GREEN",
                     "why": "x", "horizons": [], "book": b}, date(2026, 10, 9))
    assert "vs SPY -1.7%" in text

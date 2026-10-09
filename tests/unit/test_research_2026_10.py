"""Research additions (2026-10-09): candidate features, alternative labels, debiased training, engines,
SEC insider data — all point-in-time."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trading_system.alpha import model as M
from trading_system.alpha import panel as P
from trading_system.ingestion import sec_insider as SI

from test_alpha import _bdays, synth_panel, synth_prices


def test_candidate_features_are_backward_looking():
    px = synth_prices(n_tickers=6, n_days=1400, seed=4)
    base = P.price_features(px).with_columns(sector=pl.lit("x"))
    a = P.market_features(base).filter(pl.col("ticker") == "T00").sort("date")
    last = px["date"].max()
    px2 = px.with_columns(pl.when(pl.col("date") == last).then(pl.col("adj_close") * 3).otherwise(pl.col("adj_close")).alias("adj_close"))
    b = P.market_features(P.price_features(px2).with_columns(sector=pl.lit("x"))).filter(pl.col("ticker") == "T00").sort("date")
    i = 1300
    for c in P.PRICE_CANDIDATES:
        x, y = a[c][i], b[c][i]
        assert (x is None and y is None) or x == pytest.approx(y), c
    assert a["seas_21_1_5"][1300] is not None and a["res_mom_12_1"][1300] is not None


def test_alternative_labels_exist_and_are_gaussian_ranks():
    px = synth_prices(n_tickers=20, n_days=320)
    pn = P.add_labels(P.market_features(P.price_features(px).with_columns(sector=pl.lit("x"))), (21,))
    d = pn.filter(pl.col("date") == sorted(pn["date"].unique().to_list())[-80])
    for lab in ("ys_21", "yr_21"):
        assert lab in pn.columns and abs(d[lab].mean()) < 0.1 and 0.7 < d[lab].std() < 1.3


def test_debiased_training_keeps_only_that_days_most_liquid_names():
    pn = synth_panel(n_tickers=30, n_days=400)
    frame, _ = M.prepare(pn)
    assert "liq_rank" in frame.columns
    spec = M.TrainSpec(horizons=(21,), stride={21: 1}, train_top=10)
    tr = M.training_rows(frame, 21, 350, spec)
    assert tr["liq_rank"].max() <= 10 and tr.group_by("didx").len()["len"].max() <= 10
    assert M.training_rows(frame, 21, 350, M.TrainSpec(horizons=(21,), stride={21: 1})).height > tr.height


@pytest.mark.parametrize("engine", ["lgbm", "ridge"])
def test_alternative_engines_fit_and_predict(engine):
    pn = synth_panel(n_tickers=15, n_days=400)
    spec = M.TrainSpec(horizons=(21,), n_rounds=20, seeds=(0,), device="cpu", engine=engine, stride={21: 2})
    frame, rcols = M.prepare(pn, spec.features)
    tr = M.training_rows(frame, 21, 380, spec)
    X, y = M._xy(tr, rcols, 21)
    m = M.AlphaGBM(spec, 21, rcols).fit(X, y)
    p = m.predict(X[:50])
    assert p.shape == (50,) and np.isfinite(p).all()


def _trades(rows):
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "filing_date": pl.Date, "code": pl.Utf8, "value": pl.Float64,
                                      "officer": pl.Boolean, "routine": pl.Boolean, "accession": pl.Utf8})


def test_insider_trade_is_usable_only_after_its_filing_and_routine_is_ignored():
    days = _bdays(400, start=date(2020, 1, 1))
    panel = pl.DataFrame({"date": days, "ticker": ["AAA"] * len(days), "log_dv_21": [np.log(1e6)] * len(days)})
    fri = next(d for d in days[250:] if d.weekday() == 4)
    tr = _trades([{"ticker": "AAA", "filing_date": fri, "code": "P", "value": 5e4, "officer": True, "routine": False, "accession": "a1"},
                  {"ticker": "AAA", "filing_date": days[260], "code": "P", "value": 9e9, "officer": True, "routine": True, "accession": "a2"},
                  {"ticker": "AAA", "filing_date": days[0], "code": "S", "value": 1.0, "officer": False, "routine": False, "accession": "a0"}])
    out = SI.insider_features(panel, tr).sort("date")
    on_fri = out.filter(pl.col("date") == fri).row(0, named=True)
    nxt = out.filter(pl.col("date") > fri).row(0, named=True)
    assert on_fri["ins_opp_buys_90"] == 0 and nxt["ins_opp_buys_90"] == 1 and nxt["date"].weekday() == 0   # Monday
    assert nxt["ins_officer_buy_180"] == 1.0
    assert out["ins_net_180"].drop_nulls().max() < 1.0             # the routine 9e9 purchase never enters
    late = out.filter(pl.col("date") > fri + timedelta(days=95)).row(0, named=True)
    assert late["ins_opp_buys_90"] == 0                             # 90-day window rolled off


def test_cmp_routine_rule_tags_same_month_three_years_running(tmp_path):
    import io
    import zipfile
    def q(rows):
        sub = "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
        own = "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
        tr = "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
        for acc, d, code in rows:
            sub += f"{acc}\t{d}\t4\t0000000042\tAAA\tfalse\n"
            own += f"{acc}\t0000000007\tOfficer\n"
            tr += f"{acc}\t{d}\t{code}\t100\t10\tA\n"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("SUBMISSION.tsv", sub); z.writestr("REPORTINGOWNER.tsv", own); z.writestr("NONDERIV_TRANS.tsv", tr)
        return buf.getvalue()
    (tmp_path / "2023q1_form345.zip").write_bytes(q([("a20", "10-MAR-2020", "S"), ("a21", "11-MAR-2021", "S"),
                                                      ("a22", "09-MAR-2022", "S"), ("a23", "14-MAR-2023", "P"),
                                                      ("b23", "14-APR-2023", "P")]))
    df = SI.build_trades(tmp_path, tmp_path / "out.parquet", {42: "AAA"})
    r = dict(zip(df["accession"], df["routine"]))
    assert r["a23"] is True and r["b23"] is False and r["a20"] is False
    assert set(df["ticker"]) == {"AAA"}

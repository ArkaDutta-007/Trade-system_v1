"""`ts alpha …` — the alpha engine's command surface (registered by ``trading_system.cli``)."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich import print as rprint
from rich.table import Table

alpha_app = typer.Typer(add_completion=False, help="Alpha engine v2: fast cross-sectional forecaster with a "
                        "tallied forecast ledger, continuous recalibration and a causal backtest.")

CONFIG_OPT = typer.Option("configs/default.yaml", help="config file")


def _cfg(config: str):
    from ..config import get_config
    return get_config(config)


@alpha_app.command("panel")
def alpha_panel(config: str = CONFIG_OPT, start: str = typer.Option("1998-01-01", help="first date of the panel")):
    """Build the point-in-time feature panel (prices ⊕ fundamentals ⊕ news ⊕ short data) → data/gold/alpha_panel.parquet."""
    from .panel import build_panel, save_panel, FEATURE_COLS, MODEL_FEATURES
    cfg = _cfg(config)
    pn = build_panel(cfg, start=start)
    p = save_panel(cfg, pn)
    cov = pn.select([pl.col(c).is_not_null().mean().alias(c) for c in FEATURE_COLS if c in pn.columns]).to_dicts()[0]
    thin = [f"{k} {v:.0%}" for k, v in cov.items() if v < 0.5]
    rprint(f"[green]panel:[/green] {pn.height:,} rows · {pn['ticker'].n_unique()} tickers · {pn['date'].min()} → {pn['date'].max()} · "
           f"{len(MODEL_FEATURES)} model features + {len(FEATURE_COLS) - len(MODEL_FEATURES)} regime context columns → {p}")
    if thin:
        rprint(f"[dim]sparse features (expected — younger data sets): {', '.join(thin)}[/dim]")


@alpha_app.command("train")
def alpha_train(config: str = CONFIG_OPT, rounds: int = typer.Option(400), seeds: int = typer.Option(2),
                device: str = typer.Option("auto", help="auto | cuda | cpu")):
    """Fit the production forecasters (one per horizon) on every matured label → data/models/alpha/."""
    from .model import TrainSpec, fit_production, models_dir
    from .panel import load_panel
    cfg = _cfg(config)
    spec = TrainSpec(n_rounds=rounds, seeds=tuple(range(seeds)), device=device)
    models = fit_production(load_panel(cfg), spec, models_dir(cfg))
    for h, m in models.items():
        top = list(m.importance().items())[:6]
        rprint(f"[green]h={h}d[/green] trained through {m.trained_through} on {m.n_rows:,} rows ({m.device}) · "
               f"top: " + ", ".join(f"{k.removesuffix('_r')}" for k, _ in top))


@alpha_app.command("forecast")
def alpha_forecast(config: str = CONFIG_OPT, days: int = typer.Option(1, help="score the last N panel dates (fills gaps)"),
                   window_years: float = typer.Option(3.0, help="calibration window over matured forecasts"),
                   analog_weight: float = typer.Option(0.5, help="blend weight of the regime-analog calibration (0 = off)")):
    """Score the latest cross-section, recalibrate on the tallied ledger (trailing window ⊕ regime analogs),
    record live forecasts."""
    from . import ledger as L
    from .model import load_production, models_dir, predict_dates
    from .panel import load_panel
    cfg = _cfg(config)
    pn = load_panel(cfg)
    models = load_production(models_dir(cfg))
    dates = sorted(pn["date"].unique().to_list())[-days:]
    scores = predict_dates(pn, models, dates)
    lp = L.ledger_path(cfg)
    led = L.load(lp)
    dw = None
    if analog_weight > 0 and led.height:
        try:
            from .regime import build_state, similarity
            _, zf = build_state(cfg, pn)
            an = similarity(zf, as_of=dates[-1])
            dw = an.weights
        except Exception as e:
            rprint(f"[yellow]regime analogs unavailable ({str(e)[:80]}) — trailing calibration only[/yellow]")
    cal = (L.Calibrator.fit(led, list(models), window_days=int(window_years * 365), date_weights=dw, analog_blend=analog_weight)
           if led.height else L.Calibrator())
    cal.save(models_dir(cfg) / "calibration.json")
    rows = cal.apply(scores).join(pn.select("date", "ticker", entry_price=pl.col("adj_close").cast(pl.Float32)),
                                  on=["date", "ticker"], how="left").with_columns(mode=pl.lit("live"))
    n = L.upsert(lp, rows)
    w = cal.skill_weights(list(models))
    rprint(f"[green]forecast:[/green] {n:,} live forecasts recorded for {dates[0]} → {dates[-1]} "
           f"({scores['ticker'].n_unique()} tickers × {len(models)} horizons) → {lp}")
    if cal.horizons:
        rprint("[bold]recalibrated[/bold] on matured ledger rows: " + " · ".join(
            f"{h}d IC {c.ic:+.3f}" + (f" (analog {c.ic_analog:+.3f}, ESS {c.ess:,.0f})" if c.ic_analog is not None else "")
            + f" w={w.get(h, 0):.2f}" for h, c in sorted(cal.horizons.items())))
    else:
        rprint("[yellow]no matured forecasts yet — expected returns/bands are null until the ledger has history "
               "(run `ts alpha backtest` once to seed it)[/yellow]")


@alpha_app.command("tally")
def alpha_tally(config: str = CONFIG_OPT):
    """Score every matured forecast against realised prices and print the skill report."""
    from . import ledger as L
    from .panel import load_prices
    cfg = _cfg(config)
    lp = L.ledger_path(cfg)
    res = L.tally(lp, load_prices(cfg, start="1998-01-01"))
    rprint(f"[green]tally:[/green] {res['matured']:,} matured · {res['pending']:,} still open")
    _print_skill(L.load(lp))


def _print_skill(led: pl.DataFrame) -> None:
    from . import ledger as L
    for mode, since in (("live", None), ("backtest", date.today() - timedelta(days=365 * 3))):
        rep = L.skill_report(led, since=since, mode=mode)
        if rep.height == 0:
            rprint(f"[dim]{mode}: no matured forecasts yet[/dim]")
            continue
        t = Table(title=f"realised skill — {mode}" + (f" (since {since})" if since else ""), show_lines=False)
        for c in ("horizon", "n", "dates", "IC", "ICIR", "t", "hit", "decile spread", "80% cov", "calib slope"):
            t.add_column(c, justify="right")
        def f(v, fmt):
            return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
        for r in rep.iter_rows(named=True):
            t.add_row(f"{r['horizon']}d", f"{r['n']:,}", str(r.get("n_dates") or 0), f(r.get("ic_mean"), "+.4f"),
                      f(r.get("icir"), ".2f"), f(r.get("t_stat"), ".1f"), f(r.get("hit_rate"), ".1%"),
                      f(r.get("decile_spread"), "+.2%"), f(r.get("coverage_80"), ".0%"), f(r.get("calib_slope"), ".2f"))
        rprint(t)


@alpha_app.command("backtest")
def alpha_backtest(config: str = CONFIG_OPT, refit_every: int = typer.Option(63, help="trading days between model refits"),
                   refresh: bool = typer.Option(False, help="recompute walk-forward scores even if cached"),
                   extend: bool = typer.Option(False, help="score only the dates after the cached scores (weekly use)"),
                   oos_start: str = typer.Option("2004-01-01"), top_k: int = typer.Option(20),
                   trade_rate: float = typer.Option(0.35), n_boot: int = typer.Option(500)):
    """Causal walk-forward: scores → ledger tally → walked-forward calibration → book vs momentum / universe / SPY."""
    from . import ledger as L
    from .backtest import DEFAULT_NOTES, causal_composite, run_book_backtest, scores_to_ledger_rows, write_report
    from .model import TrainSpec, causal_scores
    from .panel import HORIZONS, load_panel, load_prices
    from .portfolio import BookConfig
    cfg = _cfg(config)
    pn = load_panel(cfg)
    out_dir = cfg.path("reports") / "alpha"
    cache = cfg.path("data_bronze").parent / "ledger" / "alpha_backtest_scores.parquet"
    if cache.exists() and not refresh:
        sc = pl.read_parquet(cache)
        rprint(f"[dim]using cached walk-forward scores {cache.name} ({sc.height:,} rows, {sc['date'].min()} → {sc['date'].max()})[/dim]")
        if extend:
            last = sc["date"].max()
            newer = pn.filter(pl.col("date") > last)
            if newer.select("date").unique().height >= 1:
                more = causal_scores(pn, TrainSpec(), refit_every=refit_every, min_train_days=1260,
                                     oos_start=last + timedelta(days=1))
                sc = pl.concat([sc, more.filter(pl.col("date") > last)]).unique(subset=["date", "ticker", "horizon"], keep="last")
                sc.write_parquet(cache, compression="zstd")
                rprint(f"[green]extended[/green] walk-forward scores to {sc['date'].max()}")
    else:
        sc = causal_scores(pn, TrainSpec(), refit_every=refit_every, min_train_days=1260)
        cache.parent.mkdir(parents=True, exist_ok=True)
        sc.write_parquet(cache, compression="zstd")
    lp = L.ledger_path(cfg)
    L.upsert(lp, scores_to_ledger_rows(sc, pn))
    res = L.tally(lp, load_prices(cfg, start="1998-01-01"))
    rprint(f"[green]ledger:[/green] backtest rows tallied — {res['matured']:,} matured, {res['pending']:,} open")
    led = L.load(lp).filter(pl.col("mode") == "backtest")
    signal = L.skill_report(led, since=date.fromisoformat(oos_start))
    yearly = L.yearly_ic(led)
    composite, calib_log = causal_composite(led, HORIZONS, refit_every=refit_every)
    book = run_book_backtest(cfg, pn, composite, BookConfig(top_k=top_k), oos_start=date.fromisoformat(oos_start),
                             trade_rate=trade_rate, out_dir=out_dir, n_boot=n_boot)
    p = write_report(out_dir, signal, yearly, book, calib_log, DEFAULT_NOTES)
    _print_skill(L.load(lp))
    t = Table(title="book (after costs)")
    for c in ("variant", "CAGR", "Sharpe", "95% CI", "MaxDD", "turnover", "excess vs eligible EW"):
        t.add_column(c, justify="right")
    for r in sorted(book["table"], key=lambda r: -r["Sharpe"]):
        s = book["stats"].get(r["variant"], {})
        t.add_row(r["variant"], f"{r['CAGR']:+.1%}", f"{r['Sharpe']:.2f}", str(s.get("sharpe_ci", "")),
                  f"{r['MaxDD']:.1%}", f"{r['ann_turnover']}", f"{s.get('excess_vs_eligible_ew', 0):+.1%}")
    rprint(t)
    rprint(f"[green]report:[/green] {p}")


@alpha_app.command("picks")
def alpha_picks(config: str = CONFIG_OPT, top: int = typer.Option(20), compact: bool = typer.Option(False),
                prev: str = typer.Option("", help="JSON {ticker: weight} of the current book → shows the partial-trade step"),
                json_out: str = typer.Option("", help="write targets + meta to this JSON path"),
                budget: float = typer.Option(0.0, help="dollar budget → share counts"),
                explain: bool = typer.Option(True, help="show the top model drivers per name")):
    """The book: gated, risk-adjusted, sector-capped, vol-targeted target weights (and the GP step from --prev)."""
    from .live import latest_targets, write_targets_json
    from .model import explain_latest, load_production, models_dir
    from .panel import load_panel
    cfg = _cfg(config)
    pn = load_panel(cfg)
    cur = json.loads(Path(prev).read_text()) if prev and Path(prev).exists() else None
    if isinstance(cur, dict) and "weights" in cur:
        cur = cur["weights"]
    if isinstance(cur, dict) and "holdings" in cur:          # an ops paper book: shares → weights at last close
        last_px = dict(pn.filter(pl.col("date") == pn["date"].max()).select("ticker", "close").iter_rows())
        vals = {t: q * last_px.get(t, 0.0) for t, q in cur["holdings"].items()}
        eq = sum(vals.values()) + float(cur.get("cash", 0.0))
        cur = {t: v / eq for t, v in vals.items() if eq > 0}
    book, meta = latest_targets(cfg, top, prev=cur, panel=pn)
    last = meta["as_of"]
    held = book.filter(pl.col("weight") > 0)
    drivers = {}
    if explain and held.height:
        try:
            drivers = explain_latest(pn, load_production(models_dir(cfg)), held["ticker"].to_list(), 21, top=3)
        except Exception:
            drivers = {}
    e21 = "exp_ret_21" if "exp_ret_21" in book.columns else None
    q10c, q90c = ("q10_21", "q90_21") if "q10_21" in book.columns else (None, None)
    wdesc = ", ".join(f"{h}d {v:.2f}" for h, v in meta["weights"].items())
    if compact:
        rprint(f"🎯 *Alpha book* · as of {last} · {held.height} names · gross {held['weight'].sum():.0%}")
        rprint("_gated · risk-adjusted · sector-capped · vol-targeted · partial trading_")
        for i, r in enumerate(held.iter_rows(named=True), 1):
            er = f" · 21d {r[e21]:+.1%}" if e21 and r.get(e21) is not None else ""
            chg = r["weight"] - r["prev_weight"]
            act = ("🟢 add" if chg > 0.005 else ("🔴 trim" if chg < -0.005 else "⚪ hold")) if cur is not None else "🟢 target"
            rprint(f"{i}. *{r['ticker']}*  ${r['price']:.0f} · {r['weight']:.1%} · {r['sector'].replace('_', ' ')}{er} · {act}")
        reg = "risk-on" if meta["regime_on"] else f"risk-OFF (market {meta['mkt_trend_200']:+.1%} vs 200d avg → half gross)"
        rprint(f"_horizon weights {wdesc} · {'calibrated' if meta['calibrated'] else 'uncalibrated'} · {reg}_")
    else:
        t = Table(title=f"Alpha book · as of {last} · horizon weights {wdesc} · from {meta['source']}")
        for c in ("#", "ticker", "price", "wt", "target", "Δ", "E[r] 21d", "80% band 21d", "E[r] 63d", "vol", "$vol/d", "sector", "drivers"):
            t.add_column(c, justify="right" if c not in ("ticker", "sector", "drivers") else "left")
        for i, r in enumerate(book.filter((pl.col("weight") > 0) | (pl.col("prev_weight") > 0)).iter_rows(named=True), 1):
            er21 = f"{r[e21]:+.1%}" if e21 and r.get(e21) is not None else "—"
            band = f"[{r[q10c]:+.0%}, {r[q90c]:+.0%}]" if q10c and r.get(q10c) is not None else "—"
            er63 = f"{r['exp_ret_63']:+.1%}" if r.get("exp_ret_63") is not None else "—"
            drv = ", ".join(f"{k}{'↑' if v > 0 else '↓'}" for k, v in drivers.get(r["ticker"], [])) or ""
            t.add_row(str(i), r["ticker"], f"{r['price']:.2f}", f"{r['weight']:.1%}", f"{r['target_weight']:.1%}",
                      f"{r['weight'] - r['prev_weight']:+.1%}", er21, band, er63, f"{r['vol']:.0%}", f"{r['adv'] / 1e6:.0f}M",
                      r["sector"], drv)
        rprint(t)
        gross = float(held["weight"].sum())
        bk = meta["book"]
        reg = "risk-on" if meta["regime_on"] else "risk-OFF → half gross"
        rprint(f"market {meta['mkt_trend_200']:+.1%} vs its 200d average → {reg}")
        rprint(f"gross {gross:.0%} (vol target {bk['vol_target']:.0%}) · {held.height} names · max {held['weight'].max():.1%} · "
               f"sectors {held.group_by('sector').len().height} · turnover this step "
               f"{float((book['weight'] - book['prev_weight']).abs().sum()) / 2:.1%}")
        if budget > 0:
            rprint("shares for $%.0f: " % budget + ", ".join(
                f"{r['ticker']} {int(budget * r['weight'] // r['price'])}" for r in held.iter_rows(named=True)))
    if json_out:
        p = write_targets_json(cfg, Path(json_out), top)
        rprint(f"[dim]targets → {p}[/dim]")


@alpha_app.command("regime")
def alpha_regime(config: str = CONFIG_OPT, k: int = typer.Option(10, help="nearest analog dates to show"),
                 compact: bool = typer.Option(False, help="digest-sized summary")):
    """Where are we (oil, vol, credit, rates, concentration), which past episodes look like today, and what the
    signal and the book did in those backgrounds — the inputs to the regime-aware recalibration."""
    from . import ledger as L
    from .panel import load_panel
    from .regime import (EPISODES, STRESS_COLS, IMBALANCE_COLS, REGIME_COLS, build_state, conditional_forward,
                         conditional_skill, fragility_score, gross_multiplier, similarity)
    cfg = _cfg(config)
    pn = load_panel(cfg)
    rf, zf = build_state(cfg, pn)
    an = similarity(zf, k=k)
    fr = fragility_score(zf).drop_nulls("stress")
    today_f = fr.tail(1).row(0, named=True)
    pct = float((fr["stress"] <= today_f["stress"]).mean())
    st = an.state
    def z(c):
        v = st.get(c + "_z"); return "—" if v is None else f"{v:+.1f}"
    def raw(c, fmt=".2f"):
        v = st.get(c); return "—" if v is None else format(v, fmt)
    rprint(f"[bold]Regime as of {an.as_of}[/bold] · stress {today_f['stress']:+.2f} (pct {pct:.0%}) · imbalance {today_f['imbalance']:+.2f} · "
           f"fragility {today_f['fragility']:+.2f} · gross multiplier if the stress overlay were on: {gross_multiplier(today_f['stress']):.2f}")
    rprint(f"  equity vol: VIX {raw('vix', '.1f')} (z {z('vix')}) · VXN/VIX {raw('vxn_vix')} · mkt vol {raw('mkt_vol_21', '.0%')} (z {z('mkt_vol_21')}) · "
           f"avg corr {raw('avg_corr_21')} (z {z('avg_corr_21')}) · breadth {raw('breadth_200', '.0%')} · drawdown {raw('mkt_dd_252', '.1%')}")
    rprint(f"  oil: WTI level z {raw('oil_level_z', '+.1f')} · 63d {raw('oil_ret_63', '+.0%')} (z {z('oil_ret_63')}) · realised vol {raw('oil_vol_21', '.0%')} (z {z('oil_vol_21')}) · OVX {raw('ovx', '.0f')} (z {z('ovx')})")
    rprint(f"  credit/rates: Baa spread {raw('baa_spread')} (z {z('baa_spread')}, 63d Δ {raw('baa_chg_63', '+.2f')}) · curve 10y-3m {raw('curve_10y3m', '+.2f')} · "
           f"real 10y {raw('real10')} (z {z('real10')}) · 10y 63d Δ {raw('ust10_chg_63', '+.2f')} · breakeven {raw('breakeven10')} · dollar 63d {raw('dollar_ret_63', '+.1%')}")
    rprint(f"  AI/tech: basket vol {raw('ai_vol_21', '.0%')} (z {z('ai_vol_21')}) · basket 63d {raw('ai_ret_63', '+.1%')} vs market {raw('ai_rel_63', '+.1%')} · "
           f"share of $volume {raw('ai_share', '.0%')} (z {z('ai_share')}) · Nasdaq rel. 252d {raw('nasdaq_rel_252', '+.1%')} · Nasdaq drawdown {raw('nasdaq_dd', '.1%')}")
    if an.episodes.height:
        rprint("[bold]closest historical episodes[/bold] (kernel similarity, 1 = identical background)")
        for r in an.episodes.head(6).iter_rows(named=True):
            rprint(f"  {r['similarity']:.2f}  {r['episode']:<24} {r['from']} → {r['to']}  {r['what']}")
        far = an.episodes.tail(3)
        rprint("  least similar: " + ", ".join(f"{r['episode']} {r['similarity']:.2f}" for r in far.iter_rows(named=True)))
    if not compact:
        rprint("[bold]nearest analog days[/bold]: " + ", ".join(
            f"{r['date']}{(' [' + r['episode'] + ']') if r['episode'] else ''} ({r['dist']:.2f})" for r in an.nearest.iter_rows(named=True)))
    led = L.load(L.ledger_path(cfg))
    if led.height:
        for h in (21, 63):
            cs = conditional_skill(led, an.weights, h)
            if cs:
                rprint(f"[bold]signal in analog backgrounds[/bold] {h}d rank-IC {cs['ic_analog']:+.3f} vs unconditional {cs['ic_uncond']:+.3f} "
                       f"(effective analog days {cs['effective_analog_days']:.0f})")
    curve = cfg.path("reports") / "alpha" / "daily_alpha_book.parquet"
    if curve.exists():
        d = pl.read_parquet(curve)
        cf = conditional_forward(d, an.weights, 63)
        if cf:
            rprint(f"[bold]book in analog backgrounds[/bold] next-63d return: mean {cf['fwd_mean_analog']:+.1%} (median {cf['fwd_median_analog']:+.1%}, "
                   f"worst decile {cf['fwd_q10_analog']:+.1%}) vs unconditional {cf['fwd_mean_uncond']:+.1%} · mean drawdown inside the window "
                   f"{cf['mdd_mean_analog']:.1%} vs {cf['mdd_mean_uncond']:.1%}")
    cp = __import__("trading_system.alpha.model", fromlist=["models_dir"]).models_dir(cfg) / "calibration.json"
    if cp.exists():
        cal = L.Calibrator.load(cp)
        parts = []
        for h, c in sorted(cal.horizons.items()):
            if c.ic_analog is not None:
                parts.append(f"{h}d trailing IC {c.ic:+.3f} / analog {c.ic_analog:+.3f}")
        if parts:
            rprint("[bold]calibration in use[/bold] (blend of trailing window and analog-weighted fit): " + " · ".join(parts))


@alpha_app.command("status")
def alpha_status(config: str = CONFIG_OPT):
    """Ledger, models, calibration and the rolling realised IC at a glance."""
    from . import ledger as L
    from .model import models_dir
    from .panel import panel_path
    cfg = _cfg(config)
    pp = panel_path(cfg)
    if pp.exists():
        lf = pl.scan_parquet(pp)
        n, dmax, nt = lf.select(pl.len(), pl.col("date").max(), pl.col("ticker").n_unique()).collect().row(0)
        rprint(f"[bold]panel[/bold] {n:,} rows · {nt} tickers · through {dmax}")
    else:
        rprint("[yellow]panel missing — `ts alpha panel`[/yellow]")
    md = models_dir(cfg)
    if (md / "manifest.json").exists():
        man = json.loads((md / "manifest.json").read_text())
        rprint(f"[bold]models[/bold] horizons {man['horizons']} · as of {man['as_of']} · {len(man['features'])} features")
    else:
        rprint("[yellow]no production models — `ts alpha train`[/yellow]")
    cp = md / "calibration.json"
    if cp.exists():
        cal = L.Calibrator.load(cp)
        w = cal.skill_weights()
        rprint("[bold]calibration[/bold] " + " · ".join(f"{h}d IC {c.ic:+.3f} w {w.get(h, 0):.2f} (n={c.n:,}, through {c.fitted_through})"
                                                       for h, c in sorted(cal.horizons.items())))
    led = L.load(L.ledger_path(cfg))
    if led.height == 0:
        rprint("[yellow]ledger empty — `ts alpha backtest` seeds it, `ts alpha forecast` adds live rows[/yellow]")
        return
    for mode in ("live", "backtest"):
        m = led.filter(pl.col("mode") == mode)
        if m.height:
            rprint(f"[bold]ledger/{mode}[/bold] {m.height:,} forecasts · {m['date'].min()} → {m['date'].max()} · "
                   f"matured {m.filter(pl.col('realized_ret').is_not_null()).height:,}")
    _print_skill(led)
    for h in (21,):
        r = L.rolling_ic(led, h, 63)
        if r.height:
            tail = r.tail(1).row(0, named=True)
            rprint(f"rolling 63-date IC ({h}d, all modes): {tail['ic_roll']:+.4f} as of {tail['date']}")

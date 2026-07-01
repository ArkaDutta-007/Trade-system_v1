"""Top-level CLI: `ts <command>`."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from .backtesting import compute_metrics, run_vectorized_backtest, summarize
from .backtesting.slippage import CostModel
from .config import get_config
from .decision import analyze_symbol, analyze_all
from .decision.explain import explain_report, DEEPSEEK_DEFAULT_MODEL
from .features import build_feature_matrix
from .ingestion import ingest_universe, fetch_news, compute_apprehension_scores
from .models.shap_analysis import compute_shap_summary
from .models.train import FeatureSpec, train_walk_forward
from .pipeline import run_daily_pipeline
from .quality import run_ohlcv_checks
from .strategies import (
    BuyAndHold,
    MeanReversionAfterDrop,
    MLRankerStrategy,
    MomentumRotation,
    MovingAverageCrossover,
)
import polars as pl

app = typer.Typer(add_completion=False, help="Trading-system CLI.")

STRATS = {
    "buy_and_hold": lambda: BuyAndHold(),
    "ma_crossover": lambda: MovingAverageCrossover(),
    "momentum_rotation": lambda: MomentumRotation(),
    "mean_reversion": lambda: MeanReversionAfterDrop(),
}

UNIVERSE_OPT = typer.Option(
    "", "--universe", "-u",
    help="Universe to use: core (default, from portfolio JSON) | master | 100 | path to YAML",
)


@app.command()
def ingest(
    config: str = "configs/default.yaml",
    universe: str = UNIVERSE_OPT,
    workers: int = typer.Option(8, help="concurrent fetch workers (TS_INGEST_WORKERS)"),
):
    """Ingest the configured universe to bronze parquet (threaded, with progress)."""
    cfg = get_config(config).use_universe(universe)
    out = ingest_universe(cfg, workers=workers)
    rprint(f"[green]Wrote {out}[/green]")

    # Fetch news and append to silver/events.parquet
    silver = cfg.path("data_silver")
    silver.mkdir(parents=True, exist_ok=True)
    events_path = silver / "events.parquet"
    try:
        tickers = cfg["universe"]["tickers"]
        news_cfg = cfg.get("news", {}) or {}
        rprint(f"[dim]fetching news ({len(tickers)} tickers, backends: "
               f"{news_cfg.get('backends', ['finnhub','newsdata','google_news','newsapi'])})…[/dim]")
        new_events = fetch_news(
            tickers,
            backends=news_cfg.get("backends"),
            cache_dir=silver / "news_cache",
            cache_hours=float(news_cfg.get("cache_hours", 6.0)),
            dedup_cosine=float(news_cfg.get("dedup_cosine", 0.90)),
            max_per_ticker=int(news_cfg.get("max_per_ticker", 25)),
        )
        if not new_events.is_empty():
            if events_path.exists():
                existing = pl.read_parquet(events_path)
                combined = pl.concat([existing, new_events], how="diagonal")
                cutoff = combined["known_at"].max() - pl.duration(days=90)
                combined = (
                    combined
                    .unique(subset=["event_id"], keep="first")
                    .filter(pl.col("known_at") >= cutoff)
                )
                combined.write_parquet(events_path, compression="zstd")
            else:
                new_events.write_parquet(events_path, compression="zstd")
            rprint(f"[green]News events: {len(new_events)} new rows → {events_path}[/green]")
        else:
            rprint("[yellow]No news fetched (NEWSAPI_KEY not set or no results)[/yellow]")
    except Exception as e:
        rprint(f"[yellow]News fetch failed (non-fatal): {e}[/yellow]")

    # Compute apprehension scores and save to silver/apprehension_scores.parquet
    apprehension_path = silver / "apprehension_scores.parquet"
    try:
        if events_path.exists():
            all_events = pl.read_parquet(events_path)
            new_app = compute_apprehension_scores(all_events)
            if not new_app.is_empty():
                if apprehension_path.exists():
                    existing_app = pl.read_parquet(apprehension_path)
                    new_app = (
                        pl.concat([existing_app, new_app], how="diagonal")
                        .unique(subset=["date", "ticker"], keep="last")
                        .sort(["ticker", "date"])
                    )
                new_app.write_parquet(apprehension_path, compression="zstd")
                rprint(f"[green]Apprehension scores: {len(new_app)} tickers → {apprehension_path}[/green]")
    except Exception as e:
        rprint(f"[yellow]Apprehension scoring failed (non-fatal): {e}[/yellow]")


@app.command()
def quality(config: str = "configs/default.yaml"):
    """Run data-quality checks on bronze OHLCV."""
    cfg = get_config(config)
    df = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")
    res = run_ohlcv_checks(df)
    rprint(res)


@app.command()
def features(
    config: str = "configs/default.yaml",
    universe: str = UNIVERSE_OPT,
    deep: bool = typer.Option(False, help="also compute the heavy nonlinear tier "
                                          "(sample entropy, Lyapunov, RQA, 0-1 chaos, LPPLS)"),
    jobs: int = typer.Option(0, help="parallel workers for nonlinear features (0 = auto: cores-1)"),
    no_parallel: bool = typer.Option(False, "--no-parallel", help="compute nonlinear features serially"),
    text: bool = typer.Option(False, "--text", help="add FinBERT news-sentiment features (needs .[text] + events)"),
):
    """Build the feature matrix (technical + extended + macro + events + nonlinear) → gold."""
    from trading_system.features.context import build_macro_inputs
    from trading_system.features.reserve import reserve_report

    cfg = get_config(config).use_universe(universe)
    df = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")

    # Optional silver inputs (present once `ts daily` / news ingestion has run)
    silver = cfg.path("data_silver")
    events = pl.read_parquet(silver / "events.parquet") if (silver / "events.parquet").exists() else None
    appr_p = silver / "apprehension_scores.parquet"
    apprehension = pl.read_parquet(appr_p) if appr_p.exists() else None

    macro_features, econ_cal, earnings_cal = build_macro_inputs(
        cfg, tickers=list(cfg["universe"]["tickers"]), with_earnings=True
    )
    feat = build_feature_matrix(
        df,
        events=events,
        apprehension=apprehension,
        economic_calendar=econ_cal,
        earnings_calendar=earnings_cal,
        macro_features=macro_features,
        benchmark=cfg["universe"]["benchmark"],
        nonlinear_deep=deep,
        nonlinear_parallel=not no_parallel,
        nonlinear_jobs=(jobs or None),
        add_text_features=text,
        text_cache_dir=(silver / "finbert_cache") if text else None,
    )
    out = cfg.path("data_gold") / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(out, compression="zstd")
    rprint(f"[green]Wrote {len(feat)} rows × {feat.width} cols to {out}[/green]")
    rep = reserve_report(feat)
    rprint(f"[cyan]feature reserve present: {rep['_total']['present']}/"
           f"{rep['_total']['reserve_size']}[/cyan] "
           + " · ".join(f"{g}:{i['present']}/{i['defined']}"
                        for g, i in rep.items() if not g.startswith('_')))


@app.command()
def backtest(
    strategy: str = typer.Argument("momentum_rotation"),
    config: str = "configs/default.yaml",
    universe: str = UNIVERSE_OPT,
):
    """Backtest a named strategy on the gold features."""
    if strategy not in STRATS:
        raise typer.BadParameter(f"Unknown strategy: {strategy}. Options: {list(STRATS)}")
    cfg = get_config(config).use_universe(universe)
    ohlcv = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")
    feats_path = cfg.path("data_gold") / "features.parquet"
    feat = pl.read_parquet(feats_path) if feats_path.exists() else build_feature_matrix(ohlcv)

    strat = STRATS[strategy]()
    weights = strat.generate_signals(feat)
    cost = CostModel(
        commission_bps=cfg["backtest"]["commission_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        spread_bps=cfg["backtest"]["spread_bps"],
    )
    res = run_vectorized_backtest(
        ohlcv, weights, cost=cost,
        signal_delay_days=cfg["backtest"]["signal_delay_days"],
        initial_cash=cfg["backtest"]["initial_cash"],
        max_position_weight=cfg["backtest"]["max_position_weight"],
        max_gross_exposure=cfg["backtest"]["max_gross_exposure"],
        benchmark=cfg["universe"]["benchmark"],
    )
    metrics = compute_metrics(
        res.daily["net_ret"].to_numpy(),
        turnover=res.daily["turnover"].to_numpy(),
        benchmark=res.benchmark_ret["ret"].to_numpy() if res.benchmark_ret is not None else None,
    )
    rprint(summarize(metrics))


@app.command()
def train(config: str = "configs/default.yaml"):
    """Walk-forward train 14-model ensemble over the feature matrix."""
    from rich.table import Table
    from rich.console import Console
    import json as _json
    import time as _time

    from trading_system.models.model_registry import save_model as _save_model, save_ensemble_report

    cfg = get_config(config)
    feat = pl.read_parquet(cfg.path("data_gold") / "features.parquet")
    spec = FeatureSpec(
        feature_columns=[
            "mom_5d", "mom_20d", "mom_60d", "mom_120d", "mom_12m1m",
            "vol_20d", "vol_60d", "rsi_14", "rel_vol_20",
            "sma_gap_50", "sma_gap_200", "breakout_20", "dd_from_high_60",
            "excess_ret_1d",
        ],
        target=cfg["model"]["target"],
    )
    spec.feature_columns = [c for c in spec.feature_columns if c in feat.columns]
    wf = cfg["model"]["walk_forward"]

    fold_records, oos, metrics_df = train_walk_forward(
        feat, spec,
        train_years=wf["train_years"],
        test_years=wf["test_years"],
        step_years=wf["step_years"],
    )

    # Save OOS predictions
    out = cfg.path("data_gold") / "predictions.parquet"
    oos.write_parquet(out, compression="zstd")
    rprint(f"[green]Wrote {len(oos)} OOS predictions to {out}[/green]")

    if not fold_records:
        rprint("[yellow]No folds completed — check feature data range.[/yellow]")
        return

    # ── Aggregate metrics across folds ──────────────────────────────────
    agg = (
        metrics_df
        .group_by("model")
        .agg([
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("mae").mean().alias("mae_mean"),
            pl.col("r2").mean().alias("r2_mean"),
            pl.col("weight").mean().alias("weight_mean"),
        ])
        .sort("ic_mean", descending=True)
    )

    # ── Rich comparative table ──────────────────────────────────────────
    console = Console()
    table = Table(title="Model Comparison (averaged across walk-forward folds)", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("IC (mean)", justify="right")
    table.add_column("IC (std)", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Blend Weight", justify="right")

    for row in agg.to_dicts():
        ic = row["ic_mean"]
        color = "green" if ic > 0.02 else ("yellow" if ic > 0 else "red")
        is_ensemble = row["model"].startswith("ensemble_")
        style = "bold" if is_ensemble else ""
        table.add_row(
            f"{'★ ' if is_ensemble else ''}{row['model']}",
            f"[{color}]{ic:+.4f}[/{color}]",
            f"{row['ic_std']:.4f}" if row["ic_std"] else "—",
            f"{row['mae_mean']:.6f}",
            f"{row['r2_mean']:+.4f}",
            f"{row['weight_mean']:.4f}" if row["weight_mean"] else "—",
            style=style,
        )
    console.print(table)

    # ── Save ensemble + best individual to registry ──────────────────────
    last_fold = fold_records[-1]
    ensemble = last_fold["ensemble"]
    reg_path = cfg.path("reports") / "models"
    stamp = int(_time.time())

    # Save full ensemble object
    ensemble_name = f"ensemble_{stamp}"
    _save_model(
        ensemble,
        name=ensemble_name,
        feature_columns=spec.feature_columns,
        target=spec.target,
        metadata={
            "model_type": "ensemble",
            "n_folds": len(fold_records),
            "best_variant": last_fold["best_variant"],
            "oos_rows": len(oos),
            "blend_weights": last_fold["blend_weights"],
        },
        registry=reg_path,
    )
    rprint(f"[green]Ensemble saved -> {reg_path / ensemble_name}[/green]")

    # Save comparative report
    metrics_rows = metrics_df.to_dicts()
    agg_rows = agg.to_dicts()
    report_path = save_ensemble_report(
        {"per_fold": metrics_rows, "aggregated": agg_rows},
        cfg.path("reports") / "model_comparison.json",
    )
    rprint(f"[green]Model comparison -> {report_path}[/green]")

    # SHAP summary from best tree model (lgbm in ensemble)
    lgbm_model = ensemble._models.get("lgbm")
    if lgbm_model is not None:
        try:
            sh = compute_shap_summary(lgbm_model, feat, spec.feature_columns)
            sh_out = cfg.path("reports") / "shap_summary.csv"
            sh_out.parent.mkdir(parents=True, exist_ok=True)
            sh.write_csv(sh_out)
            rprint(f"SHAP summary -> {sh_out}")
        except Exception as e:
            rprint(f"[yellow]SHAP summary skipped: {e}[/yellow]")


@app.command("train-intervals")
def train_intervals(config: str = "configs/default.yaml"):
    """Train conformalized quantile models for multi-horizon price bounds."""
    from trading_system.models.intervals import train_interval_models

    cfg = get_config(config)
    feat = pl.read_parquet(cfg.path("data_gold") / "features.parquet")
    feature_columns = [c for c in [
        "mom_5d", "mom_20d", "mom_60d", "mom_120d", "mom_12m1m",
        "vol_20d", "vol_60d", "rsi_14", "rel_vol_20",
        "sma_gap_50", "sma_gap_200", "breakout_20", "dd_from_high_60",
        "excess_ret_1d",
        # macro levels now available to the bound models too
        "macro_ust_10y", "macro_yield_curve", "macro_vix", "macro_hy_oas",
        "macro_vix_z_252", "macro_hy_oas_z_252",
    ] if c in feat.columns]

    bcfg = cfg.get("bounds", {}) or {}
    bundle = train_interval_models(
        feat, feature_columns,
        horizons=tuple(bcfg.get("horizons_days", (5, 21, 63, 126, 252))),
        quantiles=tuple(bcfg.get("quantiles", (0.05, 0.25, 0.5, 0.75, 0.95))),
        alpha=float(bcfg.get("conformal_alpha", 0.10)),
    )
    out = bundle.save(cfg.path("reports") / "models" / "intervals")
    rprint(f"[green]Interval bundle saved -> {out}[/green]")
    rprint("[bold]Out-of-sample coverage (target "
           f"{1 - bundle.alpha:.0%}):[/bold]")
    for h in bundle.horizons:
        cov = bundle.coverage.get(h)
        if cov is None:
            continue
        color = "green" if abs(cov - (1 - bundle.alpha)) < 0.05 else "yellow"
        rprint(f"  {h:>3}d: [{color}]{cov:.1%}[/{color}]  (conformal Q={bundle.conformal_q.get(h, 0):.4f})")


@app.command("train-forecast")
def train_forecast(
    config: str = "configs/default.yaml",
    horizons: str = typer.Option("21,63,126,252", help="comma-sep trading-day horizons"),
    n_splits: int = typer.Option(5, help="purged walk-forward folds"),
    groups: str = typer.Option("", help="feature-reserve groups (comma-sep); empty=all"),
    models: str = typer.Option(
        "", help="families: subset of lgbm,xgb,hist_gbm,ridge,lstm,gru,rnn "
                 "(empty = tabular default; this REPLACES the default, not adds)"),
    all_models: bool = typer.Option(
        False, "--all", help="train EVERY family incl. deep sequence models (needs torch/.[deep])"),
    lookback: int = typer.Option(64, help="lookback window for sequence models"),
    epochs: int = typer.Option(40, help="max epochs for sequence models"),
    neutralize: bool = typer.Option(
        False, "--neutralize", help="market-neutral target: learn pure stock-selection (demean per date)"),
    universe_weight: float = typer.Option(
        0.5, help="model-selection blend: 0=general perf only, 1=your-universe only"),
    priority_universe: str = typer.Option(
        "core", help="universe whose ICIR is weighted in selection (alias or path)"),
    cv: str = typer.Option(
        "walkforward", help="cross-validation: walkforward | cpcv (combinatorial purged)"),
):
    """Train + rigorously evaluate long-horizon forecasters; save best to models_store/.

    Purged+embargoed walk-forward CV, ranked by ICIR, gated by a label-shuffle
    leakage test. Also (re)trains the conformal interval bundle on the reserve.

    By default trains the 4 tabular families (lgbm, xgb, hist_gbm, ridge).
    Sequence models (lstm/gru/rnn) compete head-to-head under the same CV — add
    them with ``--all`` (every family) or an explicit ``--models`` list (which
    REPLACES the default). Deep models need ``pip install -e '.[deep]'`` and use
    CUDA/MPS automatically.
    """
    from rich.table import Table
    from rich.console import Console
    from trading_system.utils import get_compute_profile
    from trading_system.features.reserve import resolve_reserve
    from trading_system.models.forecast_train import train_all_horizons
    from trading_system.models.intervals import train_interval_models
    from trading_system.models.store import default_store, save_forecast_results, copy_interval_bundle

    cfg = get_config(config)
    prof = get_compute_profile()
    rprint(f"[cyan]compute:[/cyan] {prof.summary()}")

    feat = pl.read_parquet(cfg.path("data_gold") / "features.parquet")
    grp = [g.strip() for g in groups.split(",") if g.strip()] or None
    feat_cols = resolve_reserve(feat, groups=grp)
    rprint(f"[cyan]feature reserve:[/cyan] {len(feat_cols)} features"
           + (f" (groups: {grp})" if grp else " (all groups)"))
    if len(feat_cols) < 5:
        rprint("[red]Too few features resolved — run `ts features` first.[/red]")
        raise typer.Exit(1)

    hz = tuple(int(h) for h in horizons.split(",") if h.strip())
    if all_models:
        from trading_system.models.forecast_train import TABULAR_MODELS
        from trading_system.models.sequence import SEQUENCE_MODELS, torch_available
        mdl = list(TABULAR_MODELS) + list(SEQUENCE_MODELS)
        if not torch_available():
            rprint("[yellow]--all: torch not installed — sequence models will be skipped "
                   "(pip install -e '.[deep]'). Training tabular families only.[/yellow]")
    else:
        mdl = [m.strip() for m in models.split(",") if m.strip()] or None
    if mdl:
        rprint(f"[cyan]model families:[/cyan] {mdl}")

    # Priority universe for universe-weighted selection (the names you actually trade)
    priority: set[str] = set()
    if universe_weight > 0:
        try:
            pcfg = get_config(config).use_universe(priority_universe)
            priority = {t.upper() for t in pcfg["universe"]["tickers"]}
            rprint(f"[cyan]selection blend:[/cyan] {1-universe_weight:.0%} general + "
                   f"{universe_weight:.0%} your universe ('{priority_universe}', {len(priority)} names)")
        except Exception as e:
            rprint(f"[yellow]priority universe '{priority_universe}' unavailable ({e}); "
                   f"selecting on general performance.[/yellow]")
    if neutralize:
        rprint("[cyan]target:[/cyan] market-neutral (per-date demeaned) — pure cross-sectional selection")

    if cv == "cpcv":
        rprint("[cyan]CV:[/cyan] combinatorial purged (many paths → robust ICIR distribution)")
    results = train_all_horizons(
        feat, feat_cols, horizons=hz, n_splits=n_splits,
        models=mdl, lookback=lookback, seq_epochs=epochs,
        neutralize=neutralize, priority_tickers=(priority or None),
        universe_weight=universe_weight, cv_mode=cv,
    )
    if not results:
        rprint("[red]No horizons trained.[/red]")
        raise typer.Exit(1)

    # ── Summary table ────────────────────────────────────────────────────────
    console = Console()
    t = Table(title="Forecast models — purged walk-forward (blended ICIR selection)", show_lines=True)
    for c in ["Horizon", "Best", "ICIR", "univ", "Deflated", "IC", "Hit", "Leak", "Deflate"]:
        t.add_column(c, justify="right")
    for h, res in results.items():
        b = res.per_model.get(res.best_model_name, {})
        lk = res.leakage_gate.get("pass")
        lk_s = "[green]PASS[/green]" if lk else ("[red]FAIL[/red]" if lk is False else "—")
        df = res.deflation
        df_s = "[green]PASS[/green]" if df.get("pass") else ("[red]FAIL[/red]" if df.get("pass") is False else "—")
        icir = b.get("icir", 0)
        t.add_row(
            f"{h}d", res.best_model_name,
            f"[{'green' if icir>0.3 else 'yellow' if icir>0 else 'red'}]{icir:.2f}[/]",
            f"{b.get('univ_icir',0):.2f}", f"{df.get('deflated_icir',0):+.2f}",
            f"{b.get('ic_mean',0):+.3f}", f"{b.get('hit_rate',0):.1%}", lk_s, df_s,
        )
    console.print(t)

    store = default_store(cfg.project_root)
    save_forecast_results(results, store, compute_summary=prof.summary(), train_config={
        "neutralize": neutralize,
        "universe_weight": universe_weight,
        "priority_universe": priority_universe if priority else None,
        "n_features": len(feat_cols),
    })
    rprint(f"[green]Best models saved → {store}/forecast/[/green]")

    # ── Conformal interval bundle on the same reserve ────────────────────────
    try:
        bcfg = cfg.get("bounds", {}) or {}
        bundle = train_interval_models(
            feat, feat_cols,
            horizons=tuple(bcfg.get("horizons_days", (5, 21, 63, 126, 252))),
            quantiles=tuple(bcfg.get("quantiles", (0.05, 0.25, 0.5, 0.75, 0.95))),
            alpha=float(bcfg.get("conformal_alpha", 0.10)),
        )
        bundle.save(cfg.path("reports") / "models" / "intervals")
        copy_interval_bundle(cfg.path("reports") / "models", store)
        rprint(f"[green]Interval bundle trained + copied to store "
               f"(coverage { {h: round(c,2) for h,c in bundle.coverage.items()} }).[/green]")
    except Exception as e:
        rprint(f"[yellow]Interval bundle step skipped: {e}[/yellow]")


@app.command()
def bounds(
    ticker: str,
    config: str = "configs/default.yaml",
):
    """Show lower/median/upper price bounds for a ticker across horizons."""
    from rich.table import Table
    from rich.console import Console
    from trading_system.decision.analyze import _load_or_build_features
    from trading_system.decision.bounds import compute_bounds

    cfg = get_config(config)
    ticker = ticker.upper()
    ohlcv, features = _load_or_build_features(cfg)
    last_date = features["date"].max()
    row = features.filter((pl.col("ticker") == ticker) & (pl.col("date") == last_date))
    if row.is_empty():
        rprint(f"[red]No feature row for {ticker}.[/red]")
        raise typer.Exit(1)
    last_price = float(row["adj_close"][0]) if "adj_close" in row.columns else 0.0
    score = 0.0
    b = compute_bounds(cfg, ticker, features, ohlcv, last_price, score)
    if not b:
        rprint(f"[red]Could not compute bounds for {ticker}.[/red]")
        raise typer.Exit(1)

    console = Console()
    t = Table(title=f"{ticker} price bounds — {b.get('method')} (last ${last_price:.2f})")
    for col in ["Horizon", "Low", "Median", "High", "Low %", "Med %", "High %"]:
        t.add_column(col, justify="right")
    for label, h in b["horizons"].items():
        p, r = h["price"], h["return"]
        t.add_row(
            label, f"${p['lo']:.2f}", f"${p['median']:.2f}", f"${p['hi']:.2f}",
            f"{r['lo']*100:+.1f}%", f"{r['median']*100:+.1f}%", f"{r['hi']*100:+.1f}%",
        )
    console.print(t)


@app.command()
def picks(
    config: str = "configs/default.yaml",
    horizon: int = typer.Option(252, help="forecast horizon in trading days (5/21/63/126/252)"),
    top: int = typer.Option(20, help="how many top-ranked names to plan"),
    min_score: float = typer.Option(0.0, help="minimum forecast score to include"),
    write: bool = typer.Option(False, "--write", help="also write reports/picks/*.md + .json"),
    universe: str = UNIVERSE_OPT,
):
    """Long-term buy plan: what to buy, at what entry / target / stop, ranked.

    Ranks the universe by the committed models_store/ forecaster at ``horizon``
    and attaches each name's calibrated price band (entry, add-on-dip, median &
    stretch targets, conformal stop, reward/risk, timing).
    """
    from rich.table import Table
    from rich.console import Console
    from trading_system.decision.longterm import build_longterm_picks, write_picks

    cfg = get_config(config).use_universe(universe)
    try:
        plan = build_longterm_picks(cfg, horizon=horizon, top_n=top, min_score=min_score)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    lk = plan["model_leak_pass"]
    lk_s = "[green]leak-gate PASS[/green]" if lk else ("[red]leak-gate FAIL — treat as low-confidence[/red]" if lk is False else "leak-gate —")
    rprint(f"[bold]Long-term picks · {plan['horizon_label']} · model={plan['model']}[/bold] "
           f"· as of {plan['as_of']} · {lk_s}")
    if not plan["picks"]:
        rprint("[yellow]No names cleared the score threshold.[/yellow]")
        raise typer.Exit()

    console = Console()
    t = Table(show_lines=False)
    for c in ["#", "Ticker", "Score", "Entry", "Add≤", "Median", "Stretch", "Stop", "Up%", "Dn%", "R/R", "Timing"]:
        t.add_column(c, justify="right")
    for i, p in enumerate(plan["picks"], 1):
        rr = p["reward_risk"]
        rr_c = "green" if (rr and rr >= 2) else ("yellow" if rr and rr >= 1 else "red")
        t.add_row(
            str(i), f"[bold]{p['ticker']}[/bold]", f"{p['score']:+.4f}",
            f"${p['entry']:.2f}", f"${p['add_on_dip_below']:.2f}", f"${p['median_target']:.2f}",
            f"${p['stretch_target']:.2f}", f"${p['invalidation_stop']:.2f}",
            f"[green]{p['upside_pct']*100:+.0f}%[/green]", f"[red]{p['downside_pct']*100:+.0f}%[/red]",
            f"[{rr_c}]{rr if rr is not None else '—'}[/{rr_c}]", p["timing"],
        )
    console.print(t)
    if write:
        md = write_picks(cfg, plan)
        rprint(f"[green]Wrote {md}[/green]")


@app.command()
def complexity(
    ticker: str,
    config: str = "configs/default.yaml",
    universe: str = UNIVERSE_OPT,
):
    """Nonlinear-dynamics fingerprint of a ticker — chaos, fractal, entropy, bubbles.

    Computes every estimator in the nonlinear library on the latest trailing
    windows and prints an interpreted, domain-grouped read of the price's
    mathematical character (long memory, complexity, chaoticity, tail risk,
    early-warning / bubble signatures).
    """
    from rich.table import Table
    from rich.console import Console
    from trading_system.features.nonlinear_report import fingerprint, synthesis

    cfg = get_config(config).use_universe(universe)
    ticker = ticker.upper()
    bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
    if not bronze.exists():
        rprint("[red]No OHLCV — run `ts ingest` first.[/red]")
        raise typer.Exit(1)
    df = pl.read_parquet(bronze).filter(pl.col("ticker") == ticker).sort("date")
    if df.height < 120:
        rprint(f"[red]Need ≥120 days of history for {ticker} (have {df.height}).[/red]")
        raise typer.Exit(1)
    close = df["adj_close"].to_numpy()
    rows = fingerprint(close)

    console = Console()
    last_d = df["date"].max()
    t = Table(title=f"{ticker} · nonlinear fingerprint (through {last_d}, {df.height}d history)",
              show_lines=False)
    for c, j in [("Metric", "left"), ("Borrowed from", "left"), ("Value", "right"), ("Reading", "left")]:
        t.add_column(c, justify=j)
    cur = None
    for r in rows:
        if r["domain"] != cur:
            cur = r["domain"]
            t.add_row(f"[bold cyan]{cur}[/bold cyan]", "", "", "")
        v = r["value"]
        vs = f"{v:+.3f}" if isinstance(v, float) and v == v else "n/a"
        reading = r["reading"]
        style = "red" if r["flag"] else ""
        t.add_row(f"  {r['metric']}", f"[dim]{r['source']}[/dim]", vs,
                  f"[{style}]{reading}[/{style}]" if style else reading)
    console.print(t)
    rprint(f"\n[bold]Synthesis:[/bold] {synthesis(rows)}")


@app.command()
def daily(config: str = "configs/default.yaml", universe: str = UNIVERSE_OPT):
    """Run the full daily pipeline."""
    cfg = get_config(config).use_universe(universe)
    path = run_daily_pipeline(cfg)
    rprint(f"[green]Daily report: {path}[/green]")


@app.command("paper-trade")
def paper_trade(
    config: str = "configs/default.yaml",
    backfill: bool = typer.Option(False, "--backfill", help="Replay all history from OOS predictions"),
    start_date: str = typer.Option("2015-01-01", "--start-date", help="Backfill start date (YYYY-MM-DD)"),
):
    """Execute daily paper trades based on current model decisions.

    Use --backfill to replay the full history from OOS predictions
    (use after ts train to build a historical track record).
    """
    from datetime import date as _date
    from .execution.paper_portfolio import PaperPortfolio

    cfg = get_config(config)
    gold = cfg.path("data_gold")
    journal = gold / "paper_portfolio_journal.json"
    equity_log = gold / "paper_equity_log.parquet"

    portfolio = PaperPortfolio(
        journal_path=journal,
        equity_log_path=equity_log,
        initial_cash=cfg["backtest"].get("initial_cash", 100_000.0),
    )

    if backfill:
        preds_path = gold / "predictions.parquet"
        features_path = gold / "features.parquet"
        if not preds_path.exists():
            rprint("[red]No predictions found — run `ts train` first.[/red]")
            raise typer.Exit(1)
        rprint(f"[bold]Backfilling paper portfolio from {start_date}…[/bold]")
        days = portfolio.backfill_from_predictions(
            features_path=features_path,
            predictions_path=preds_path,
            start_date=start_date,
        )
        rprint(f"[green]Backfill complete: {days} trading days replayed.[/green]")
    else:
        # Live: run analyze_all, then process decisions
        rprint("[bold]Running analyze_all for today's decisions…[/bold]")
        results = analyze_all(cfg)
        # Build price map from latest features
        feat_path = gold / "features.parquet"
        if feat_path.exists():
            feat = pl.read_parquet(feat_path)
            last_date = feat["date"].max()
            prices = {
                r["ticker"]: float(r["adj_close"])
                for r in feat.filter(pl.col("date") == last_date).to_dicts()
                if r.get("adj_close")
            }
        else:
            prices = {}
        orders = portfolio.process_decisions(results, prices)
        snap = portfolio.snapshot(_date.today(), prices)
        rprint(f"[green]Paper trade complete: {len(orders)} orders, equity={snap['equity']:,.0f}[/green]")

    # Always print status summary
    _print_paper_status(portfolio)


@app.command("paper-status")
def paper_status(config: str = "configs/default.yaml"):
    """Show paper portfolio status: equity, holdings, and horizon PnL."""
    from .execution.paper_portfolio import PaperPortfolio

    cfg = get_config(config)
    gold = cfg.path("data_gold")
    portfolio = PaperPortfolio(
        journal_path=gold / "paper_portfolio_journal.json",
        equity_log_path=gold / "paper_equity_log.parquet",
    )
    _print_paper_status(portfolio)


def _print_paper_status(portfolio) -> None:
    """Print a Rich table summary of the paper portfolio."""
    from rich.table import Table
    from rich.console import Console

    # Load latest prices for MTM
    feat_path = Path("data/gold/features.parquet")
    prices: dict[str, float] = {}
    if feat_path.exists():
        feat = pl.read_parquet(feat_path)
        last_date = feat["date"].max()
        prices = {
            r["ticker"]: float(r["adj_close"])
            for r in feat.filter(pl.col("date") == last_date).to_dicts()
            if r.get("adj_close")
        }

    summary = portfolio.summary(prices)
    console = Console()

    # Main stats
    equity = summary["equity"]
    pnl_total = summary.get("pnl_total", 0.0) or 0.0
    color = "green" if pnl_total >= 0 else "red"
    console.print(f"\n[bold]Paper Portfolio Status[/bold]")
    console.print(f"  Equity:      [bold]{equity:>12,.2f}[/bold]")
    console.print(f"  Total PnL:   [{color}]{pnl_total * 100:>+.2f}%[/{color}]")
    console.print(f"  Cash:        {summary['cash']:>12,.2f}")
    console.print(f"  Positions:   {summary['n_positions']}")
    console.print(f"  Trades:      {summary['n_trades']}")
    if summary.get("win_rate") is not None:
        console.print(f"  Win Rate:    {summary['win_rate'] * 100:.1f}%")

    # Horizon PnL
    horizons = Table(title="Horizon PnL", show_lines=False)
    horizons.add_column("Horizon")
    horizons.add_column("Return", justify="right")
    for label in ["pnl_1m", "pnl_3m", "pnl_6m", "pnl_1y"]:
        v = summary.get(label)
        h_label = label.replace("pnl_", "")
        if v is None:
            horizons.add_row(h_label, "—")
        else:
            c = "green" if v >= 0 else "red"
            horizons.add_row(h_label, f"[{c}]{v * 100:+.2f}%[/{c}]")
    console.print(horizons)

    # Holdings table
    holdings = summary.get("holdings", {})
    active = {t: q for t, q in holdings.items() if q > 0.001}
    if active:
        htable = Table(title="Current Holdings", show_lines=False)
        htable.add_column("Ticker")
        htable.add_column("Qty", justify="right")
        htable.add_column("Price", justify="right")
        htable.add_column("Value", justify="right")
        for ticker, qty in sorted(active.items()):
            px = prices.get(ticker, 0.0)
            value = qty * px
            htable.add_row(ticker, f"{qty:.2f}", f"{px:.2f}", f"{value:,.0f}")
        console.print(htable)


@app.command()
def dashboard(config: str = "configs/default.yaml"):
    """Launch the Streamlit dashboard."""
    import subprocess, sys
    proj = get_config(config).project_root
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(proj / "scripts" / "dashboard.py")])


@app.command()
def universe(config: str = "configs/default.yaml", universe: str = UNIVERSE_OPT):
    """Print the configured universe."""
    cfg = get_config(config).use_universe(universe)
    u = cfg["universe"]
    rprint(f"[bold]Universe:[/bold] {u.get('name')} (benchmark={u.get('benchmark')})")
    rprint(f"[bold]Required ({len(u.get('required', []))}):[/bold] {', '.join(u.get('required', []))}")
    rprint(f"[bold]Additions ({len(u.get('additions', []))}):[/bold] {', '.join(u.get('additions', []))}")
    rprint(f"[bold]Total tradeable:[/bold] {len(u['tickers'])}")


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Symbol to analyze (e.g. GOOGL)"),
    config: str = "configs/default.yaml",
    no_report: bool = typer.Option(False, "--no-report", help="Skip writing markdown/JSON"),
    universe: str = UNIVERSE_OPT,
):
    """Run the full decision pipeline on a single symbol and write a report."""
    cfg = get_config(config).use_universe(universe)
    res = analyze_symbol(ticker.upper(), cfg=cfg, write_report=not no_report)
    color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(res.stance, "white")
    rprint(
        f"[{color}][bold]{res.ticker}[/bold] -> {res.stance}[/{color}] "
        f"(conf {res.confidence:.2f}, score_src={res.score_source})"
    )
    rprint(f"  5d forecast:  {res.forecast_5d * 100:+.2f}%")
    rprint(f"  20d forecast: {res.forecast_20d * 100:+.2f}%")
    rprint("  rationale:")
    for r in res.rationale:
        rprint(f"   - {r}")
    if res.report_path:
        rprint(f"  report: {res.report_path}")
        rprint(f"  json:   {res.json_path}")


@app.command("analyze-all")
def analyze_all_cmd(
    config: str = "configs/default.yaml",
    universe: str = UNIVERSE_OPT,
    workers: int = typer.Option(6, help="concurrent workers (1 = serial); IV+LLM bound"),
):
    """Run the decision pipeline across the entire configured universe (threaded)."""
    cfg = get_config(config).use_universe(universe)
    results = analyze_all(cfg, workers=workers)
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for r in results:
        counts[r.stance] = counts.get(r.stance, 0) + 1
    rprint(f"[bold]Analyzed {len(results)} symbols.[/bold] {counts}")
    # Show top BUY signals by confidence
    buys = sorted(
        [r for r in results if r.stance == "BUY"],
        key=lambda r: r.confidence,
        reverse=True,
    )
    if buys:
        rprint("[green]Top BUY signals (by confidence):[/green]")
        for r in buys[:15]:
            rprint(f"  {r.ticker}  conf={r.confidence:.2f}  5d={r.forecast_5d * 100:+.2f}%")


@app.command()
def explain(
    report: str = typer.Argument(..., help="Path to a reports/decisions/<TICKER>_<stamp>.md file"),
    model: str = typer.Option(DEEPSEEK_DEFAULT_MODEL, "--model", help="DeepSeek model name"),
):
    """Explain a decision report in plain English using DeepSeek V4."""
    from pathlib import Path
    import os

    # Resolve relative paths from project root
    cfg = get_config()
    p = Path(report)
    if not p.is_absolute():
        p = cfg.project_root / p
    if not p.exists():
        # Last-ditch: glob for the latest report for a ticker name
        candidates = sorted((cfg.project_root / "reports" / "decisions").glob(f"{report}*.md"))
        if candidates:
            p = candidates[-1]
            rprint(f"[dim]Resolved to: {p}[/dim]")
        else:
            rprint(f"[red]Report not found: {report}[/red]")
            raise typer.Exit(1)

    rprint(f"[bold]Explaining:[/bold] {p.name}  [dim](model={model})[/dim]\n")
    text = explain_report(p, api_key=os.environ.get("DEEPSEEK_API_KEY"), model=model)
    rprint(text)


# ── Future prediction commands ────────────────────────────────────────────────

@app.command("future-predict")
def future_predict(
    config: str = "configs/default.yaml",
    budget: float = typer.Option(10_000.0, "--budget", help="Starting budget in USD"),
    date_override: str = typer.Option("", "--date", help="Override prediction date (YYYY-MM-DD)"),
):
    """Create a new forward-looking forecast session with a $10k portfolio.

    Scores all universe tickers, allocates up to 60% of budget into top BUY
    signals (≤10% each), and saves the session to future_predict/YYYY-MM-DD/.
    Run `ts future-status` any time to see how the predictions are tracking.
    """
    from datetime import date as _dt
    from rich.table import Table
    from rich.console import Console
    from .future_predict.forecast import run_forecast, list_sessions
    from .models.model_registry import load_model, best_ensemble_artifact, list_models

    cfg = get_config(config)
    console = Console()

    pred_date = _dt.fromisoformat(date_override) if date_override else _dt.today()

    base_dir = cfg.project_root / "future_predict"
    session_dir = base_dir / pred_date.isoformat()

    if session_dir.exists() and (session_dir / "forecast.json").exists():
        rprint(f"[yellow]Session for {pred_date} already exists at {session_dir}.[/yellow]")
        rprint("[dim]Use `ts future-status` to view it, or pick a different --date.[/dim]")
        raise typer.Exit(0)

    # Load ensemble model
    reg_path = cfg.path("reports") / "models"
    models = list_models(reg_path)
    if not models:
        rprint("[red]No trained model found — run `ts train` first.[/red]")
        raise typer.Exit(1)

    art_name = best_ensemble_artifact(reg_path)
    model, art = load_model(art_name, registry=reg_path)
    feature_columns = art.feature_columns
    best_variant = art.metadata.get("best_variant", "ensemble_blend")

    rprint(f"[bold]Running future forecast[/bold] for [cyan]{pred_date}[/cyan] "
           f"using [cyan]{art_name}[/cyan] (variant={best_variant})")
    rprint(f"Budget: [green]${budget:,.0f}[/green] — max deploy 60%, max 10% per position")

    features_path = cfg.path("data_gold") / "features.parquet"
    ohlcv_path    = cfg.path("data_bronze") / "ohlcv_daily.parquet"

    forecast = run_forecast(
        session_dir=session_dir,
        features_path=features_path,
        ohlcv_path=ohlcv_path,
        model=model,
        feature_columns=feature_columns,
        best_variant=best_variant,
        model_name=art_name,
        budget=budget,
        prediction_date=pred_date,
    )

    # Print summary
    port = forecast["portfolio"]
    positions = port["positions"]
    n_pos = len(positions)

    console.print(f"\n[bold green]Forecast session created:[/bold green] {session_dir}")
    console.print(f"  Tickers scored:  {len(forecast['all_predictions'])}")
    console.print(f"  Positions taken: {n_pos}")
    console.print(f"  Deployed:        ${port['deployed']:,.2f}  "
                  f"({port['deployed'] / budget * 100:.0f}% of budget)")
    console.print(f"  Cash reserved:   ${port['cash_reserved']:,.2f}  "
                  f"({port['cash_reserved'] / budget * 100:.0f}% liquid)")

    # Horizon target dates
    htable = Table(title="Horizon Target Dates", show_lines=False)
    htable.add_column("Horizon")
    htable.add_column("Target Date")
    for label, tdate in forecast["horizons"].items():
        htable.add_row(label, tdate)
    console.print(htable)

    # Positions table
    if positions:
        ptable = Table(title="Allocated Positions", show_lines=False)
        ptable.add_column("Ticker")
        ptable.add_column("Score", justify="right")
        ptable.add_column("Entry $", justify="right")
        ptable.add_column("Shares", justify="right")
        ptable.add_column("Allocated $", justify="right")
        for p in positions:
            ptable.add_row(
                p["ticker"],
                f"{p['score']:+.4f}",
                f"{p['entry_price']:.2f}",
                f"{p['shares']:.3f}",
                f"${p['allocated']:,.0f}",
            )
        console.print(ptable)

    rprint(f"\n[dim]Run [bold]ts future-update[/bold] daily (or via ts daily) to track MTM equity.[/dim]")
    rprint(f"[dim]Run [bold]ts future-status[/bold] any time to see P&L and prediction accuracy.[/dim]")


@app.command("future-status")
def future_status(
    config: str = "configs/default.yaml",
    session_date: str = typer.Option("", "--date", help="Session date YYYY-MM-DD (default: latest)"),
    all_sessions: bool = typer.Option(False, "--all", help="Show all sessions"),
):
    """Show status and prediction accuracy of a future forecast session."""
    from rich.table import Table
    from rich.console import Console
    from .future_predict.forecast import (
        list_sessions, update_session_equity, evaluate_predictions,
    )
    import json as _json

    cfg = get_config(config)
    console = Console()
    base_dir = cfg.project_root / "future_predict"
    ohlcv_path = cfg.path("data_bronze") / "ohlcv_daily.parquet"

    sessions = list_sessions(base_dir)
    if not sessions:
        rprint("[yellow]No future-predict sessions found. Run `ts future-predict` first.[/yellow]")
        raise typer.Exit(0)

    if all_sessions:
        # Summary table of all sessions
        stable = Table(title="All Future-Predict Sessions", show_lines=True)
        stable.add_column("Date")
        stable.add_column("Model")
        stable.add_column("Positions", justify="right")
        stable.add_column("Deployed $", justify="right")
        stable.add_column("Current Equity", justify="right")
        stable.add_column("Return", justify="right")
        for s in sessions:
            fc = _json.loads((s / "forecast.json").read_text())
            try:
                snap = update_session_equity(s, ohlcv_path)
                eq = snap["equity"]
                ret = snap["return_pct"]
                color = "green" if ret >= 0 else "red"
                eq_str  = f"${eq:,.0f}"
                ret_str = f"[{color}]{ret * 100:+.2f}%[/{color}]"
            except Exception:
                eq_str = ret_str = "—"
            stable.add_row(
                s.name,
                fc.get("model", "?")[:30],
                str(len(fc["portfolio"]["positions"])),
                f"${fc['portfolio']['deployed']:,.0f}",
                eq_str, ret_str,
            )
        console.print(stable)
        return

    # Single session
    if session_date:
        session_dir = base_dir / session_date
        if not session_dir.exists():
            rprint(f"[red]Session {session_date} not found.[/red]")
            raise typer.Exit(1)
    else:
        session_dir = sessions[0]

    fc = _json.loads((session_dir / "forecast.json").read_text())
    rprint(f"\n[bold]Future-Predict Session:[/bold] [cyan]{session_dir.name}[/cyan]")
    rprint(f"  Model: {fc['model']} | Variant: {fc['best_variant']}")
    rprint(f"  Prices as-of: {fc['prices_as_of']}")

    # Current equity MTM
    try:
        snap = update_session_equity(session_dir, ohlcv_path)
        initial = fc["budget"]
        eq  = snap["equity"]
        ret = snap["return_pct"]
        color = "green" if ret >= 0 else "red"
        console.print(f"\n[bold]Portfolio (${initial:,.0f} budget)[/bold]")
        console.print(f"  Current Equity:  [bold]${eq:,.2f}[/bold]  "
                      f"([{color}]{ret * 100:+.2f}%[/{color}])")
        console.print(f"  Deployed MTM:    ${snap['deployed_mtm']:,.2f}")
        console.print(f"  Cash Reserved:   ${snap['cash']:,.2f}")
        console.print(f"  Prices as-of:    {snap['prices_as_of']}")
    except Exception as exc:
        rprint(f"[yellow]Could not refresh equity: {exc}[/yellow]")

    # Horizon target dates
    htable = Table(title="Forecast Horizons", show_lines=False)
    htable.add_column("Horizon")
    htable.add_column("Target Date")
    htable.add_column("Days to Go", justify="right")
    from datetime import date as _dt
    today = _dt.today()
    for label, tdate in fc["horizons"].items():
        td = _dt.fromisoformat(tdate)
        days_left = (td - today).days
        status = f"{days_left}d" if days_left > 0 else f"[green]{abs(days_left)}d ago[/green]"
        htable.add_row(label, tdate, status)
    console.print(htable)

    # Positions table with live P&L
    positions = fc["portfolio"]["positions"]
    if positions:
        try:
            ohlcv = pl.read_parquet(ohlcv_path)
            latest_px = ohlcv["date"].max()
            live_prices = {
                r["ticker"]: float(r["adj_close"])
                for r in ohlcv.filter(pl.col("date") == latest_px)
                               .select(["ticker", "adj_close"]).to_dicts()
            }
        except Exception:
            live_prices = {}

        pos_table = Table(title="Positions (live MTM)", show_lines=False)
        pos_table.add_column("Ticker")
        pos_table.add_column("Score", justify="right")
        pos_table.add_column("Entry $", justify="right")
        pos_table.add_column("Now $", justify="right")
        pos_table.add_column("Return", justify="right")
        pos_table.add_column("Value $", justify="right")
        for p in positions:
            entry = p["entry_price"]
            now   = live_prices.get(p["ticker"], entry)
            ret   = (now - entry) / entry if entry else 0.0
            val   = p["shares"] * now
            c     = "green" if ret >= 0 else "red"
            pos_table.add_row(
                p["ticker"],
                f"{p['score']:+.4f}",
                f"{entry:.2f}",
                f"{now:.2f}",
                f"[{c}]{ret * 100:+.2f}%[/{c}]",
                f"${val:,.0f}",
            )
        console.print(pos_table)

    # Prediction accuracy for elapsed horizons
    eval_results = evaluate_predictions(session_dir, ohlcv_path)
    elapsed = {k: v for k, v in eval_results.items() if v.get("status") == "available"}
    if elapsed:
        etable = Table(title="Prediction Accuracy (elapsed horizons)", show_lines=True)
        etable.add_column("Horizon")
        etable.add_column("Actual Date")
        etable.add_column("Hit Rate", justify="right")
        etable.add_column("Mean Return", justify="right")
        etable.add_column("Tickers", justify="right")
        for label, res in elapsed.items():
            hr  = res["hit_rate"]
            mr  = res["mean_return"]
            hrc = "green" if (hr or 0) >= 0.5 else "red"
            mrc = "green" if (mr or 0) >= 0 else "red"
            etable.add_row(
                label,
                res["actual_date"],
                f"[{hrc}]{hr * 100:.1f}%[/{hrc}]" if hr is not None else "—",
                f"[{mrc}]{mr * 100:+.2f}%[/{mrc}]" if mr is not None else "—",
                str(res["total_tickers"]),
            )
        console.print(etable)
    else:
        rprint("[dim]No horizons have elapsed yet — check back later.[/dim]")


@app.command("future-update")
def future_update(config: str = "configs/default.yaml"):
    """Update equity snapshots for all active future-predict sessions.

    Also attempts to redeploy dry-powder cash when new high-quality signals appear.
    Automatically called by `ts daily`. Safe to run at any time.
    """
    from .future_predict.forecast import (
        list_sessions, update_session_equity, redeploy_cash,
    )
    from .models.model_registry import load_model, best_ensemble_artifact, list_models

    cfg = get_config(config)
    base_dir   = cfg.project_root / "future_predict"
    ohlcv_path = cfg.path("data_bronze") / "ohlcv_daily.parquet"
    feat_path  = cfg.path("data_gold") / "features.parquet"
    sessions   = list_sessions(base_dir)

    if not sessions:
        rprint("[dim]No future-predict sessions to update.[/dim]")
        return

    # Load model once for redeployment (optional — gracefully skip if unavailable)
    _model = _feat_cols = _variant = None
    try:
        reg_path = cfg.path("reports") / "models"
        if list_models(reg_path) and feat_path.exists():
            art_name = best_ensemble_artifact(reg_path)
            _model, art = load_model(art_name, registry=reg_path)
            _feat_cols = art.feature_columns
            _variant   = art.metadata.get("best_variant", "ensemble_blend")
    except Exception:
        pass

    updated = 0
    for s in sessions:
        try:
            snap = update_session_equity(s, ohlcv_path)
            ret   = snap["return_pct"]
            color = "green" if ret >= 0 else "red"
            rprint(f"  [{color}]{s.name}[/{color}]  equity=${snap['equity']:,.0f}  "
                   f"return=[{color}]{ret * 100:+.2f}%[/{color}]")

            # Redeploy dry-powder cash when possible
            if _model is not None:
                result = redeploy_cash(
                    session_dir=s,
                    features_path=feat_path,
                    ohlcv_path=ohlcv_path,
                    model=_model,
                    feature_columns=_feat_cols,
                    best_variant=_variant,
                )
                if result["redeployed"] > 0:
                    new_tickers = [p["ticker"] for p in result["new_positions"]]
                    rprint(
                        f"    [cyan]Redeployed ${result['redeployed']:,.0f} → "
                        f"{new_tickers}  (cash left=${result['cash_remaining']:,.0f})[/cyan]"
                    )
            updated += 1
        except Exception as exc:
            rprint(f"  [yellow]{s.name}: skipped ({exc})[/yellow]")

    rprint(f"[green]Updated {updated} future-predict session(s).[/green]")


# ── V2: Agent commands ────────────────────────────────────────────────────────

@app.command("agent-analyze")
def agent_analyze(
    ticker: str = typer.Argument(..., help="Symbol to analyze (e.g. AAPL)"),
    config: str = "configs/default.yaml",
    save: bool = typer.Option(True, "--save/--no-save", help="Save result JSON to reports/agent/"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full thought/action chain"),
):
    """Run the ReAct LLM agent for deep analysis of a single ticker.

    The agent autonomously fetches news, gets the ML model score, computes
    SHAP attribution, checks apprehension, live price, and economic calendar
    before synthesizing a BUY/HOLD/SELL recommendation.

    Uses DeepSeek cloud API with Ollama local fallback.
    """
    from rich.panel import Panel
    from rich.console import Console
    from .agent import TradingAgentOrchestrator

    cfg = get_config(config)
    console = Console()

    rprint(f"[bold]Running agent analysis for [cyan]{ticker.upper()}[/cyan]…[/bold]")
    orchestrator = TradingAgentOrchestrator(cfg)
    result = orchestrator.run_ticker_analysis(ticker)

    if verbose and result.steps:
        for i, step in enumerate(result.steps, 1):
            console.print(Panel(
                f"[dim]{step.thought}[/dim]",
                title=f"Step {i} — Thought",
                border_style="dim",
            ))
            if step.action and step.action != "FINISH":
                console.print(f"  [cyan]Action:[/cyan] {step.action}({step.action_input[:80]})")
                console.print(f"  [yellow]Observation:[/yellow] {step.observation[:300]}")
        rprint()

    color = "green" if result.success else "yellow"
    rprint(Panel(
        result.final_answer or "No answer generated.",
        title=f"[{color}]Agent Analysis: {ticker.upper()} (backend={result.backend_used})[/{color}]",
        border_style=color,
    ))

    if save and result.success:
        path = orchestrator.save_result(result)
        rprint(f"[dim]Saved → {path}[/dim]")


@app.command("agent-portfolio")
def agent_portfolio(
    config: str = "configs/default.yaml",
    save: bool = typer.Option(True, "--save/--no-save", help="Save result JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the ReAct agent to review the current paper portfolio.

    The agent evaluates each held position using model scores, news sentiment,
    and apprehension to recommend HOLD / ADD / EXIT actions.
    """
    from rich.panel import Panel
    from rich.console import Console
    from .agent import TradingAgentOrchestrator

    cfg = get_config(config)
    console = Console()

    rprint("[bold]Running agent portfolio review…[/bold]")
    orchestrator = TradingAgentOrchestrator(cfg)
    results = orchestrator.run_portfolio_review()

    if not results:
        rprint("[yellow]Portfolio is empty or no broker state found.[/yellow]")
        return

    for result in results:
        color = "green" if result.success else "yellow"
        if verbose and result.steps:
            for i, step in enumerate(result.steps, 1):
                console.print(f"[dim]Step {i}[/dim] {step.action}: {step.observation[:200]}")

        rprint(Panel(
            result.final_answer or "No answer generated.",
            title=f"[{color}]Portfolio Review (backend={result.backend_used})[/{color}]",
            border_style=color,
        ))

        if save and result.success:
            path = orchestrator.save_result(result)
            rprint(f"[dim]Saved → {path}[/dim]")


@app.command("agent-briefing")
def agent_briefing(
    config: str = "configs/default.yaml",
    save: bool = typer.Option(True, "--save/--no-save", help="Save result JSON"),
):
    """Generate a daily market briefing via the ReAct agent.

    The agent checks the economic calendar, reviews top model scores,
    and synthesizes a morning market narrative with top picks and risks.
    """
    from rich.panel import Panel
    from .agent import TradingAgentOrchestrator

    cfg = get_config(config)
    rprint("[bold]Generating daily market briefing…[/bold]")

    orchestrator = TradingAgentOrchestrator(cfg)
    result = orchestrator.run_daily_briefing()

    color = "green" if result.success else "yellow"
    rprint(Panel(
        result.final_answer or "No briefing generated.",
        title=f"[{color}]Daily Briefing (backend={result.backend_used})[/{color}]",
        border_style=color,
    ))

    if save and result.success:
        path = orchestrator.save_result(result)
        rprint(f"[dim]Saved → {path}[/dim]")


@app.command("signals")
def signals_cmd(
    config: str = "configs/default.yaml",
    stance: str = typer.Option("ALL", "--stance", "-s", help="ALL | BUY | SELL | HOLD"),
    min_conf: float = typer.Option(0.0, "--min-conf", help="Minimum confidence (0-1)"),
    top: int = typer.Option(0, "--top", "-n", help="Show only top N rows (0 = all)"),
):
    """Print the latest BUY/SELL/HOLD signal table for the whole universe.

    Reads the most recent decision JSON for each ticker from reports/decisions/
    and displays a colour-coded terminal table.
    """
    import json
    from pathlib import Path
    from rich.table import Table
    from rich.console import Console

    cfg = get_config(config)
    decisions_dir = cfg.path("reports") / "decisions"
    if not decisions_dir.exists():
        rprint("[yellow]No decision reports found. Run `ts analyze TICKER` first.[/yellow]")
        return

    json_files = sorted(decisions_dir.glob("*.json"), reverse=True)
    latest: dict[str, dict] = {}
    for jf in json_files:
        t = jf.name.split("_")[0]
        if t not in latest:
            try:
                latest[t] = json.loads(jf.read_text())
            except Exception:
                pass

    rows = sorted(latest.values(), key=lambda d: d.get("confidence", 0), reverse=True)

    # Apply filters
    if stance.upper() != "ALL":
        rows = [r for r in rows if r.get("stance", "") == stance.upper()]
    if min_conf > 0:
        rows = [r for r in rows if r.get("confidence", 0) >= min_conf]
    if top > 0:
        rows = rows[:top]

    table = Table(title="Trade Signals", show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("Signal", width=7)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("5d Fcst", justify="right", width=9)
    table.add_column("20d Fcst", justify="right", width=9)
    table.add_column("As Of", width=12)

    for d in rows:
        s = d.get("stance", "HOLD")
        color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(s, "white")
        conf = d.get("confidence", 0)
        f5 = d.get("forecast_5d") or 0
        f20 = d.get("forecast_20d") or 0
        table.add_row(
            d.get("ticker", ""),
            f"[{color}]{s}[/{color}]",
            f"{conf:.0%}",
            f"{f5*100:+.2f}%",
            f"{f20*100:+.2f}%",
            d.get("as_of", ""),
        )

    buy_n = sum(1 for d in latest.values() if d.get("stance") == "BUY")
    hold_n = sum(1 for d in latest.values() if d.get("stance") == "HOLD")
    sell_n = sum(1 for d in latest.values() if d.get("stance") == "SELL")

    Console().print(table)
    rprint(
        f"\n[green]BUY: {buy_n}[/green]  [yellow]HOLD: {hold_n}[/yellow]  "
        f"[red]SELL: {sell_n}[/red]  (total {len(latest)} tickers)"
    )


# ── V3: Decision-tree playbook commands (flags / brief / playbook / check) ───

def _flag_color_style(color: str) -> str:
    return {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}.get(color, "magenta")


@app.command("flags")
def flags_cmd(
    config: str = "configs/default.yaml",
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Force live lookups (ignore cache)"),
    as_json: bool = typer.Option(False, "--json", help="Print raw snapshot JSON"),
):
    """Live lookup of the five flags (O/F/I/S/C) + composite deployment rule.

    O Brent · F Fed · I core CPI m/m · S NDX level · C AI capex.
    Quantitative flags are fetched live (yfinance + FRED, keyless); qualitative
    ones (F tone, C) come from configs/flag_overrides.yaml.
    """
    import json as _json
    from rich.table import Table
    from rich.console import Console
    from .flags import get_flag_snapshot, FLAG_ORDER

    cfg = get_config(config)
    snap = get_flag_snapshot(
        cfg, refresh=refresh,
        max_age_minutes=float(cfg.get("playbook", {}).get("flag_cache_minutes", 60)),
    )
    if as_json:
        rprint(_json.dumps(snap.to_dict(), indent=2, default=str))
        return

    table = Table(title=f"Flag Board — {snap.as_of}", show_lines=True)
    table.add_column("Flag", style="bold")
    table.add_column("Color")
    table.add_column("Value", justify="right")
    table.add_column("Reading")
    table.add_column("Source", style="dim")
    for f in FLAG_ORDER:
        r = snap.readings[f]
        c = _flag_color_style(r.color.value)
        stale = " ⚠stale" if r.stale else ""
        table.add_row(
            f"{f} · {r.name}",
            f"[{c}]{r.color.value}[/{c}]{stale}",
            str(r.value) if r.value is not None else "—",
            r.detail,
            r.source,
        )
    Console().print(table)

    comp = snap.composite
    cc = _flag_color_style(comp.color.value)
    rprint(f"\n[bold]COMPOSITE: [{cc}]{comp.color.value}[/{cc}][/bold] — {comp.rationale}")
    if comp.semi_freeze:
        rprint("[red bold]SEMI FREEZE — no semi/semicap buys, period.[/red bold]")
    for w in comp.data_warnings:
        rprint(f"[yellow]⚠ {w}[/yellow]")


@app.command("brief")
def brief_cmd(
    config: str = "configs/default.yaml",
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Force fresh flag lookups"),
    no_write: bool = typer.Option(False, "--no-write", help="Print only, skip writing files"),
):
    """Morning briefing: flags, catalysts, standing rules, cycle rules, caps."""
    from rich.markdown import Markdown
    from rich.console import Console
    from .flags import get_flag_snapshot
    from .playbook import build_briefing, render_markdown, write_briefing

    cfg = get_config(config)
    snap = get_flag_snapshot(
        cfg, refresh=refresh,
        max_age_minutes=float(cfg.get("playbook", {}).get("flag_cache_minutes", 60)),
    )
    brief = build_briefing(cfg, snap)
    md = render_markdown(brief)
    Console().print(Markdown(md))
    if not no_write:
        md_path, json_path = write_briefing(cfg, brief)
        rprint(f"\n[dim]written: {md_path}\n         {json_path}[/dim]")


@app.command("playbook")
def playbook_cmd(
    config: str = "configs/default.yaml",
    cycle: int = typer.Option(-1, "--cycle", help="Show only this cycle (0–3)"),
    all_rules: bool = typer.Option(False, "--all", help="Include INACTIVE rules"),
):
    """Evaluate the §4 cycle rules against live flags, prices, and events."""
    from rich.console import Console
    from rich.table import Table
    from .flags import get_flag_snapshot
    from .playbook import evaluate_cycles, load_playbook, load_portfolio
    from .playbook.briefing import _live_prices

    cfg = get_config(config)
    pb = load_playbook(cfg)
    pf = load_portfolio(cfg)
    snap = get_flag_snapshot(
        cfg, max_age_minutes=float(cfg.get("playbook", {}).get("flag_cache_minutes", 60))
    )
    prices = _live_prices(pb, pf)
    evals = evaluate_cycles(pb, pf, snap, prices, include_inactive=all_rules)
    if cycle >= 0:
        evals = [e for e in evals if e.cycle == cycle]

    rprint(f"[bold]{snap.summary_line()}[/bold]\n")
    table = Table(title="Cycle Rules", show_lines=True)
    table.add_column("Rule", style="bold", width=6)
    table.add_column("Status", width=14)
    table.add_column("Label")
    table.add_column("Why / Orders")
    style = {
        "FIRES": "green", "PRICE_GUARD": "yellow",
        "AWAITING_EVENT": "cyan", "BLOCKED_FLAGS": "red", "INACTIVE": "dim",
    }
    for e in evals:
        s = style.get(e.status, "white")
        bits = list(e.reasons)
        for o in e.orders:
            px = f"{o.price:.2f}" if o.price else "?"
            mark = "✓" if o.price_ok else "✗"
            comp = (o.compliance or {}).get("verdict", "")
            bits.append(f"{mark} {o.ticker} ${o.dollars:,.0f} (guard {o.guard}, last {px}) {comp}")
        table.add_row(e.rule_id, f"[{s}]{e.status}[/{s}]", e.label, "\n".join(bits) or "—")
    Console().print(table)


@app.command("check")
def check_cmd(
    ticker: str = typer.Argument(..., help="Symbol to screen"),
    side: str = typer.Argument("BUY", help="BUY or SELL"),
    dollars: float = typer.Argument(0.0, help="Order size in USD (SELL: 0 = full position)"),
    config: str = "configs/default.yaml",
):
    """Pre-trade compliance check: never-buy, lockouts, caps, semi freeze."""
    from .flags import get_flag_snapshot
    from .playbook import check_trade, load_playbook, load_portfolio
    from .playbook.briefing import _live_prices

    cfg = get_config(config)
    pb = load_playbook(cfg)
    pf = load_portfolio(cfg)
    snap = get_flag_snapshot(
        cfg, max_age_minutes=float(cfg.get("playbook", {}).get("flag_cache_minutes", 60))
    )
    prices = _live_prices(pb, pf)

    # 50d MA for lockout names (needed for the re-entry condition)
    sma50: dict[str, float] = {}
    if ticker.upper() in pb.lockout_tickers:
        try:
            from .flags.lookups import _yf_history
            px = _yf_history(ticker.upper(), period="6mo").sort("date")
            sma50[ticker.upper()] = float(px["close"].tail(50).mean())
        except Exception:
            pass

    res = check_trade(
        ticker, side, dollars, pb, pf,
        snapshot=snap, prices=prices, sma50=sma50,
    )
    color = "green" if res.allowed else "red"
    rprint(f"[bold {color}]{res.verdict}[/bold {color}] — {res.side} {res.ticker} "
           + (f"${dollars:,.0f}" if dollars else "(full position)" if res.side == 'SELL' else ""))
    for v in res.violations:
        rprint(f"  [red]✗ {v}[/red]")
    for w in res.warnings:
        rprint(f"  [yellow]⚠ {w}[/yellow]")
    rprint(f"\n[dim]{snap.summary_line()}[/dim]")


@app.command("log-trade")
def log_trade_cmd(
    ticker: str = typer.Argument(...),
    side: str = typer.Argument(..., help="BUY or SELL"),
    qty: float = typer.Argument(...),
    price: float = typer.Argument(...),
    account: str = typer.Option("Z32148892", "--account"),
    basis: float = typer.Option(-1.0, "--basis", help="Average cost basis (SELL: enables realized P&L)"),
    date_str: str = typer.Option("", "--date", help="Trade date YYYY-MM-DD (default today)"),
    note: str = typer.Option("", "--note"),
    config: str = "configs/default.yaml",
):
    """Log an executed fill to the blotter (reports/blotter.csv).

    Runs the compliance check first and stores its verdict alongside the fill;
    SELLs with a known basis record realized P&L.
    """
    from .flags import get_flag_snapshot
    from .playbook import check_trade, load_playbook, load_portfolio, log_trade

    cfg = get_config(config)
    verdict = ""
    try:
        pb = load_playbook(cfg)
        pf = load_portfolio(cfg)
        snap = get_flag_snapshot(
            cfg, max_age_minutes=float(cfg.get("playbook", {}).get("flag_cache_minutes", 60))
        )
        res = check_trade(ticker, side, qty * price, pb, pf, snapshot=snap,
                          prices={ticker.upper(): price})
        verdict = res.verdict
        if not res.allowed:
            rprint(f"[red bold]NOTE: this fill violates the playbook:[/red bold]")
            for v in res.violations:
                rprint(f"  [red]✗ {v}[/red]")
        # auto-fill basis from the portfolio snapshot if not provided
        if basis < 0 and side.upper() == "SELL":
            pos = pf.position(ticker)
            if pos:
                basis = pos.average_cost
                rprint(f"[dim]basis from portfolio snapshot: {basis:.2f}[/dim]")
    except Exception as e:
        rprint(f"[yellow]compliance check skipped: {e}[/yellow]")

    row = log_trade(
        cfg.path("reports"), ticker, side, qty, price,
        trade_date=date_str or None, account=account,
        avg_cost_basis=basis if basis >= 0 else None,
        compliance_verdict=verdict, note=note,
    )
    rprint(f"[green]logged:[/green] {row['trade_date']} {row['side']} {row['qty']} "
           f"{row['ticker']} @ {row['price']} (${row['dollars']:,.2f})"
           + (f"  realized P&L: {row['realized_pnl']}" if row['realized_pnl'] != "" else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Command directory — single source of truth for grouping + docs
# ─────────────────────────────────────────────────────────────────────────────
COMMAND_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Data & features": [
        ("ingest", "Pull daily OHLCV for the universe (yfinance)"),
        ("quality", "OHLCV data-quality checks"),
        ("features", "Build the gold feature matrix (technical+macro+events+nonlinear; --deep)"),
        ("universe", "Show the active universe"),
    ],
    "Forecasting & models": [
        ("train", "Walk-forward 14-model ensemble (5d target)"),
        ("train-forecast", "★ Long-horizon best models (trees+RNN/LSTM/GRU) via purged CV"),
        ("train-intervals", "Conformalized quantile price-bound models (90% coverage)"),
        ("bounds", "Show lower/median/upper price bounds for a ticker"),
        ("backtest", "Vectorized backtest of a strategy with metrics"),
    ],
    "Decisions": [
        ("picks", "★ Long-term buy plan: what/when/entry/target/stop, ranked"),
        ("analyze", "Single-symbol decision + report (bounds, SHAP, narration)"),
        ("analyze-all", "Run analyze across the whole universe"),
        ("signals", "Cross-sectional signal table"),
        ("complexity", "Nonlinear fingerprint: chaos/fractal/entropy/bubble signatures"),
        ("explain", "DeepSeek plain-English narration of a report"),
    ],
    "Future-predict sessions": [
        ("future-predict", "Open a dated forecast session with allocation"),
        ("future-status", "Show a session's equity + hit-rate"),
        ("future-update", "MTM + redeploy dry powder for live sessions"),
    ],
    "Playbook": [
        ("flags", "Live O/F/I/S/C flag board + composite"),
        ("brief", "One-page morning briefing"),
        ("playbook", "Which §4 cycle rules fire today"),
        ("check", "Pre-trade compliance (never-buy, caps, freeze)"),
        ("log-trade", "Record a fill in the blotter"),
    ],
    "Paper trading": [
        ("paper-trade", "Run the ML/ensemble paper portfolio"),
        ("paper-status", "Show paper-portfolio holdings + equity"),
        ("daily", "Full daily pipeline (ingest→features→signals→rebalance)"),
    ],
    "Agent (LLM)": [
        ("agent-analyze", "Multi-step ReAct analysis of a ticker"),
        ("agent-portfolio", "Agent review of the whole portfolio"),
        ("agent-briefing", "Agent-written morning brief"),
    ],
    "App": [
        ("dashboard", "Launch the Streamlit desk"),
        ("commands", "This command directory"),
    ],
}


def _render_command_directory() -> str:
    """Markdown for docs/COMMANDS.md (also used by `ts commands --md`)."""
    lines = ["# Command directory", "", "All commands are invoked as `ts <command>`. "
             "Run `ts <command> --help` for options.", ""]
    for group, cmds in COMMAND_GROUPS.items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| Command | What it does |")
        lines.append("| --- | --- |")
        for name, desc in cmds:
            lines.append(f"| `ts {name}` | {desc} |")
        lines.append("")
    return "\n".join(lines)


@app.command()
def commands(
    md: bool = typer.Option(False, "--md", help="print raw markdown instead of a table"),
    write: bool = typer.Option(False, "--write", help="(re)write docs/COMMANDS.md"),
):
    """Show the grouped command directory (and optionally regenerate the docs)."""
    if write:
        out = get_config().project_root / "docs" / "COMMANDS.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_command_directory())
        rprint(f"[green]Wrote {out}[/green]")
        return
    if md:
        print(_render_command_directory())
        return
    from rich.table import Table
    from rich.console import Console
    console = Console()
    for group, cmds in COMMAND_GROUPS.items():
        t = Table(title=group, show_header=True, title_justify="left", expand=True)
        t.add_column("Command", style="cyan", no_wrap=True)
        t.add_column("What it does")
        for name, desc in cmds:
            t.add_row(f"ts {name}", desc)
        console.print(t)


if __name__ == "__main__":
    app()

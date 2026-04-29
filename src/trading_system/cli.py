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
from .ingestion import ingest_universe
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


@app.command()
def ingest(config: str = "configs/default.yaml"):
    """Ingest the configured universe to bronze parquet."""
    cfg = get_config(config)
    out = ingest_universe(cfg)
    rprint(f"[green]Wrote {out}[/green]")


@app.command()
def quality(config: str = "configs/default.yaml"):
    """Run data-quality checks on bronze OHLCV."""
    cfg = get_config(config)
    df = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")
    res = run_ohlcv_checks(df)
    rprint(res)


@app.command()
def features(config: str = "configs/default.yaml"):
    """Build the feature matrix and write to gold."""
    cfg = get_config(config)
    df = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")
    feat = build_feature_matrix(df, benchmark=cfg["universe"]["benchmark"])
    out = cfg.path("data_gold") / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(out, compression="zstd")
    rprint(f"[green]Wrote {len(feat)} rows to {out}[/green]")


@app.command()
def backtest(
    strategy: str = typer.Argument("momentum_rotation"),
    config: str = "configs/default.yaml",
):
    """Backtest a named strategy on the gold features."""
    if strategy not in STRATS:
        raise typer.BadParameter(f"Unknown strategy: {strategy}. Options: {list(STRATS)}")
    cfg = get_config(config)
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


@app.command()
def daily(config: str = "configs/default.yaml"):
    """Run the full daily pipeline."""
    cfg = get_config(config)
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
def universe(config: str = "configs/default.yaml"):
    """Print the configured universe."""
    cfg = get_config(config)
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
):
    """Run the full decision pipeline on a single symbol and write a report."""
    cfg = get_config(config)
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
def analyze_all_cmd(config: str = "configs/default.yaml"):
    """Run the decision pipeline across the entire configured universe."""
    cfg = get_config(config)
    results = analyze_all(cfg)
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


if __name__ == "__main__":
    app()

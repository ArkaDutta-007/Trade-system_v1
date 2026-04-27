"""Top-level CLI: `ts <command>`."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from .backtesting import compute_metrics, run_vectorized_backtest, summarize
from .backtesting.slippage import CostModel
from .config import get_config
from .decision import analyze_symbol, analyze_all
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
    """Walk-forward train an ML ranker over the feature matrix."""
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
    models, oos = train_walk_forward(
        feat, spec,
        train_years=wf["train_years"],
        test_years=wf["test_years"],
        step_years=wf["step_years"],
        params=cfg["model"]["params"],
    )
    out = cfg.path("data_gold") / "predictions.parquet"
    oos.write_parquet(out, compression="zstd")
    rprint(f"[green]Wrote {len(oos)} OOS predictions to {out}[/green]")

    if models:
        last = models[-1]["model"]
        sh = compute_shap_summary(last, feat, spec.feature_columns)
        sh_out = cfg.path("reports") / "shap_summary.csv"
        sh_out.parent.mkdir(parents=True, exist_ok=True)
        sh.write_csv(sh_out)
        rprint(f"SHAP summary -> {sh_out}")


@app.command()
def daily(config: str = "configs/default.yaml"):
    """Run the full daily pipeline."""
    cfg = get_config(config)
    path = run_daily_pipeline(cfg)
    rprint(f"[green]Daily report: {path}[/green]")


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


if __name__ == "__main__":
    app()

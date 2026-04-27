"""Render a DecisionResult to a markdown audit doc + JSON sidecar."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _render_technical(t: dict) -> str:
    if not t.get("available"):
        return "_no technical data_"
    rows = [
        ("As of", t.get("date")),
        ("Last close", _fmt_num(t.get("adj_close"))),
        ("1d return", _fmt_pct(t.get("ret_1d"))),
        ("5d momentum", _fmt_pct(t.get("mom_5d"))),
        ("20d momentum", _fmt_pct(t.get("mom_20d"))),
        ("60d momentum", _fmt_pct(t.get("mom_60d"))),
        ("12-1m momentum", _fmt_pct(t.get("mom_12m1m"))),
        ("Realized vol (20d, ann.)", _fmt_pct(t.get("vol_20d"))),
        ("Realized vol (60d, ann.)", _fmt_pct(t.get("vol_60d"))),
        ("RSI(14)", _fmt_num(t.get("rsi_14"), 1)),
        ("ATR(14)", _fmt_num(t.get("atr_14"), 3)),
        ("Gap vs 50d SMA", _fmt_pct(t.get("sma_gap_50"))),
        ("Gap vs 200d SMA", _fmt_pct(t.get("sma_gap_200"))),
        ("Breakout vs 20d high", _fmt_pct(t.get("breakout_20"))),
        ("Drawdown from 60d high", _fmt_pct(t.get("dd_from_high_60"))),
        ("Relative volume (20d)", _fmt_num(t.get("rel_vol_20"), 2)),
        ("Avg $ volume (20d)", f"${(t.get('avg_dollar_volume_20') or 0):,.0f}"),
    ]
    return "\n".join(f"- **{k}:** {v}" for k, v in rows)


def _render_regime(r: dict) -> str:
    if not r.get("available"):
        return "_no regime data_"
    return "\n".join([
        f"- **Bull regime (price > 200d SMA):** {bool(r.get('bull_regime'))}",
        f"- **High vol regime:** {bool(r.get('high_vol_regime'))}",
        f"- **Cross-sectional 20d momentum percentile:** {_fmt_num(r.get('mom_20d_rank'), 3)}",
        f"- **Excess return vs benchmark (1d):** {_fmt_pct(r.get('excess_ret_1d'))}",
    ])


def _render_cross_section(c: dict) -> str:
    if not c.get("available"):
        return "_no cross-sectional data_"
    top = "\n".join(
        f"  - {row['ticker']}: {_fmt_pct(row.get('mom_20d'))}" for row in c.get("top_by_mom_20d", [])
    )
    bot = "\n".join(
        f"  - {row['ticker']}: {_fmt_pct(row.get('mom_20d'))}" for row in c.get("bottom_by_mom_20d", [])
    )
    return (
        f"- **Rank in universe (by 20d momentum):** {c['rank_in_universe']} of {c['universe_size']}\n"
        f"- **As of:** {c['as_of']}\n"
        f"- **Top 10 by 20d momentum:**\n{top}\n"
        f"- **Bottom 10 by 20d momentum:**\n{bot}"
    )


def _render_events(e: dict) -> str:
    if not e.get("available") or not e.get("rows"):
        return "_no recent events recorded (run news/SEC ingestion to populate)_"
    lines = []
    for row in e["rows"]:
        ka = row.get("known_at")
        sent = row.get("sentiment")
        sent_s = f"sentiment={sent:+.2f}" if sent is not None else ""
        url = row.get("source_url") or ""
        link = f" [link]({url})" if url else ""
        lines.append(
            f"- {ka} ({row.get('source')}, {row.get('event_type')}, {sent_s}): "
            f"{row.get('summary')}{link}"
        )
    return "\n".join(lines)


def _render_model(m: dict) -> str:
    if not m.get("available"):
        return f"_{m.get('reason', 'no model artifact found')}_"
    out = [
        f"- **Model score:** {_fmt_num(m['score'], 4)} ({m['score_meaning']})",
        f"- **Feature columns used:** {len(m.get('feature_columns', []))}",
    ]
    feats = m.get("top_features_by_mean_abs_shap")
    if feats:
        out.append("- **Top features by |SHAP| (most influential first):**")
        for r in feats:
            out.append(f"  - `{r['feature']}` — mean|SHAP|={_fmt_num(r['mean_abs_shap'], 5)}")
    return "\n".join(out)


def render_markdown(result) -> str:
    g = result.groundings
    return f"""# Decision: {result.ticker} — {result.stance}

**As of:** {result.as_of}
**Stance:** **{result.stance}** (confidence {result.confidence:.2f})
**Score source:** {result.score_source}
**5-day forecast:** {_fmt_pct(result.forecast_5d)}
**20-day forecast:** {_fmt_pct(result.forecast_20d)}

## Rationale

{chr(10).join(f"- {r}" for r in result.rationale) or "- _no rationale recorded_"}

## Technical state

{_render_technical(g.get('technical', {}))}

## Regime context

{_render_regime(g.get('regime', {}))}

## Cross-sectional position

{_render_cross_section(g.get('cross_section', {}))}

## Recent events (≤14d window)

{_render_events(g.get('events', {}))}

## Model groundings

{_render_model(g.get('model', {}))}

---
_Generated {datetime.utcnow().isoformat()}Z. Not financial advice; research only._
"""


def write_decision_report(result, root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    md_path = root / f"{result.ticker}_{stamp}.md"
    json_path = root / f"{result.ticker}_{stamp}.json"
    md_path.write_text(render_markdown(result))
    json_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return md_path, json_path

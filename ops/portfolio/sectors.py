#!/usr/bin/env python3
"""Sector/industry map for the trading universe — fetched once, cached on disk.

The gold feature matrix has no sector column, so diversification constraints
had nothing to work with. This fetches sector + industry + market cap from
Yahoo once per ticker and caches to ~/trade-ops/portfolio/sectors.json.

Usage:
    python3 sectors.py            # refresh missing tickers only
    python3 sectors.py --all      # re-fetch everything
    python3 sectors.py --show     # print the sector histogram
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()
CACHE = ops_root() / "portfolio" / "sectors.json"   # state, not code

# Coarse "theme" grouping — what actually drives correlated drawdowns in this
# book. Yahoo's sector labels are too coarse in one direction (everything AI is
# "Technology") and too fine in another, so we overlay explicit theme buckets.
THEME_OVERRIDES = {
    # AI infrastructure / semis — the dominant correlated cluster
    "NVDA": "ai_semi", "AMD": "ai_semi", "AVGO": "ai_semi", "TSM": "ai_semi",
    "ASML": "ai_semi", "LRCX": "ai_semi", "AMAT": "ai_semi", "KLAC": "ai_semi",
    "MU": "ai_semi", "ARM": "ai_semi", "MRVL": "ai_semi", "SMCI": "ai_semi",
    "GFS": "ai_semi", "TER": "ai_semi", "SNDK": "ai_semi", "INTC": "ai_semi",
    "TOELY": "ai_semi", "SUMCF": "ai_semi", "SHECY": "ai_semi", "MKKGY": "ai_semi",
    "CDNS": "ai_semi", "SNPS": "ai_semi", "QCOM": "ai_semi", "TXN": "ai_semi",
    # AI datacenter / neocloud / power
    "CRWV": "ai_cloud", "IREN": "ai_cloud", "NBIS": "ai_cloud", "APLD": "ai_cloud",
    "GEV": "ai_cloud", "VRT": "ai_cloud", "BE": "ai_cloud", "STRL": "ai_cloud",
    # crypto-linked
    "COIN": "crypto", "MSTR": "crypto", "SBET": "crypto", "MARA": "crypto",
    "RIOT": "crypto", "HOOD": "crypto",
    # mega-cap platforms
    "AAPL": "megacap", "MSFT": "megacap", "GOOGL": "megacap", "GOOG": "megacap",
    "AMZN": "megacap", "META": "megacap", "NFLX": "megacap",
    # speculative EV / story stocks
    "TSLA": "ev_story", "LCID": "ev_story", "RIVN": "ev_story", "QS": "ev_story",
}


def load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def universe_tickers() -> list[str]:
    import polars as pl
    f = pl.read_parquet(REPO / "data/gold/features.parquet", columns=["ticker"])
    return sorted(f["ticker"].unique().to_list())


def fetch(tickers: list[str], existing: dict) -> dict:
    import yfinance as yf
    out = dict(existing)
    todo = [t for t in tickers if t not in out]
    print(f"fetching {len(todo)} tickers ({len(tickers)-len(todo)} cached)")
    for i, t in enumerate(todo, 1):
        try:
            info = yf.Ticker(t).info
            out[t] = {
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
                "market_cap": info.get("marketCap"),
            }
        except Exception as e:  # noqa: BLE001
            out[t] = {"sector": "Unknown", "industry": "Unknown",
                      "market_cap": None, "error": str(e)[:80]}
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}…", flush=True)
            CACHE.write_text(json.dumps(out, indent=1))
        time.sleep(0.15)
    CACHE.write_text(json.dumps(out, indent=1))
    return out


def theme_of(ticker: str, meta: dict) -> str:
    """Correlated-risk bucket: explicit theme override, else Yahoo sector."""
    if ticker in THEME_OVERRIDES:
        return THEME_OVERRIDES[ticker]
    sec = (meta.get(ticker) or {}).get("sector") or "Unknown"
    return sec.lower().replace(" ", "_")


def main() -> int:
    args = sys.argv[1:]
    existing = {} if "--all" in args else load()
    if "--show" in args:
        m = load()
        from collections import Counter
        c = Counter(theme_of(t, m) for t in m)
        for k, v in c.most_common():
            print(f"{v:4d}  {k}")
        print(f"total {len(m)} tickers")
        return 0
    m = fetch(universe_tickers(), existing)
    print(f"cached {len(m)} tickers → {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

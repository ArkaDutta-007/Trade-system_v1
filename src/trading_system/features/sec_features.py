"""SEC filing features — point-in-time disclosure/insider intensity + recency.

From the ``sec_history`` silver table (one row per filing: ticker, date, form)
this builds trailing counts as of each panel date D, using only filings with
``filing_date <= D`` (public that day → leakage-free vs the D→D+h target):

  sec_filings_30d      all filings in the trailing 30 calendar days (disclosure intensity)
  sec_8k_30d           8-K filings in 30d (material corporate events)
  sec_form4_90d        Form-4 filings in 90d (insider transaction activity)
  sec_days_since_filing calendar days since the most recent filing (recency)

Tickers with no SEC filings (e.g. ETFs) stay null and are handled downstream by
:mod:`features.sparse_signals` (neutral-fill + a ``sec_present`` flag).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

SEC_COLUMNS = ["sec_filings_30d", "sec_8k_30d", "sec_form4_90d", "sec_days_since_filing"]


def _is_prefix(forms: np.ndarray, prefix: str) -> np.ndarray:
    return np.array([f.startswith(prefix) for f in forms], dtype=np.int64)


# Insider transaction reports are exactly Form 4 (and its amendment) — not the
# unrelated 40-F / 424B forms that merely start with "4".
_FORM4 = {"4", "4/A"}


def _is_form4(forms: np.ndarray) -> np.ndarray:
    return np.array([f in _FORM4 for f in forms], dtype=np.int64)


def compute_sec_features(features: pl.DataFrame, sec: pl.DataFrame | None) -> pl.DataFrame:
    """Join point-in-time SEC filing-intensity features onto the (ticker, date) panel."""
    if sec is None or sec.is_empty() or "form" not in sec.columns:
        return features

    panel = features.select(["ticker", "date"]).with_row_index("_row").sort(["ticker", "date"])
    # ordinal (calendar) day for O(log n) window counts via searchsorted
    epoch = np.datetime64("1970-01-01")
    panel_days = (panel["date"].to_numpy().astype("datetime64[D]") - epoch).astype(np.int64)
    panel_tk = panel["ticker"].to_numpy()
    rows_idx = panel["_row"].to_numpy()

    filings_by_ticker: dict[str, dict] = {}
    for tk, grp in sec.sort(["ticker", "date"]).group_by("ticker", maintain_order=True):
        tk = tk[0] if isinstance(tk, tuple) else tk
        fd = (grp["date"].to_numpy().astype("datetime64[D]") - epoch).astype(np.int64)
        forms = grp["form"].to_numpy().astype(str)
        # prefix sums so a window count is one subtraction
        c_all = np.concatenate([[0], np.cumsum(np.ones_like(fd))])
        c_8k = np.concatenate([[0], np.cumsum(_is_prefix(forms, "8-K"))])
        c_f4 = np.concatenate([[0], np.cumsum(_is_form4(forms))])
        filings_by_ticker[tk] = {"fd": fd, "c_all": c_all, "c_8k": c_8k, "c_f4": c_f4}

    n = panel.height
    f30 = np.full(n, np.nan); e8 = np.full(n, np.nan)
    f4 = np.full(n, np.nan); since = np.full(n, np.nan)

    # contiguous per-ticker slices (panel is sorted by ticker,date)
    start = 0
    for i in range(1, n + 1):
        if i == n or panel_tk[i] != panel_tk[start]:
            tk = panel_tk[start]
            info = filings_by_ticker.get(tk)
            if info is not None:
                d = panel_days[start:i]
                fd = info["fd"]
                hi = np.searchsorted(fd, d, side="right")           # filings on/before D
                lo30 = np.searchsorted(fd, d - 30, side="right")
                lo90 = np.searchsorted(fd, d - 90, side="right")
                sl = slice(start, i)
                f30[sl] = info["c_all"][hi] - info["c_all"][lo30]
                e8[sl] = info["c_8k"][hi] - info["c_8k"][lo30]
                f4[sl] = info["c_f4"][hi] - info["c_f4"][lo90]
                # days since last filing (nan until the first filing exists)
                last = np.where(hi > 0, fd[np.clip(hi - 1, 0, len(fd) - 1)], np.nan)
                since[sl] = np.where(hi > 0, d - last, np.nan)
            start = i

    feat = pl.DataFrame({
        "_row": rows_idx,
        "sec_filings_30d": f30, "sec_8k_30d": e8,
        "sec_form4_90d": f4, "sec_days_since_filing": since,
    })
    out = (
        features.with_row_index("_row")
        .join(feat, on="_row", how="left")
        .drop("_row")
        # NaN placeholders (uncovered ticker / pre-first-filing) → proper nulls, so
        # the coverage gate and densify's fill_null treat them as "absent".
        .with_columns([pl.col(c).fill_nan(None) for c in SEC_COLUMNS])
    )
    n_cov = out["sec_filings_30d"].is_not_null().sum()
    logger.info(f"SEC features: filing counts populated on {n_cov:,}/{out.height:,} rows "
                f"({100*n_cov/max(out.height,1):.0f}% of panel)")
    return out

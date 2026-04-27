"""Performance metrics. All assume daily returns unless stated."""
from __future__ import annotations

import numpy as np
import polars as pl

TRADING_DAYS = 252


def _to_np(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def cagr(returns) -> float:
    r = _to_np(returns)
    if len(r) == 0:
        return 0.0
    eq = np.prod(1 + r)
    years = len(r) / TRADING_DAYS
    return eq ** (1 / years) - 1 if years > 0 else 0.0


def annual_vol(returns) -> float:
    r = _to_np(returns)
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe(returns, rf: float = 0.0) -> float:
    r = _to_np(returns) - rf / TRADING_DAYS
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(returns, rf: float = 0.0) -> float:
    r = _to_np(returns) - rf / TRADING_DAYS
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(r) > 0 else 0.0
    dd = np.std(downside, ddof=1)
    return float(np.mean(r) / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def max_drawdown(returns) -> float:
    r = _to_np(returns)
    eq = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return float(dd.min()) if len(dd) else 0.0


def calmar(returns) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else 0.0


def hit_rate(returns) -> float:
    r = _to_np(returns)
    return float((r > 0).mean()) if len(r) else 0.0


def turnover_total(turnover) -> float:
    return float(np.sum(_to_np(turnover)))


def var_cvar(returns, alpha: float = 0.05) -> tuple[float, float]:
    r = _to_np(returns)
    if len(r) == 0:
        return 0.0, 0.0
    var = float(np.quantile(r, alpha))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    return var, cvar


def compute_metrics(returns, turnover=None, benchmark=None) -> dict:
    out = {
        "CAGR": cagr(returns),
        "AnnualVol": annual_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDrawdown": max_drawdown(returns),
        "Calmar": calmar(returns),
        "HitRate": hit_rate(returns),
    }
    var, cvar = var_cvar(returns)
    out["VaR_5pct"] = var
    out["CVaR_5pct"] = cvar

    if turnover is not None:
        out["TotalTurnover"] = turnover_total(turnover)
    if benchmark is not None:
        b = _to_np(benchmark)
        r = _to_np(returns)
        n = min(len(r), len(b))
        if n > 0:
            excess = r[-n:] - b[-n:]
            out["ExcessReturn_Annual"] = float(np.mean(excess) * TRADING_DAYS)
            if np.std(excess) > 0:
                out["InformationRatio"] = float(
                    np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS)
                )
    return out


def summarize(metrics: dict) -> str:
    lines = ["Performance Summary", "-" * 30]
    for k, v in metrics.items():
        if isinstance(v, float):
            if "Rate" in k or "Drawdown" in k or "VaR" in k or "CVaR" in k or k in ("CAGR", "AnnualVol", "ExcessReturn_Annual"):
                lines.append(f"{k:>22}: {v:>8.2%}")
            else:
                lines.append(f"{k:>22}: {v:>8.3f}")
        else:
            lines.append(f"{k:>22}: {v}")
    return "\n".join(lines)

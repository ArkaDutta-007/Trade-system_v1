"""Feature reserve resolution + compute-profile detection."""
import polars as pl

from trading_system.features.reserve import (
    FEATURE_RESERVE, GROUPS, resolve_reserve, reserve_report,
)
from trading_system.utils.compute import get_compute_profile, ComputeProfile


def test_reserve_has_no_duplicates():
    assert len(FEATURE_RESERVE) == len(set(FEATURE_RESERVE))
    # every grouped column is in the reserve
    for cols in GROUPS.values():
        for c in cols:
            assert c in FEATURE_RESERVE


def test_resolve_keeps_present_drops_absent_and_sparse():
    df = pl.DataFrame({
        "mom_20d": [0.1, 0.2, 0.3, 0.4, 0.5],          # present, full coverage
        "vol_20d": [0.2, 0.2, 0.2, 0.2, 0.2],          # present, full coverage
        "rsi_14": [None, None, None, None, 55.0],      # mostly null -> dropped
        "unrelated_col": [1, 2, 3, 4, 5],              # not in reserve -> ignored
    })
    resolved = resolve_reserve(df, min_non_null_frac=0.6)
    assert "mom_20d" in resolved and "vol_20d" in resolved
    assert "rsi_14" not in resolved          # 20% coverage < 60%
    assert "unrelated_col" not in resolved   # not a reserve feature


def test_resolve_group_filter():
    df = pl.DataFrame({"mom_20d": [0.1] * 5, "macro_vix": [13.0] * 5})
    only_trend = resolve_reserve(df, groups=["trend"])
    assert "mom_20d" in only_trend
    assert "macro_vix" not in only_trend  # macro group excluded


def test_reserve_report_shape():
    df = pl.DataFrame({"mom_20d": [0.1] * 5, "macro_vix": [13.0] * 5})
    rep = reserve_report(df)
    assert rep["trend"]["present"] >= 1
    assert "_total" in rep and rep["_total"]["reserve_size"] == len(FEATURE_RESERVE)


def test_compute_profile_sane():
    p = get_compute_profile()
    assert isinstance(p, ComputeProfile)
    assert p.n_jobs >= 1
    assert p.lgbm_device in ("cpu", "gpu")
    assert p.xgb_device in ("cpu", "cuda")
    assert "n_jobs" in p.lgbm_params()
    assert p.xgb_params()["tree_method"] == "hist"
    assert p.platform in ("apple_silicon", "linux_gpu", "cpu")

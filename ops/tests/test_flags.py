"""Unit tests for the automated flag engine's statistics and state machine."""
import json
import numpy as np
import pytest

import auto_flags as af


# ── robust statistics ───────────────────────────────────────────────────────
def test_robust_z_is_zero_at_the_median():
    x = np.arange(101, dtype=float)
    assert af.robust_z(x, float(np.median(x))) == pytest.approx(0.0)


def test_robust_z_resists_outliers_unlike_mean_sd():
    """One absurd spike must not rescale the whole series.

    Uses a genuinely dispersed base (a constant series has MAD 0, which the
    guard maps to 0.0 by design and would not exercise the comparison).
    """
    rng = np.random.default_rng(11)
    base = np.concatenate([rng.normal(100.0, 5.0, 200), np.array([1e6])])
    z_robust = af.robust_z(base, 130.0)
    mean_sd_z = (130.0 - base.mean()) / base.std(ddof=1)
    assert z_robust > 3.0            # robust scale still sees 130 as far out
    assert abs(mean_sd_z) < 0.1      # mean/sd is destroyed by the single spike


def test_robust_z_degenerate_inputs():
    assert af.robust_z(np.array([1.0, 2.0]), 1.5) == 0.0     # too few points
    assert af.robust_z(np.full(50, 5.0), 9.0) == 0.0         # zero MAD


def test_theil_sen_recovers_a_known_slope():
    y = 3.0 + 2.0 * np.arange(50)
    assert af.theil_sen(y) == pytest.approx(2.0, rel=1e-9)


def test_theil_sen_ignores_a_corrupted_point():
    y = (1.0 * np.arange(60)).astype(float)
    y[30] = 5000.0                                    # single corrupted reading
    assert af.theil_sen(y) == pytest.approx(1.0, rel=0.05)


def test_theil_sen_flat_and_short():
    assert af.theil_sen(np.full(30, 7.0)) == pytest.approx(0.0)
    assert af.theil_sen(np.array([1.0, 2.0])) == 0.0


def test_norm_slope_is_scale_invariant():
    """Doubling the units must not change a unit-free trend measure."""
    y = np.cumsum(np.random.default_rng(1).normal(0.05, 1.0, 120))
    assert af.norm_slope(y) == pytest.approx(af.norm_slope(y * 1000.0), rel=1e-6)


# ── hysteresis state machine ────────────────────────────────────────────────
def test_cut_basic_mapping_without_history():
    assert af.cut(2.0, None) == "GREEN"
    assert af.cut(0.0, None) == "YELLOW"
    assert af.cut(-2.0, None) == "RED"


def test_cut_holds_state_inside_the_hysteresis_band():
    """A score barely over the GREEN line must not flip a YELLOW flag."""
    just_over = 0.75 + af.HYST / 2
    assert af.cut(just_over, "YELLOW") == "YELLOW"
    assert af.cut(0.75 + af.HYST + 0.01, "YELLOW") == "GREEN"   # clears margin


def test_cut_is_sticky_downward_too():
    just_under = -0.75 - af.HYST / 2
    assert af.cut(just_under, "YELLOW") == "YELLOW"
    assert af.cut(-0.75 - af.HYST - 0.01, "YELLOW") == "RED"


def test_cut_green_persists_until_clearly_broken():
    assert af.cut(0.70, "GREEN") == "GREEN"       # small dip: hold
    assert af.cut(0.10, "GREEN") == "YELLOW"      # decisive: flip


def test_cut_no_chatter_on_a_noisy_walk_around_the_boundary():
    """The point of hysteresis: noise at the threshold must not thrash."""
    rng = np.random.default_rng(3)
    prev, flips = "YELLOW", 0
    for _ in range(400):
        score = 0.75 + rng.normal(0, 0.05)        # hovering exactly on the line
        cur = af.cut(score, prev)
        flips += cur != prev
        prev = cur
    assert flips <= 3, f"flag chattered {flips} times"


# ── basket / series helpers ─────────────────────────────────────────────────
def test_basket_curve_is_normalised_and_equal_weighted():
    import polars as pl
    dates = list(range(300))
    rows = []
    for t, mult in (("AAA", 1.0), ("BBB", 10.0)):     # very different price scales
        for i in dates:
            rows.append({"date": i, "ticker": t, "adj_close": mult * (100 + i)})
    px = pl.DataFrame(rows)
    curve = af.basket_curve(px, ["AAA", "BBB"], 300)
    assert curve[0] == pytest.approx(1.0)             # normalised to start
    # equal weight => a 10x price level must not dominate
    assert curve[-1] == pytest.approx((100 + 299) / 100.0, rel=1e-6)


def test_basket_curve_skips_missing_tickers():
    import polars as pl
    px = pl.DataFrame([{"date": i, "ticker": "AAA", "adj_close": 100.0 + i}
                       for i in range(300)])
    assert af.basket_curve(px, ["AAA", "NOPE"], 300).size > 0
    assert af.basket_curve(px, ["NOPE"], 300).size == 0


# ── output contract ─────────────────────────────────────────────────────────
def test_written_yaml_is_parseable_and_complete():
    """The generated overrides file must satisfy the repo's schema."""
    import yaml
    p = af.OUT_YAML
    if not p.exists():
        pytest.skip("auto_overrides.yaml not generated yet")
    d = yaml.safe_load(p.read_text())
    assert set(d["flags"]) == {"O", "F", "I", "S", "C"}
    for k, v in d["flags"].items():
        assert v["color"] in ("GREEN", "YELLOW", "RED", None), k
        assert "note" in v and "as_of" in v
    assert "events" in d
    assert isinstance(d.get("max_age_days"), int)


def test_state_file_roundtrips():
    if not af.STATE_JSON.exists():
        pytest.skip("no state yet")
    d = json.loads(af.STATE_JSON.read_text())
    assert "flags" in d and "as_of" in d
    for k, v in d["flags"].items():
        assert v["color"] in ("GREEN", "YELLOW", "RED")
        assert isinstance(v["score"], (int, float))


# ── regression tests for the 2026-09-08 F recalibration ─────────────────────
def test_formula_version_is_declared():
    """Hysteresis must be invalidated when a scoring formula changes."""
    assert isinstance(af.FORMULA_VERSION, str) and af.FORMULA_VERSION


def test_state_file_records_the_formula_version():
    if not af.STATE_JSON.exists():
        pytest.skip("no state yet")
    d = json.loads(af.STATE_JSON.read_text())
    assert d.get("formula_version") == af.FORMULA_VERSION, (
        "state was written by a different formula version — hysteresis would "
        "carry a colour the current formula never produced")


def test_f_flag_is_regime_relative_not_absolute():
    """Regression: an absolute spread cut made a 70th-percentile reading RED.

    A 2y-minus-funds spread of +0.74pp is ordinary across 1976-2026 (median
    +0.39). Scoring must be relative to the trailing distribution, so an
    unremarkable level cannot force the playbook into 'defensives only'.
    """
    rng = np.random.default_rng(5)
    # a 5y window whose median already sits near the current level
    win = np.concatenate([rng.normal(0.70, 0.25, 1259), [0.74]])
    z = af.robust_z(win, 0.74)
    score = float(np.clip(-z / 3.0 - 0.15 * 0.0, -3, 3))
    assert abs(z) < 1.0, "a level near its own median must not be an outlier"
    assert score > -0.75, f"typical reading must not score RED (got {score})"


def test_f_flag_still_fires_on_a_genuine_outlier():
    """The recalibration must not make the flag inert."""
    rng = np.random.default_rng(6)
    win = np.concatenate([rng.normal(-0.40, 0.20, 1259), [1.60]])
    z = af.robust_z(win, 1.60)
    score = float(np.clip(-z / 3.0 - 0.15 * 0.5, -3, 3))
    assert z > 3.0
    assert score <= -0.75, f"a true outlier must score RED (got {score})"

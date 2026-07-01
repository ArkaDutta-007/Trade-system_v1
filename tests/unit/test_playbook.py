"""Unit tests for the playbook engine: compliance, standing rules, cycles."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml

from trading_system.config import Config
from trading_system.flags.composite import compute_composite
from trading_system.flags.models import FlagColor, FlagReading, FlagSnapshot
from trading_system.playbook import (
    check_trade,
    evaluate_cycles,
    evaluate_standing_rules,
    load_playbook,
    load_portfolio,
    log_trade,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def playbook(cfg):
    return load_playbook(cfg)


@pytest.fixture(scope="module")
def portfolio(cfg):
    return load_portfolio(cfg)


def _snapshot(o="YELLOW", f="YELLOW", i="GREEN", s="YELLOW", c="GREEN") -> FlagSnapshot:
    colors = dict(zip("OFISC", [o, f, i, s, c]))
    readings = {
        k: FlagReading(
            flag=k, name=k, color=FlagColor(v), value=None, detail="test",
            source="test", as_of=datetime.now(timezone.utc).isoformat(),
        )
        for k, v in colors.items()
    }
    return FlagSnapshot(
        as_of=datetime.now(timezone.utc).isoformat(),
        readings=readings,
        composite=compute_composite(readings),
    )


# ── portfolio loading ────────────────────────────────────────────────────────

class TestPortfolioLoad:
    def test_holdings_loaded(self, portfolio):
        assert "NVDA" in portfolio.held_symbols
        assert "TSM" in portfolio.held_symbols
        # NFLX held in both accounts → aggregated
        pos = portfolio.position("NFLX")
        assert pos.quantity == pytest.approx(5.274 + 2.04, rel=1e-6)

    def test_watchlist_starmine(self, portfolio):
        assert portfolio.starmine("ASML") == pytest.approx(9.9)
        assert portfolio.starmine("TSLA") == pytest.approx(0.5)

    def test_cash_includes_pending(self, portfolio):
        # 46.48 + 3051.58 + 0.65 + 222.45
        assert portfolio.cash == pytest.approx(3321.16, abs=0.01)


# ── blotter (trade log + realized P&L) ───────────────────────────────────────

class TestBlotter:
    def test_sell_records_realized_pnl(self, tmp_path):
        from trading_system.playbook.blotter import blotter_realized
        row = log_trade(tmp_path, "HOOD", "SELL", qty=3.627, price=112.0, avg_cost_basis=75.21)
        assert row["realized_pnl"] == pytest.approx((112.0 - 75.21) * 3.627, abs=0.01)
        realized = blotter_realized(tmp_path, year=date.today().year)
        assert len(realized) == 1 and realized[0] > 0


# ── compliance gates ─────────────────────────────────────────────────────────

class TestCompliance:
    def test_never_buy_blocked(self, playbook, portfolio):
        res = check_trade("TSLA", "BUY", 500, playbook, portfolio, snapshot=_snapshot())
        assert not res.allowed
        assert any("never-buy" in v for v in res.violations)

    def test_hold_not_add_blocked(self, playbook, portfolio):
        res = check_trade("HOOD", "BUY", 500, playbook, portfolio, snapshot=_snapshot())
        assert not res.allowed
        assert any("hold ≠ add" in v for v in res.violations)

    def test_lockout_blocked_when_failing_conditions(self, playbook, portfolio):
        # COIN: StarMine 0.5 → fails the >6 condition regardless of MA
        res = check_trade("COIN", "BUY", 500, playbook, portfolio, snapshot=_snapshot())
        assert not res.allowed
        assert any("lockout" in v for v in res.violations)

    def test_lockout_allows_when_conditions_met(self, playbook, portfolio):
        # Lockout passes only with StarMine > 6 AND price > 50d MA.
        # MNDY has no StarMine in the JSON → still blocked; simulate via DUOL?
        # DUOL StarMine is 3.4 → blocked. So patch watchlist for the test.
        portfolio.watchlist["MNDY"] = {"symbol": "MNDY", "starmine_score": 7.0, "last_price": 82.75}
        res = check_trade(
            "MNDY", "BUY", 500, playbook, portfolio, snapshot=_snapshot(),
            prices={"MNDY": 82.75}, sma50={"MNDY": 75.0},
        )
        # never-buy doesn't include MNDY; lockout conditions now pass
        assert res.allowed, res.violations
        del portfolio.watchlist["MNDY"]

    def test_semi_freeze_blocks_semis(self, playbook, portfolio):
        snap = _snapshot(c="RED")
        res = check_trade("ASML", "BUY", 700, playbook, portfolio, snapshot=snap)
        assert not res.allowed
        assert any("SEMI FREEZE" in v for v in res.violations)

    def test_semi_freeze_does_not_block_nonsemis(self, playbook, portfolio):
        snap = _snapshot(c="RED")  # composite RED too → defensives only
        res = check_trade("NVO", "BUY", 450, playbook, portfolio, snapshot=snap)
        assert res.allowed, res.violations

    def test_composite_red_blocks_non_defensives(self, playbook, portfolio):
        snap = _snapshot(o="RED")
        res = check_trade("GOOGL", "BUY", 700, playbook, portfolio, snapshot=snap)
        assert not res.allowed
        assert any("defensives only" in v for v in res.violations)

    def test_mu_4pct_cap(self, playbook, portfolio):
        assert playbook.cap_for("MU") == 4.0
        assert playbook.cap_for("META") == 13.0
        assert playbook.cap_for("GOOGL") == 13.0  # default

    def test_concentration_cap_blocks_new_cash(self, playbook, portfolio):
        # Inflate META price so it breaches its 13% cap, then try to add
        prices = {"META": 5000.0}
        res = check_trade("META", "BUY", 500, playbook, portfolio,
                          snapshot=_snapshot(), prices=prices)
        assert not res.allowed
        assert any("position cap" in v for v in res.violations)
        assert any("do NOT sell" in v for v in res.violations)

    def test_sell_winner_warns_hold_directive(self, playbook, portfolio):
        res = check_trade("TSM", "SELL", 0, playbook, portfolio, snapshot=_snapshot())
        assert res.allowed  # advisory, not blocking
        assert any("hold-winner" in w for w in res.warnings)
        assert set(res.to_dict()) == {"ticker", "side", "verdict", "violations", "warnings"}


# ── standing rules (§3) ──────────────────────────────────────────────────────

class TestStandingRules:
    def _checks_for(self, playbook, portfolio, ticker, price):
        checks = evaluate_standing_rules(playbook, portfolio, {ticker: price})
        return [c for c in checks if c.ticker == ticker]

    def test_hood_stop_triggers(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "HOOD", 74.0)[0]
        assert c.status == "TRIGGERED"
        assert c.action == "SELL ALL"

    def test_hood_trim_triggers(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "HOOD", 112.0)[0]
        assert c.status == "TRIGGERED"
        assert "TRIM" in c.action

    def test_hood_ok_in_band(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "HOOD", 88.0)[0]
        assert c.status == "OK"

    def test_now_hard_floor(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "NOW", 94.0)[0]
        assert c.status == "TRIGGERED"

    def test_now_awaiting_earnings_event(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "NOW", 110.0)[0]
        assert c.status == "AWAITING"  # now_q2 unresolved in overrides

    def test_uber_strength_exit(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "UBER", 80.0)[0]
        assert c.status == "TRIGGERED"
        assert "strength" in c.action

    def test_dash_capitulation(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "DASH", 139.0)[0]
        assert c.status == "TRIGGERED"
        assert "capitulation" in c.action

    def test_dash_recovery(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "DASH", 175.0)[0]
        assert c.status == "TRIGGERED"
        assert "recovery" in c.action

    def test_crwv_kill_switch_idle_by_default(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "CRWV", 95.0)[0]
        assert c.status == "OK"  # crwv_equity_raise=false in overrides

    def test_tw_floor(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "TW", 89.0)[0]
        assert c.status == "TRIGGERED"

    def test_winners_always_hold(self, playbook, portfolio):
        c = self._checks_for(playbook, portfolio, "TSM", 200.0)[0]
        assert c.status == "OK"
        assert c.action == "HOLD"


# ── cycle rules (§4) ─────────────────────────────────────────────────────────

class TestCycles:
    def test_quiet_window_rule_00_active_jun10_16(self, playbook, portfolio):
        evals = evaluate_cycles(
            playbook, portfolio, _snapshot(), prices={}, today=date(2026, 6, 12),
        )
        r = {e.rule_id: e for e in evals}
        assert r["0.0"].status == "FIRES"          # buy nothing window
        assert r["0.5"].status == "FIRES"          # MU pre-earnings: $0
        assert "0.1" not in r or r["0.1"].status == "INACTIVE"

    def test_rule_01_fires_with_yellow_fed_green_cpi(self, playbook, portfolio):
        snap = _snapshot(f="YELLOW", i="GREEN", s="YELLOW")
        prices = {"NVO": 44.0, "MSFT": 401.0}
        evals = evaluate_cycles(playbook, portfolio, snap, prices, today=date(2026, 6, 22))
        r = {e.rule_id: e for e in evals}
        assert r["0.1"].status == "FIRES"
        assert all(o.price_ok for o in r["0.1"].orders)

    def test_rule_01_price_guard_blocks(self, playbook, portfolio):
        snap = _snapshot(f="YELLOW", i="GREEN")
        prices = {"NVO": 48.0, "MSFT": 401.0}  # NVO above its ≤45 guard
        evals = evaluate_cycles(playbook, portfolio, snap, prices, today=date(2026, 6, 22))
        r = {e.rule_id: e for e in evals}
        assert r["0.1"].status == "PRICE_GUARD"

    def test_rule_02_blocked_unless_fed_green(self, playbook, portfolio):
        snap = _snapshot(f="YELLOW", i="GREEN")
        evals = evaluate_cycles(playbook, portfolio, snap,
                                {"GOOGL": 350.0}, today=date(2026, 6, 22))
        r = {e.rule_id: e for e in evals}
        assert r["0.2"].status == "BLOCKED_FLAGS"

    def test_mu_rule_awaits_event(self, playbook, portfolio):
        evals = evaluate_cycles(playbook, portfolio, _snapshot(),
                                {"MU": 850.0}, today=date(2026, 6, 26))
        r = {e.rule_id: e for e in evals}
        assert r["0.6"].status == "AWAITING_EVENT"

    def test_sizes_scale_with_contribution(self, playbook, portfolio):
        snap = _snapshot(f="YELLOW", i="GREEN")
        prices = {"NVO": 44.0, "MSFT": 401.0}
        old = playbook.raw["monthly_contribution"]
        playbook.raw["monthly_contribution"] = 3000
        try:
            evals = evaluate_cycles(playbook, portfolio, snap, prices, today=date(2026, 6, 22))
            r = {e.rule_id: e for e in evals}
            nvo = [o for o in r["0.1"].orders if o.ticker == "NVO"][0]
            assert nvo.dollars == pytest.approx(450 * 3000 / 2500)
        finally:
            playbook.raw["monthly_contribution"] = old

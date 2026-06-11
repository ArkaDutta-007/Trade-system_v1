"""Unit tests for the five-flag (O/F/I/S/C) regime system — pure logic only."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_system.flags import (
    FlagColor,
    FlagReading,
    classify_core_cpi,
    classify_fed,
    classify_oil,
    classify_semi_tape,
    compute_composite,
)


def _reading(flag: str, color: FlagColor, stale: bool = False) -> FlagReading:
    return FlagReading(
        flag=flag, name=flag, color=color, value=None, detail="test",
        source="test", as_of=datetime.now(timezone.utc).isoformat(), stale=stale,
    )


def _board(o, f, i, s, c) -> dict[str, FlagReading]:
    colors = dict(zip("OFISC", [o, f, i, s, c]))
    return {k: _reading(k, v) for k, v in colors.items()}


# ── classifiers ──────────────────────────────────────────────────────────────

class TestClassifyOil:
    def test_green_below_85_falling(self):
        color, _ = classify_oil(80.0, falling=True, avg_5d=81.0)
        assert color == FlagColor.GREEN

    def test_below_85_not_falling_is_yellow(self):
        color, _ = classify_oil(84.0, falling=False, avg_5d=83.0)
        assert color == FlagColor.YELLOW

    def test_band_85_105_yellow(self):
        color, _ = classify_oil(95.0, falling=True, avg_5d=96.0)
        assert color == FlagColor.YELLOW

    def test_above_105_sustained_red(self):
        color, _ = classify_oil(110.0, falling=False, avg_5d=108.0)
        assert color == FlagColor.RED

    def test_spike_above_105_not_sustained_yellow(self):
        color, detail = classify_oil(107.0, falling=False, avg_5d=101.0)
        assert color == FlagColor.YELLOW
        assert "not sustained" in detail

    def test_hormuz_override_forces_red(self):
        color, _ = classify_oil(70.0, falling=True, hormuz_closed=True)
        assert color == FlagColor.RED


class TestClassifyFed:
    @pytest.mark.parametrize("move,expected", [
        ("hike", FlagColor.RED),
        ("cut", FlagColor.GREEN),
        ("hold", FlagColor.YELLOW),
        (None, FlagColor.UNKNOWN),
    ])
    def test_moves(self, move, expected):
        assert classify_fed(move)[0] == expected


class TestClassifyCoreCpi:
    def test_green_at_or_below_0_2(self):
        assert classify_core_cpi(0.18)[0] == FlagColor.GREEN
        assert classify_core_cpi(0.2)[0] == FlagColor.GREEN

    def test_yellow_at_0_3(self):
        assert classify_core_cpi(0.31)[0] == FlagColor.YELLOW

    def test_red_at_or_above_0_4(self):
        assert classify_core_cpi(0.40)[0] == FlagColor.RED
        assert classify_core_cpi(0.55)[0] == FlagColor.RED

    def test_headline_above_5_forces_red(self):
        assert classify_core_cpi(0.1, headline_yoy_pct=5.3)[0] == FlagColor.RED


class TestClassifySemiTape:
    def test_levels(self):
        assert classify_semi_tape(30500.0)[0] == FlagColor.GREEN
        assert classify_semi_tape(29000.0)[0] == FlagColor.YELLOW
        assert classify_semi_tape(28000.0)[0] == FlagColor.RED


# ── composite ────────────────────────────────────────────────────────────────

class TestComposite:
    def test_green_needs_4_greens_0_red(self):
        comp = compute_composite(_board(*[FlagColor.GREEN] * 4, FlagColor.YELLOW))
        assert comp.color == FlagColor.GREEN
        assert comp.deployment_fraction == 1.0

    def test_any_red_is_red(self):
        comp = compute_composite(_board(FlagColor.RED, *[FlagColor.GREEN] * 4))
        assert comp.color == FlagColor.RED
        assert comp.defensives_only
        assert comp.deployment_fraction == 0.25

    def test_current_pdf_reading_2g3y_is_yellow(self):
        # Jun 10 print: 2 green (I, C) / 3 yellow (O, F, S)
        comp = compute_composite(_board(
            FlagColor.YELLOW, FlagColor.YELLOW, FlagColor.GREEN,
            FlagColor.YELLOW, FlagColor.GREEN,
        ))
        assert comp.color == FlagColor.YELLOW
        assert comp.deployment_fraction == 0.5
        assert not comp.semi_freeze

    def test_c_red_sets_semi_freeze(self):
        comp = compute_composite(_board(
            FlagColor.GREEN, FlagColor.GREEN, FlagColor.GREEN,
            FlagColor.GREEN, FlagColor.RED,
        ))
        assert comp.semi_freeze
        assert comp.color == FlagColor.RED

    def test_s_red_sets_semi_freeze(self):
        comp = compute_composite(_board(
            FlagColor.YELLOW, FlagColor.YELLOW, FlagColor.YELLOW,
            FlagColor.RED, FlagColor.GREEN,
        ))
        assert comp.semi_freeze

    def test_unknown_counts_as_yellow_with_warning(self):
        board = _board(*[FlagColor.GREEN] * 4, FlagColor.UNKNOWN)
        comp = compute_composite(board)
        # 4 greens + 1 unknown→yellow, no reds → still GREEN per ≥4 rule
        assert comp.color == FlagColor.GREEN
        assert any("C" in w for w in comp.data_warnings)

    def test_unknown_can_never_make_red_silently_green(self):
        board = _board(*[FlagColor.UNKNOWN] * 5)
        comp = compute_composite(board)
        assert comp.color == FlagColor.YELLOW
        assert len(comp.data_warnings) == 5

    def test_stale_override_is_flagged(self):
        board = _board(*[FlagColor.GREEN] * 5)
        board["C"].stale = True
        comp = compute_composite(board)
        assert any("stale" in w for w in comp.data_warnings)

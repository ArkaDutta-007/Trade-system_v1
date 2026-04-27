import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from trading_system.backtesting import metrics as M


def test_sharpe_zero_returns():
    assert M.sharpe(np.zeros(252)) == 0.0


def test_max_drawdown_known():
    # +10%, -20%, +5%
    rets = np.array([0.10, -0.20, 0.05])
    eq = np.cumprod(1 + rets)
    expected = (eq.min() - eq.max()) / eq.max() if False else None
    # Compare to manual calc
    peaks = np.maximum.accumulate(eq)
    expected_dd = (eq - peaks) / peaks
    assert M.max_drawdown(rets) == pytest.approx(expected_dd.min())


def test_cagr_positive_for_positive_drift():
    rets = np.full(252, 0.001)
    assert M.cagr(rets) > 0


@settings(max_examples=50)
@given(st.lists(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False), min_size=10, max_size=300))
def test_sharpe_finite(rets):
    out = M.sharpe(np.array(rets))
    assert np.isfinite(out) or out == 0.0

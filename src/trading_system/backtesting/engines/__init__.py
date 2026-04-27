"""Adapters for popular open-source backtest engines.

Each adapter is optional: if the underlying library is not installed, the
factory function raises an informative ImportError. The native engine
(`run_vectorized_backtest`) remains the workhorse; these are for
parameter sweeps (vectorbt), single-asset prototyping (backtesting.py), and
portfolio rotation (bt).
"""
from .vectorbt_engine import vectorbt_backtest
from .backtesting_py_engine import backtesting_py_backtest
from .bt_engine import bt_backtest

__all__ = ["vectorbt_backtest", "backtesting_py_backtest", "bt_backtest"]

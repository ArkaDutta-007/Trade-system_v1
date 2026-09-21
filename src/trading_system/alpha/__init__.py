"""Alpha engine v2 — fast, point-in-time, continuously-learning cross-sectional forecaster.

One panel (``panel.py``) → gradient-boosted rank forecasts per horizon
(``model.py``) → every forecast written to a ledger and tallied against what
actually happened (``ledger.py``) → calibration + horizon weights re-estimated
from the tally (continuous learning) → a constrained long-only book
(``portfolio.py``) → a causal walk-forward backtest that reuses the research
simulator (``backtest.py``).  `ts alpha …` drives it.
"""

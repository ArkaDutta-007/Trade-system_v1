"""Live broker placeholder. Intentionally raises until explicitly wired up."""
from __future__ import annotations


class LiveBroker:
    """Stub live broker. Wire to Alpaca/IBKR/etc. only after extensive paper trading.

    Required before going live:
      * 3-6 months paper trading with realistic costs
      * manual approval mode for orders above a notional threshold
      * kill switch + drawdown halt
      * broker-side and network-side failure simulation
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LiveBroker is intentionally not implemented. Use PaperBroker until "
            "live readiness is reviewed."
        )

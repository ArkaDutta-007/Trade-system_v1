from trading_system.execution.paper_broker import PaperBroker
from trading_system.portfolio.order_policy import Order, weights_to_orders


def test_paper_broker_executes_buy():
    b = PaperBroker(cash=10_000.0, cost_bps=0.0)
    orders = [Order(ticker="SPY", qty=10.0, side="buy", notional=4500)]
    b.submit(orders, prices={"SPY": 450.0})
    assert b.holdings["SPY"] == 10.0
    assert b.cash == 10_000.0 - 10.0 * 450.0


def test_kill_switch_blocks_orders():
    b = PaperBroker(cash=10_000.0)
    b.kill()
    fills = b.submit(
        [Order(ticker="SPY", qty=1.0, side="buy", notional=450)], prices={"SPY": 450.0}
    )
    assert fills == []
    assert b.holdings == {}


def test_weights_to_orders_basic():
    orders = weights_to_orders(
        target_weights={"SPY": 0.5, "QQQ": 0.5},
        holdings={"SPY": 0.0, "QQQ": 0.0},
        prices={"SPY": 100.0, "QQQ": 50.0},
        equity=10_000.0,
    )
    by_t = {o.ticker: o for o in orders}
    assert by_t["SPY"].qty == 50.0
    assert by_t["QQQ"].qty == 100.0
    assert all(o.side == "buy" for o in orders)


def test_weights_to_orders_skips_below_min_notional():
    orders = weights_to_orders(
        target_weights={"SPY": 0.0001},
        holdings={"SPY": 0.0},
        prices={"SPY": 100.0},
        equity=1_000.0,
        min_notional=10.0,
    )
    assert orders == []

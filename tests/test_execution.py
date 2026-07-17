from __future__ import annotations

from types import SimpleNamespace

import MetaTrader5 as mt5

from src.execution import OrderExecutor
from src.strategy import Signal


class Connection:
    def __init__(self, trade_mode=0):
        self.trade_mode = trade_mode

    def ensure(self):
        return True

    def account_info(self):
        return SimpleNamespace(trade_mode=self.trade_mode)

    def symbol_info(self, symbol):
        return SimpleNamespace(digits=1, point=0.1, filling_mode=2)


def _position(side, volume=1.0, ticket=42, magic=260717):
    return SimpleNamespace(
        symbol="#US30",
        type=mt5.POSITION_TYPE_BUY if side == Signal.LONG else mt5.POSITION_TYPE_SELL,
        volume=volume,
        ticket=ticket,
        magic=magic,
        time=1_700_000_000,
    )


def test_cancel_pending_respects_real_account_guard(monkeypatch):
    called = False

    def orders_get(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(mt5, "orders_get", orders_get)
    executor = OrderExecutor(
        Connection(trade_mode=2),
        execute=True,
        account_type="demo",
        allow_live=False,
    )

    assert executor.cancel_pending("#US30") == 0
    assert called is False


def test_matching_target_does_not_stack(monkeypatch):
    sent = []
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [_position(Signal.LONG)])
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)

    assert result.ok is True
    assert result.action == "hold"
    assert sent == []


def test_matching_pending_awaits_fill_without_replacing(monkeypatch):
    sent = []
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)

    assert result.ok is True
    assert result.action == "await_fill"
    assert sent == []


def test_reversal_closes_exact_ticket_before_new_entry(monkeypatch):
    sent = []
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, ticket=77)],
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )

    def order_send(request):
        sent.append(request)
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=1,
            deal=2,
            comment="done",
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.SHORT, volume=1.0)

    assert result.ok is True
    assert result.action == "reverse_or_resize"
    assert len(sent) == 2
    assert sent[0]["action"] == mt5.TRADE_ACTION_DEAL
    assert sent[0]["position"] == 77
    assert sent[1]["action"] == mt5.TRADE_ACTION_PENDING
    assert sent[1]["type"] == mt5.ORDER_TYPE_SELL_LIMIT
    assert "awaiting fill" in result.message


def test_positions_get_none_does_not_open_as_flat(monkeypatch):
    sent = []
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: None)
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "last_error", lambda: (-1, "terminal busy"))
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)

    assert result.ok is False
    assert result.action == "fetch_failed"
    assert "positions" in result.message
    assert sent == []


def test_orders_get_none_blocks_flatten_and_entry(monkeypatch):
    sent = []
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: None)
    monkeypatch.setattr(mt5, "last_error", lambda: (-1, "no connection"))
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")

    flat = executor.reconcile_target(symbol="#US30", side=Signal.FLAT, volume=0.0)
    assert flat.ok is False
    assert flat.action == "fetch_failed"
    assert "orders" in flat.message

    open_ = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)
    assert open_.ok is False
    assert open_.action == "fetch_failed"
    assert sent == []


def test_close_all_aborts_when_positions_get_fails(monkeypatch):
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: None)
    monkeypatch.setattr(mt5, "last_error", lambda: (-1, "rpc failed"))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.close_all("#US30")

    assert result.ok is False
    assert "positions_get failed" in result.message


def test_empty_tuple_is_success_zero_not_error(monkeypatch):
    sent = []
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: ())
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: ())
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )

    def order_send(request):
        sent.append(request)
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=1,
            deal=2,
            comment="done",
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)

    assert result.ok is True
    assert result.action == "open"
    assert len(sent) == 1

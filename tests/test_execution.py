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
        return SimpleNamespace(digits=1, point=0.1, filling_mode=2, trade_stops_level=0)


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
    sent = []
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(
        Connection(trade_mode=2),
        execute=True,
        account_type="demo",
        allow_live=False,
    )

    result = executor.cancel_pending("#US30")
    assert result.ok is False
    assert result.skipped is True
    assert result.remaining == [9]
    assert sent == []


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


def test_partial_fill_plus_remainder_pending_awaits_fill(monkeypatch):
    """0.4 filled + 0.6 same-side pending must not be flattened and re-ordered."""
    sent = []
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=0.6,
    )
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, volume=0.4, ticket=77)],
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0)

    assert result.ok is True
    assert result.action == "await_fill"
    assert sent == []


def test_same_side_within_rebalance_band_holds_without_churn(monkeypatch):
    sent = []
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, volume=1.0)],
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "order_send", lambda request: sent.append(request))

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    # ~10% drift from equity change — inside default 15% band
    result = executor.reconcile_target(
        symbol="#US30", side=Signal.LONG, volume=1.10, rebalance_band=0.15
    )

    assert result.ok is True
    assert result.action == "hold"
    assert "rebalance band" in result.message
    assert sent == []


def test_same_side_top_up_places_delta_only(monkeypatch):
    sent = []
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, volume=1.0, ticket=77)],
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
            retcode=mt5.TRADE_RETCODE_PLACED, order=1, deal=0, comment="placed"
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(
        symbol="#US30",
        side=Signal.LONG,
        volume=1.5,
        sl=39500.0,
        rebalance_band=0.15,
    )

    assert result.ok is True
    assert result.action == "top_up"
    assert len(sent) == 1
    assert sent[0]["action"] == mt5.TRADE_ACTION_PENDING
    assert sent[0]["volume"] == 0.5


def test_same_side_trim_partial_closes_excess(monkeypatch):
    sent = []
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, volume=1.0, ticket=77)],
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
            retcode=mt5.TRADE_RETCODE_DONE, order=1, deal=2, comment="done"
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(
        symbol="#US30", side=Signal.LONG, volume=0.5, rebalance_band=0.15
    )

    assert result.ok is True
    assert result.action == "trim"
    assert len(sent) == 1
    assert sent[0]["action"] == mt5.TRADE_ACTION_DEAL
    assert sent[0]["volume"] == 0.5
    assert sent[0]["position"] == 77


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
    assert sent[0]["type_filling"] == mt5.ORDER_FILLING_IOC
    assert sent[1]["action"] == mt5.TRADE_ACTION_PENDING
    assert sent[1]["type"] == mt5.ORDER_TYPE_SELL_LIMIT
    assert sent[1]["type_filling"] == mt5.ORDER_FILLING_RETURN
    assert "awaiting fill" in result.message


def test_place_limit_includes_stop_loss(monkeypatch):
    sent = []
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [])
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
    result = executor.reconcile_target(
        symbol="#US30",
        side=Signal.LONG,
        volume=1.0,
        sl=39500.0,
    )

    assert result.ok is True
    assert sent[-1]["sl"] == 39500.0
    assert sent[-1]["type_filling"] == mt5.ORDER_FILLING_RETURN


def test_pending_ignores_symbol_ioc_filling_flag():
    info = SimpleNamespace(filling_mode=2)  # SYMBOL_FILLING_IOC
    assert OrderExecutor._pending_filling_mode() == mt5.ORDER_FILLING_RETURN
    assert OrderExecutor._deal_filling_mode(info) == mt5.ORDER_FILLING_IOC


def test_send_treats_placed_as_success(monkeypatch):
    monkeypatch.setattr(
        mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_PLACED,
            order=55,
            deal=0,
            comment="placed",
        ),
    )
    result = OrderExecutor._send(
        {"action": mt5.TRADE_ACTION_PENDING, "symbol": "#US30"},
        "order",
    )
    assert result.ok is True
    assert result.retcode == mt5.TRADE_RETCODE_PLACED
    assert result.order == 55


def test_send_treats_done_partial_as_success(monkeypatch):
    monkeypatch.setattr(
        mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE_PARTIAL,
            order=1,
            deal=2,
            comment="partial",
        ),
    )
    result = OrderExecutor._send({"action": mt5.TRADE_ACTION_DEAL}, "close")
    assert result.ok is True
    assert result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL


def test_place_limit_ok_when_retcode_placed(monkeypatch):
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [])
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )
    monkeypatch.setattr(
        mt5,
        "order_send",
        lambda request: SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_PLACED,
            order=99,
            deal=0,
            comment="Request placed",
        ),
    )
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.LONG, volume=1.0, sl=39500.0)
    assert result.ok is True
    assert result.action == "open"
    assert result.orders[-1].retcode == mt5.TRADE_RETCODE_PLACED


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


def test_cancel_failure_blocks_flatten_before_close(monkeypatch):
    sent = []
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [_position(Signal.LONG)])
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )

    def order_send(request):
        sent.append(request)
        if request.get("action") == mt5.TRADE_ACTION_REMOVE:
            return SimpleNamespace(retcode=10013, comment="reject", order=0, deal=0)
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=1,
            deal=2,
            comment="done",
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.FLAT, volume=0.0)

    assert result.ok is False
    assert result.action == "cancel_failed"
    assert any(r.get("action") == mt5.TRADE_ACTION_REMOVE for r in sent)
    assert not any(r.get("action") == mt5.TRADE_ACTION_DEAL for r in sent)


def test_cancel_failure_blocks_reverse_entry(monkeypatch):
    sent = []
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    monkeypatch.setattr(
        mt5,
        "positions_get",
        lambda **kwargs: [_position(Signal.LONG, ticket=77)],
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )

    def order_send(request):
        sent.append(request)
        if request.get("action") == mt5.TRADE_ACTION_REMOVE:
            return SimpleNamespace(retcode=10013, comment="reject", order=0, deal=0)
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=1,
            deal=2,
            comment="done",
        )

    monkeypatch.setattr(mt5, "order_send", order_send)
    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.SHORT, volume=1.0)

    assert result.ok is False
    assert result.action == "cancel_failed"
    assert not any(r.get("action") == mt5.TRADE_ACTION_PENDING for r in sent)


def test_close_all_aborts_when_pending_remain(monkeypatch):
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    monkeypatch.setattr(mt5, "orders_get", lambda **kwargs: [pending])
    monkeypatch.setattr(mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(
        mt5,
        "order_send",
        lambda request: SimpleNamespace(retcode=10013, comment="reject", order=0, deal=0),
    )

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.close_all("#US30")

    assert result.ok is False
    assert "cancel_failed" in result.message


def test_successful_cancel_then_flat_requires_zero_pending(monkeypatch):
    pending = SimpleNamespace(
        ticket=9,
        magic=260717,
        type=mt5.ORDER_TYPE_BUY_LIMIT,
        volume_current=1.0,
    )
    book = {"orders": [pending], "positions": [_position(Signal.LONG, ticket=77)]}
    sent = []

    def orders_get(**kwargs):
        return list(book["orders"])

    def positions_get(**kwargs):
        return list(book["positions"])

    def order_send(request):
        sent.append(request)
        if request.get("action") == mt5.TRADE_ACTION_REMOVE:
            book["orders"] = []
            return SimpleNamespace(
                retcode=mt5.TRADE_RETCODE_DONE, order=1, deal=0, comment="done"
            )
        if request.get("action") == mt5.TRADE_ACTION_DEAL:
            book["positions"] = []
            return SimpleNamespace(
                retcode=mt5.TRADE_RETCODE_DONE, order=2, deal=3, comment="done"
            )
        return SimpleNamespace(retcode=10013, comment="unexpected", order=0, deal=0)

    monkeypatch.setattr(mt5, "orders_get", orders_get)
    monkeypatch.setattr(mt5, "positions_get", positions_get)
    monkeypatch.setattr(mt5, "order_send", order_send)
    monkeypatch.setattr(
        mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=40000.0, ask=40001.0),
    )

    executor = OrderExecutor(Connection(), execute=True, account_type="demo")
    result = executor.reconcile_target(symbol="#US30", side=Signal.FLAT, volume=0.0)

    assert result.ok is True
    assert result.action == "flatten"
    assert "positions=0, pending=0" in result.message
    assert book["orders"] == []
    assert book["positions"] == []

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import MetaTrader5 as mt5

from src.connection import MT5Connection, MT5InvokeTimeout


def test_invoke_serializes_calls_across_threads(monkeypatch):
    """Kill-switch and trading loop must not interleave MT5 API work."""
    conn = MT5Connection(invoke_timeout_sec=5.0)
    conn.start()
    active = 0
    max_active = 0
    lock = threading.Lock()
    events = []

    def slow_op(label, hold=0.05):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            events.append(("enter", label, threading.current_thread().name))
        time.sleep(hold)
        with lock:
            active -= 1
            events.append(("exit", label, threading.current_thread().name))
        return label

    monkeypatch.setattr(mt5, "terminal_info", lambda: SimpleNamespace())
    monkeypatch.setattr(mt5, "shutdown", lambda: None)
    monkeypatch.setattr(mt5, "initialize", lambda **kwargs: True)
    monkeypatch.setattr(
        mt5,
        "account_info",
        lambda: SimpleNamespace(login=1, server="t", balance=1000.0),
    )

    results = [None, None]
    errors = []

    def worker(idx, label):
        try:
            results[idx] = conn.invoke(slow_op, label)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(0, "trade"))
    t2 = threading.Thread(target=worker, args=(1, "kill"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    conn.stop()

    assert errors == []
    assert set(results) == {"trade", "kill"}
    assert max_active == 1
    # All work runs on the MT5 worker thread.
    assert all(name == "MT5Worker" for _, _, name in events)


def test_connect_runs_shutdown_before_initialize_on_worker(monkeypatch):
    conn = MT5Connection()
    order = []

    monkeypatch.setattr(mt5, "shutdown", lambda: order.append("shutdown"))
    monkeypatch.setattr(
        mt5,
        "initialize",
        lambda **kwargs: order.append("initialize") or True,
    )
    monkeypatch.setattr(
        mt5,
        "account_info",
        lambda: SimpleNamespace(login=1, server="t", balance=1.0),
    )
    monkeypatch.setattr(mt5, "terminal_info", lambda: SimpleNamespace())

    assert conn.connect() is True
    assert order[:2] == ["shutdown", "initialize"]
    conn.shutdown()


def test_ensure_reconnect_blocks_other_invokes(monkeypatch):
    conn = MT5Connection(reconnect_attempts=2, reconnect_delay_sec=0.05)
    phase = {"n": 0}
    order: list[str] = []

    def terminal_info():
        return None if phase["n"] == 0 else SimpleNamespace()

    def initialize(**kwargs):
        order.append("init_start")
        time.sleep(0.08)
        phase["n"] += 1
        order.append("init_end")
        return True

    monkeypatch.setattr(mt5, "terminal_info", terminal_info)
    monkeypatch.setattr(mt5, "shutdown", lambda: None)
    monkeypatch.setattr(mt5, "initialize", initialize)
    monkeypatch.setattr(
        mt5,
        "account_info",
        lambda: SimpleNamespace(login=1, server="t", balance=1.0),
    )

    conn._connected = True
    done = []

    def other():
        done.append(conn.invoke(lambda: order.append("other_run") or "ok"))

    t_ensure = threading.Thread(target=lambda: conn.ensure())
    t_other = threading.Thread(target=other)
    t_ensure.start()
    time.sleep(0.01)
    t_other.start()
    t_ensure.join(timeout=5)
    t_other.join(timeout=5)
    conn.shutdown()

    assert done == ["ok"]
    assert "init_start" in order and "init_end" in order and "other_run" in order
    assert order.index("other_run") > order.index("init_end")


def test_invoke_timeout_abandons_queued_job(monkeypatch):
    """Timed-out jobs still in the queue must not execute (no late order_send)."""
    conn = MT5Connection(invoke_timeout_sec=0.05)
    conn.start()
    started = threading.Event()
    release = threading.Event()
    ran_second = []

    def blocker():
        started.set()
        release.wait(timeout=2.0)
        return "blocked"

    def second():
        ran_second.append("ran")
        return "second"

    monkeypatch.setattr(mt5, "terminal_info", lambda: SimpleNamespace())
    monkeypatch.setattr(mt5, "shutdown", lambda: None)

    err_block = []

    def run_block():
        try:
            conn.invoke(blocker)
        except MT5InvokeTimeout as exc:
            err_block.append(exc)

    t_block = threading.Thread(target=run_block)
    t_block.start()
    assert started.wait(timeout=1.0)

    err = None

    def call_second():
        nonlocal err
        try:
            conn.invoke(second)
        except Exception as exc:  # noqa: BLE001
            err = exc

    t_second = threading.Thread(target=call_second)
    t_second.start()
    t_second.join(timeout=2.0)
    release.set()
    t_block.join(timeout=2.0)
    # Allow worker to observe abandoned flag on the queued second job.
    time.sleep(0.1)
    conn.stop()

    assert isinstance(err, MT5InvokeTimeout)
    assert ran_second == []
    assert err_block  # blocker itself also timed out while holding the worker


def test_invoke_timeout_marks_in_flight_abandoned(monkeypatch):
    conn = MT5Connection(invoke_timeout_sec=0.05)
    conn.start()
    entered = threading.Event()
    release = threading.Event()

    def slow():
        entered.set()
        release.wait(timeout=2.0)
        return "late"

    monkeypatch.setattr(mt5, "terminal_info", lambda: SimpleNamespace())

    err = None
    try:
        conn.invoke(slow)
    except Exception as exc:  # noqa: BLE001
        err = exc

    assert entered.wait(timeout=1.0)
    assert isinstance(err, MT5InvokeTimeout)
    release.set()
    time.sleep(0.1)
    conn.stop()

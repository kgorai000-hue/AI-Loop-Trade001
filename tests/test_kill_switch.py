from __future__ import annotations

from types import SimpleNamespace

from src.execution import CancelPendingResult, OrderResult
from src.kill_switch import KillPhase, KillSwitchMonitor


class Store:
    def __init__(self, locked=False, equity_peak=1000.0):
        self.state = {"locked": locked, "equity_peak": equity_peak}
        self.lock_calls = []

    def is_locked(self):
        return bool(self.state.get("locked"))

    def set_locked(self, locked, reason=""):
        self.lock_calls.append((locked, reason))
        self.state["locked"] = bool(locked)

    def read_state(self):
        return dict(self.state)

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def read_skills(self):
        return []


class Connection:
    def ensure(self):
        return True

    def account_info(self):
        return SimpleNamespace(equity=800.0, margin=0.0)


class Executor:
    def __init__(self):
        self.pending = []
        self.positions = []
        self.cancel_results = []
        self.close_results = []
        self.cancel_calls = 0
        self.close_calls = 0

    def cancel_pending(self, symbol, magic=0):
        self.cancel_calls += 1
        if self.cancel_results:
            return self.cancel_results.pop(0)
        self.pending = []
        return CancelPendingResult(ok=True, message="cancelled")

    def managed_pending(self, symbol, magic=0):
        return list(self.pending)

    def managed_positions(self, symbol, magic=0):
        return list(self.positions)

    def close_position_market(self, position, magic=260717):
        self.close_calls += 1
        if self.close_results:
            return self.close_results.pop(0)
        self.positions = [p for p in self.positions if p is not position]
        return OrderResult(ok=True, message="closed")


def _monitor(executor=None, store=None, on_trigger=None):
    store = store or Store()
    executor = executor or Executor()
    return (
        KillSwitchMonitor(
            connection=Connection(),
            executor=executor,
            state_stores=[store],
            symbols=["#US30"],
            max_drawdown=0.10,
            poll_seconds=15,
            on_trigger=on_trigger,
        ),
        store,
        executor,
    )


def test_drawdown_runs_state_machine_to_locked_and_flat():
    monitor, store, executor = _monitor()
    executor.positions = [SimpleNamespace(ticket=1, symbol="#US30")]
    executor.pending = [SimpleNamespace(ticket=9)]

    executor.cancel_results = [
        CancelPendingResult(
            ok=False, message="reject", attempted=[9], failed=[9], remaining=[9]
        ),
        CancelPendingResult(ok=True, cancelled=[9]),
    ]

    monitor._tick()
    assert store.is_locked() is True
    assert monitor.phase == KillPhase.CANCEL_PENDING
    assert monitor.triggered is True

    executor.pending = []
    monitor._tick()
    assert monitor.phase == KillPhase.LOCKED_AND_FLAT
    assert executor.close_calls >= 1
    assert store.is_locked() is True


def test_cancel_failure_retries_without_claiming_flat():
    monitor, store, executor = _monitor()
    executor.pending = [SimpleNamespace(ticket=9)]
    executor.cancel_results = [
        CancelPendingResult(ok=False, message="busy", remaining=[9]),
        CancelPendingResult(ok=False, message="busy", remaining=[9]),
    ]

    monitor._enter_triggered("test dd")
    assert monitor.phase == KillPhase.CANCEL_PENDING
    assert executor.close_calls == 0

    monitor._tick()
    assert monitor.phase == KillPhase.CANCEL_PENDING
    assert monitor.phase != KillPhase.LOCKED_AND_FLAT


def test_resume_from_locked_state_continues_flatten():
    store = Store(locked=True)
    executor = Executor()
    executor.positions = [SimpleNamespace(ticket=5, symbol="#US30")]
    monitor, _, _ = _monitor(executor=executor, store=store)

    monitor._bootstrap_from_locked_state()
    assert monitor.phase == KillPhase.CANCEL_PENDING

    monitor._advance_flatten()
    assert monitor.phase == KillPhase.LOCKED_AND_FLAT
    assert executor.positions == []


def test_verify_flat_restarts_cancel_if_pending_reappears():
    monitor, store, executor = _monitor()
    monitor._reason = "dd"
    monitor._phase = KillPhase.VERIFY_FLAT
    executor.positions = []
    executor.pending = [SimpleNamespace(ticket=3)]

    monitor._advance_flatten()
    assert monitor.phase == KillPhase.CANCEL_PENDING


def test_locked_and_flat_stays_dormant_while_state_locked():
    monitor, store, _ = _monitor(store=Store(locked=True, equity_peak=1000.0))
    monitor._phase = KillPhase.LOCKED_AND_FLAT
    monitor._reason = "prior dd"
    monitor._on_trigger_fired = True

    monitor._tick()
    assert monitor.phase == KillPhase.LOCKED_AND_FLAT
    assert store.is_locked() is True


def test_manual_unlock_resumes_kill_switch_monitoring():
    """STATE.locked=false must restart DD monitoring (not leave KS dead)."""
    store = Store(locked=True, equity_peak=1000.0)
    monitor, store, executor = _monitor(store=store)
    monitor.connection = SimpleNamespace(
        ensure=lambda: True,
        account_info=lambda: SimpleNamespace(equity=1000.0, margin=0.0),
    )
    monitor._phase = KillPhase.LOCKED_AND_FLAT
    monitor._reason = "prior dd"
    monitor._on_trigger_fired = True
    monitor._equity_peak = 1000.0

    store.state["locked"] = False
    monitor._tick()

    assert monitor.phase == KillPhase.IDLE
    assert monitor.triggered is False
    assert monitor._on_trigger_fired is False
    assert store.state.get("equity") == 1000.0


def test_manual_unlock_retriggers_if_drawdown_still_breached():
    store = Store(locked=True, equity_peak=1000.0)
    monitor, store, executor = _monitor(store=store)
    # Default Connection equity=800 -> 20% DD >= 10% max
    monitor._phase = KillPhase.LOCKED_AND_FLAT
    monitor._reason = "prior dd"
    monitor._on_trigger_fired = True
    monitor._equity_peak = 1000.0

    store.state["locked"] = False
    monitor._tick()

    assert store.is_locked() is True
    assert monitor.phase != KillPhase.IDLE
    assert monitor.triggered is True

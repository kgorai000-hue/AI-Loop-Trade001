"""Account drawdown kill-switch monitor (daemon thread)."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class KillPhase(str, Enum):
    """Flatten state machine; advances until LOCKED_AND_FLAT."""

    IDLE = "idle"
    TRIGGERED = "triggered"
    BLOCK_NEW_ENTRIES = "block_new_entries"
    CANCEL_PENDING = "cancel_pending"
    VERIFY_NO_PENDING = "verify_no_pending"
    CLOSE_POSITIONS = "close_positions"
    VERIFY_FLAT = "verify_flat"
    LOCKED_AND_FLAT = "locked_and_flat"


class KillSwitchMonitor:
    """
    Poll account equity; on drawdown breach (or locked STATE at startup),
    run a flatten state machine until positions and pending are both zero.

    Unlock is manual only (STATE.locked = false). When every STATE is unlocked
    after ``LOCKED_AND_FLAT``, the monitor returns to IDLE and resumes DD checks
    so trading never runs without kill-switch coverage.
    """

    def __init__(
        self,
        connection: Any,
        executor: Any,
        state_stores: list[Any],
        symbols: list[str],
        max_drawdown: float = 0.10,
        poll_seconds: float = 15.0,
        on_trigger: Optional[Callable[[str], None]] = None,
        magic: int = 260717,
    ) -> None:
        self.connection = connection
        self.executor = executor
        self.state_stores = state_stores
        self.symbols = symbols
        self.max_drawdown = float(max_drawdown)
        self.poll_seconds = float(poll_seconds)
        self.on_trigger = on_trigger
        self.magic = int(magic)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._phase = KillPhase.IDLE
        self._reason = ""
        self._on_trigger_fired = False
        self._equity_peak: Optional[float] = None

    @property
    def phase(self) -> KillPhase:
        return self._phase

    @property
    def triggered(self) -> bool:
        return self._phase != KillPhase.IDLE

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._bootstrap_from_locked_state()
        self._thread = threading.Thread(
            target=self._run,
            name="KillSwitchMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "KillSwitchMonitor started (max_dd=%.2f%% poll=%ss phase=%s)",
            self.max_drawdown * 100,
            self.poll_seconds,
            self._phase.value,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_seconds + 2)
        self._thread = None

    def _any_locked(self) -> bool:
        return any(s.is_locked() for s in self.state_stores)

    def _lock_reason_from_state(self) -> str:
        for store in self.state_stores:
            lessons = getattr(store, "read_skills", None)
            if callable(lessons):
                for lesson in lessons():
                    if "Kill-switch locked:" in str(lesson):
                        return str(lesson).split("Kill-switch locked:", 1)[-1].strip()
        return "resume from locked STATE"

    def _bootstrap_from_locked_state(self) -> None:
        """If STATE is already locked, resume flatten instead of idling."""
        if not self._any_locked():
            return
        if self._phase == KillPhase.LOCKED_AND_FLAT:
            return
        self._reason = self._lock_reason_from_state()
        self._phase = KillPhase.CANCEL_PENDING
        self._on_trigger_fired = True
        logger.critical(
            "KILL SWITCH resume: locked STATE detected -> phase=%s (%s)",
            self._phase.value,
            self._reason,
        )

    def _resume_after_manual_unlock(self) -> None:
        """Return to IDLE when operator clears STATE.locked after LOCKED_AND_FLAT."""
        logger.warning(
            "KILL SWITCH resumed after manual unlock (was LOCKED_AND_FLAT); "
            "equity drawdown monitoring active again"
        )
        self._phase = KillPhase.IDLE
        self._reason = ""
        self._on_trigger_fired = False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("KillSwitchMonitor tick failed")
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        if self._phase == KillPhase.LOCKED_AND_FLAT:
            # Stay dormant only while STATE remains locked. Manual unlock must
            # resume monitoring -- otherwise the trade loop can run unprotected.
            if self._any_locked():
                return
            self._resume_after_manual_unlock()

        if self._phase != KillPhase.IDLE:
            self._advance_flatten()
            return

        if self._any_locked():
            self._bootstrap_from_locked_state()
            self._advance_flatten()
            return

        if not self.connection.ensure():
            return

        info = self.connection.account_info()
        if info is None:
            return

        equity = float(info.equity)
        margin = float(getattr(info, "margin", 0.0) or 0.0)

        if self._equity_peak is None:
            peaks = []
            for store in self.state_stores:
                st = store.read_state()
                ep = st.get("equity_peak")
                if ep is not None:
                    try:
                        peaks.append(float(ep))
                    except (TypeError, ValueError):
                        pass
            self._equity_peak = max(peaks) if peaks else equity

        if equity > self._equity_peak:
            self._equity_peak = equity

        for store in self.state_stores:
            store.update_state(equity=equity, margin=margin, equity_peak=self._equity_peak)

        if self._equity_peak <= 0:
            return

        dd = (self._equity_peak - equity) / self._equity_peak
        if dd < self.max_drawdown:
            return

        reason = (
            f"equity drawdown {dd:.2%} >= {self.max_drawdown:.2%} "
            f"(equity={equity:.2f} peak={self._equity_peak:.2f})"
        )
        self._enter_triggered(reason)

    def _enter_triggered(self, reason: str) -> None:
        logger.critical("KILL SWITCH: %s", reason)
        self._reason = reason
        self._phase = KillPhase.TRIGGERED
        self._advance_flatten()

    def _advance_flatten(self) -> None:
        """Progress through flatten phases; stop on failure and retry next poll."""
        while self._phase not in (KillPhase.IDLE, KillPhase.LOCKED_AND_FLAT):
            prev = self._phase
            if self._phase == KillPhase.TRIGGERED:
                self._phase = KillPhase.BLOCK_NEW_ENTRIES
            elif self._phase == KillPhase.BLOCK_NEW_ENTRIES:
                self._block_new_entries()
                self._phase = KillPhase.CANCEL_PENDING
            elif self._phase == KillPhase.CANCEL_PENDING:
                if not self._cancel_all_pending():
                    logger.critical(
                        "KILL SWITCH retry pending cancel next poll (phase=%s)",
                        self._phase.value,
                    )
                    return
                self._phase = KillPhase.VERIFY_NO_PENDING
            elif self._phase == KillPhase.VERIFY_NO_PENDING:
                if not self._verify_no_pending():
                    self._phase = KillPhase.CANCEL_PENDING
                    logger.critical("KILL SWITCH pending remain -> re-cancel next poll")
                    return
                self._phase = KillPhase.CLOSE_POSITIONS
            elif self._phase == KillPhase.CLOSE_POSITIONS:
                if not self._close_all_positions():
                    logger.critical(
                        "KILL SWITCH retry position close next poll (phase=%s)",
                        self._phase.value,
                    )
                    return
                self._phase = KillPhase.VERIFY_FLAT
            elif self._phase == KillPhase.VERIFY_FLAT:
                if not self._verify_flat():
                    if not self._verify_no_pending():
                        self._phase = KillPhase.CANCEL_PENDING
                    else:
                        self._phase = KillPhase.CLOSE_POSITIONS
                    logger.critical(
                        "KILL SWITCH not flat -> resume at %s next poll",
                        self._phase.value,
                    )
                    return
                self._phase = KillPhase.LOCKED_AND_FLAT
                logger.critical(
                    "KILL SWITCH LOCKED_AND_FLAT (%s) symbols=%s",
                    self._reason,
                    self.symbols,
                )
                return
            else:
                logger.error("Unknown kill phase %s", self._phase)
                return

            if self._phase == prev:
                return

    def _block_new_entries(self) -> None:
        for store in self.state_stores:
            if not store.is_locked():
                store.set_locked(True, reason=self._reason)
        if self.on_trigger and not self._on_trigger_fired:
            self._on_trigger_fired = True
            try:
                self.on_trigger(self._reason)
            except Exception:
                logger.exception("on_trigger callback failed")

    def _cancel_all_pending(self) -> bool:
        if not self.connection.ensure():
            logger.critical("KILL SWITCH cancel aborted: MT5 not connected")
            return False
        ok = True
        for symbol in self.symbols:
            try:
                result = self.executor.cancel_pending(symbol, magic=0)
            except Exception:
                logger.exception("KILL SWITCH cancel raised for %s", symbol)
                return False
            if getattr(result, "fetch_failed", False) or not getattr(result, "ok", False):
                logger.critical(
                    "KILL SWITCH cancel failed %s -> %s",
                    symbol,
                    getattr(result, "message", result),
                )
                ok = False
        return ok

    def _verify_no_pending(self) -> bool:
        if not self.connection.ensure():
            return False
        for symbol in self.symbols:
            pending = self.executor.managed_pending(symbol, magic=0)
            if pending is None:
                logger.critical("KILL SWITCH orders_get failed for %s", symbol)
                return False
            if pending:
                tickets = [int(getattr(o, "ticket", 0) or 0) for o in pending]
                logger.critical(
                    "KILL SWITCH pending remain %s tickets=%s", symbol, tickets
                )
                return False
        return True

    def _close_all_positions(self) -> bool:
        if not self.connection.ensure():
            logger.critical("KILL SWITCH close aborted: MT5 not connected")
            return False
        ok = True
        for symbol in self.symbols:
            positions = self.executor.managed_positions(symbol, magic=0)
            if positions is None:
                logger.critical("KILL SWITCH positions_get failed for %s", symbol)
                return False
            for position in positions:
                try:
                    result = self.executor.close_position_market(
                        position, magic=self.magic
                    )
                except Exception:
                    logger.exception(
                        "KILL SWITCH close raised for %s ticket=%s",
                        symbol,
                        getattr(position, "ticket", None),
                    )
                    return False
                if not result.ok:
                    logger.critical(
                        "KILL SWITCH close failed %s ticket=%s -> %s",
                        symbol,
                        getattr(position, "ticket", None),
                        result.message,
                    )
                    ok = False
                elif result.dry_run:
                    logger.critical(
                        "KILL SWITCH close dry-run %s ticket=%s (EXECUTE blocked)",
                        symbol,
                        getattr(position, "ticket", None),
                    )
                    ok = False
        return ok

    def _verify_flat(self) -> bool:
        if not self.connection.ensure():
            return False
        for symbol in self.symbols:
            positions = self.executor.managed_positions(symbol, magic=0)
            pending = self.executor.managed_pending(symbol, magic=0)
            if positions is None or pending is None:
                logger.critical("KILL SWITCH flat verify fetch failed for %s", symbol)
                return False
            if positions or pending:
                logger.critical(
                    "KILL SWITCH not flat %s positions=%s pending=%s",
                    symbol,
                    len(positions),
                    len(pending),
                )
                return False
        return True
